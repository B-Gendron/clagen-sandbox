import os
from owlready2 import *
from termcolor import colored
from pprint import pprint
from datetime import datetime
import torch
import numpy as np
import logging
from datasets import Dataset, DatasetDict
from collections import Counter
import json
import re
import random
from math import sqrt
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_auc_score, f1_score
from nltk.tokenize import TweetTokenizer
tokenizer = TweetTokenizer()


# -----------------------------------------------------------------------------------------
# General purpose auxiliary functions
# -----------------------------------------------------------------------------------------

def get_datetime():
    '''
        This function gets the current date time and returns it as a string.

        @returns dt_string (str): current time formatted in a string
    '''
    now = datetime.now()
    dt_string = now.strftime("%d%m%Y%H%M%S")
    return dt_string


def args2filename(dico):
    '''
        This function builds a file name regarding the parameters of the experiment given in a dictionnary

        @param dico (dict): a parameters dictionnary

        @returns filename (str): a string to name a file regarding the parameters nature and values
    '''
    filename = "_".join([f"{k}{v}" for k,v in dico.items()])
    return filename


def custom_flatten(ll):
    '''
        A function to flatten a list of lists where sub lists are of heterogeneous sizes.
        @param ll (list): the input list of lists
        @return l (list): the flattened list   
    '''
    l = []
    for sl in ll:
        if type(sl) is not list:
            sl = [sl]
        l.extend(sl)
    return l


def custom_flatten_sep(ll):
    '''
        A function to flatten a list of lists where sub lists are of heterogeneous sizes. This variant of the above function keeps a track of where the sublists have been concatenated by adding a [SEP] marker as a float 0.0

        @param ll (list): the input list of lists

        @return l (list): the flattened list   
    '''
    l = []
    for sl in ll:
        l.extend(sl + [0.0])
    return l[:len(l)-1]


def concatenate(iterable, sep=""):
    sentence = iterable[0]
    for word in iterable[1:]:
        sentence += (sep + word)
    return sentence


def activate_gpu(force_cpu=False):
    '''
        A function to return the right device depending on GPU availability
    '''
    device = "cpu"
    if not force_cpu:
        if torch.cuda.is_available():
            device = 'cuda'
            print('DEVICE = ', colored(torch.cuda.get_device_name(0), "green" ) )
        elif torch.backends.mps.is_available():
            device = 'mps'
            print('DEVICE = ', colored("mps", "green" ) )
        else:
            device = 'cpu'
            print('DEVICE = ', colored('CPU', "blue"))
    return device


def set_logger(target):
    '''
        Set a logger to be used when running experiments. 
    '''
    experiment = f'{target}_{get_datetime()}'
    logging.getLogger(__name__).setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)-8s - %(message)s')
    fh = logging.FileHandler(f'./logs/{experiment}.log', mode="w")
    fh.setFormatter(formatter)
    fh.setLevel(logging.INFO)
    logging.getLogger().addHandler(fh)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logging.debug(f"Experiment name : {experiment}")
    return logger


# -----------------------------------------------------------------------------------------
# Data management
# -----------------------------------------------------------------------------------------

def load_dataset(dataset_name):
    '''
        Load a dataset save in the huggingface format, located in the data folder.

        @param dataset_name (str): the name of the dataset, i.e. the associated folder inside the data folder. This string should correspond to an existing dataset in the folder, otherwise an exception will be raised.

        @return dataset (DatasetDict): the dataset as a DatasetDict object.
    '''
    path = f"./data/{dataset_name}"
    if os.path.isdir(path):
        dataset = DatasetDict.load_from_disk(f"../data/{dataset_name}")
    else: 
        raise Exception("No folder with the such name found in the data folder.")

    return dataset


def save_dataset(dataset, dataset_name, output_format='huggingface'):
    '''
        Save the dataset into a HuggingFace format using the save_to_disk method.

        @param dataset (DatasetDict):      the dataset to save in a DatasetDict format
        @param dataset_name (str):         the name to give to the file
        @param output_format (str):        either 'huggingface' or 'json'. Default is 'huggingface'.
    '''
    if output_format == "huggingface":
        dataset.save_to_disk(f'../data/{dataset_name}')
    elif output_format == "json":
        with open(f"data/{dataset_name}.json", 'w') as f:
            json.dump(dataset, f)
    else:
        print("The given output format is not recognized. Please note that accepted formats are 'huggingface' and 'json")


# -----------------------------------------------------------------------------------------
# Training precedure utils
# -----------------------------------------------------------------------------------------

def get_labels_weights(training_set, device, num_labels=7, index=0, penalty=1e9, apply_penalty=True):
    '''
        @param training_set (DatsetDict): the training samples to compute the weights on. Can be the whole test set or just a batch
        @param device
        @param num_labels (int): the number of labels for multiclass classification. Default is the number of emotion labels in dailydialog
        @param index (int): the index of the label to penalize
        @param penalty (float): the magnitude of the penalty
    
    '''
    labels = training_set['labels']
    n = labels.size()[0]
    labels_list = labels[labels != -1].flatten().tolist()
    percentages = { l:Counter(labels_list)[l] / float(n) for l in range(num_labels) }
    weights = [ 1-percentages[l] if l in percentages else 0.0 for l in list(range(num_labels)) ]
    # weights[0] = 0.01 # almost remove the non emotion class
    weights = torch.tensor(weights, device=device)

    # add further penalty to no_emotion class
    if apply_penalty:
        weights[index] = weights[index]/penalty

    # save tensor
    # torch.save(weights, 'classes_weights.pt')
    return weights


def vocab_dicts(vocab):
    stoi, itos = vocab.get_stoi(), vocab.get_itos()

    # dump stoi dict
    with open("../objects/vocab_stoi.json", "w") as f:
        json.dump(stoi, f)

    # dump itos dict
    with open("../objects/vocab_itos.json", "w") as f:
        json.dump(itos, f)

    # return dicts to be used in training (encode function)
    return stoi, itos

# -----------------------------------------------------------------------------------------
# Preprocessing utils
# -----------------------------------------------------------------------------------------

def add_sentencebert_random_vectors(embedding, size, max_length):
    '''
        This auxiliary function is to be used for Sentence BERT preprocessing. It adds the number of max_length-sized random vectors defined by the parameter size.

        @param size (int):              the number of random vectors to generate.

        @return vectors (list):         the list of length size of generated random vectors.
    '''
    # DescribeResult(nobs=3840, minmax=(-0.20482617616653442, 0.17243704199790955), mean=0.00021893562853630052, variance=0.0026047970850628303, skewness=-0.06372909078235393, kurtosis=-0.001260392396507104)
    mean = 0.00021893562853630052
    variance = 0.0026047970850628303
    for _ in range(size):
        embedding.append([random.gauss(mu=mean, sigma=sqrt(variance)) for _ in range(max_length)])
    return embedding

def encode(stoi, text):
    '''
        From token strings to indexes

        @param stoi (dict):             string to index mapping from the vocab
        @param text (str):              the text to encode

        @return idxes (list of int):    the list of indexes that correspond to the text encoding according to the underlying vocab
    '''
    encoding = []
    for token in text:
        if token not in (' ', '\n'):
            try:
                stoi[token.lower()]
            except KeyError:
                encoding.append(stoi['<unk>']) 
            else:
                encoding.append(stoi[token.lower()])
                
    return encoding

def decode(idxes, itos):
    '''
        From vocab indexes to token strings

        @param idexes (list of int): the list of indexes to be mapped to their associated tokens
        @param itos (dict): index to string mapping from the vocab

        @return tokens (str): the concatenated tokens that form the encoded sentence
    '''
    return ' '.join([itos[i] for i in idxes])


def load_vocab_mappings():
    with open("../objects/vocab_itos.json") as f:
        itos = f.read()

    with open("../objects/vocab_stoi.json") as f:
        stoi = f.read()
    stoi = json.loads(stoi)

    return itos, stoi

# -----------------------------------------------------------------------------------------
# Prompt utils
# -----------------------------------------------------------------------------------------

def parse_indexes(levels_dict):
    '''
        This auxiliary function parses and retrieves the utterance indexes from the readability level dictionnary when an utterance is reffered to using the ontology identifier. It return a dictionary with the same format than the input, containing only the indexes instead of the complete identifier.

        @param levels_dict (dict):          the dict from the ontology individual search.
        
        @return levels_indexes_dict (dict): a dict that maps the readability levels to the associated utterance indexes in the current dialog.
    '''
    levels_indexes_dict = {}

    # iterate through the input dict
    for k in levels_dict.keys():
        for v in levels_dict[k]:
            # parse the utterance full name to retrieve its index
            splitted_v = v.split('_')
            index = splitted_v[2] # the 3rd element when individual_592.592_598_utt_21271711 is splitted by _ is 598, which is the utterance index
            # store the readability class for the current index
            levels_indexes_dict[int(index)] = k

    return dict(sorted(levels_indexes_dict.items()))


def get_prompt_and_label(dialog_file, split, stoi, onto_path="../../../OntoUttPreprocessing/rdf/wikitalk"):
    '''
        This function outputs a prompt encoded with respect to the right vocab, containing the dialog stored in the .json file dialog_file.

        @param dialog_file (str):       the name of the (temp) json file where the dialog information is stored
        @param split (str):             the dataset split from which the dialog comes from
        @param stoi (dict):             the string to index mapping from the pretraining vocabulary
        @param onto_path (str):         
    '''
    # get labels mapping for different readability level
    readability_levels_mapping = {'EasilyReadableText':0, 'StandardReadableText':1, 'HardlyreadableText':2}

    # load the encoded dialog file
    with open(f"../objects/{dialog_file}.json", 'r') as f:
        dial_descr = json.load(f)

    dial_id = dial_descr['dial_id']
    dial_enc = dial_descr['dial_encoding']

    # retrieve the corresponding ontology individual
    indiv_path = os.path.join(onto_path, split, f'individual_{dial_id}.rdf')
    individual = get_ontology(indiv_path).load()
    # crop first element as it simply corresponds to the class occurence
    utterance_levels = {
        'EasilyReadableText':       [str(i) for i in individual.search(is_a=individual.EasilyReadableText)[1:]],
        'StandardReadableText':     [str(i) for i in individual.search(is_a=individual.StandardReadableText)[1:]],
        'HardlyReadableText':       [str(i) for i in individual.search(is_a=individual.HardlyReadableText)[1:]]
    }
    utterance_levels = parse_indexes(utterance_levels)
    idx = 0
    for k in utterance_levels.keys():
        # concatenate the readability level information to the prompt
        readability_info = encode(stoi, tokenizer.tokenize(f"(ReadabilityLevel: {utterance_levels[k]})"))
        dial_enc[idx].extend(readability_info)
        idx += 1

    # remove last utterance
    del dial_enc[idx-1]
    # deduce soft prompt by dropping one dimension
    soft_prompt = custom_flatten(dial_enc)

    # retrieve label which is the readability level of the last utterance
    last_utterance_index = max(utterance_levels.keys())
    label = readability_levels_mapping[utterance_levels[last_utterance_index]]

    return soft_prompt, label

_, stoi = load_vocab_mappings()
get_prompt_and_label('batch_0', 'train', stoi)

# -----------------------------------------------------------------------------------------
# Display utils
# -----------------------------------------------------------------------------------------

def plot_loss(target, args, loss_list, display=True):
    '''
        Plots a simple curve showing the different values of the validation loss for each epoch.

        @param target (str):     indicates if the evaluation is on validation or on training
        @param args (dict):      a dictionnary that contains the model parameters
        @param loss_list (list): a list of losses which length corresponds to the number of epochs
        @param display (bool):   a parameter to indicate if the graph should be automatially shown to the user when the function is called (default = True)
    '''
    fig = plt.figure()
    plt.plot(range(len(loss_list)), loss_list)
    plt.xlabel('Epochs')
    plt.ylabel(f'Triplet loss on {target} set')
    plt.title('epochs: {}, batch size: {}, lr: {}, optimizer:{}'.format(args['max_eps'], args['train_bsize'], args['lr'], 'Adam'))

    if display:
        plt.show()

    plt.savefig(f"./{target}_losses.png")


def plot_accuracies(target, args, acc_list, display=True):
    '''
        Plots a simple curve showing the different values of the accuracy for each epoch.

        @param acc_list (list):  a list of accuracies which length corresponds to the number of epochs
        @param args (dict):      a dictionnary that contains the model parameters
        @param display (bool):   a parameter to indicate if the graph should be automatially shown to the user when the function is called (default = True)
    '''
    fig = plt.figure()   
    plt.plot(range(len(acc_list)), acc_list)
    plt.xlabel('Epochs')
    plt.ylabel(f'Accuracy on {target} set (percentage)')
    plt.title('epochs: {}, batch size: {}, lr: {}, optimizer:{}'.format(args['max_eps'], args['train_bsize'], args['lr'], 'Adam'))
    if display:
        plt.show()

    plt.savefig(f"./{target}_accuracies.png")


def display_evaluation_report(task_name, trues, preds):
    '''
        This function provides a display with many classification metrics given the ground truth labels and the predicted labels for a certain task.

        @param task_name (str): the name of the task to display along with the metrics
        @param trues (list): ground truth labels
        @param preds (list): the associated predicted labels
    '''
    print(colored(40*"-", 'blue'))
    print(colored(f"Evaluation report for {task_name} task", 'blue'))
    print(40*"-")
    print("")
    print(colored("Classification report", 'cyan'))
    print(classification_report(trues, preds))
    print("")
    print(colored("F1 score: ", 'cyan'), f1_score(trues, preds))
    print(colored(roc_auc_score(trues, preds), 'cyan'))