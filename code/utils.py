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
import multiprocess as mp
import threading
import json
import re
import csv
from numpy import random as rd
import random
from math import sqrt
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_auc_score, f1_score
from nltk.tokenize import TweetTokenizer
tokenizer = TweetTokenizer()

import warnings
warnings.filterwarnings('ignore')

# set default tensor type
torch.set_default_dtype(torch.float16)

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


def save_batch_generations(batch_generations, batch_index):
    file_path = f'../objects/batch_generations_{batch_index}.tsv'
    with open(file_path, 'w', newline='') as f:
        tsv_writer = csv.writer(f, delimiter='\t')

        for row in batch_generations:
            row = row.replace('\n', '\\n')
            tsv_writer.writerow([row])

    return file_path[3:] # remove ../ to make it relative

def store_split_generations(split, file_paths, trues, experiment):
    '''
        This function concatenate generated sentences from all batches of the split 'split' into one tsv file. 

        @param split (str):             the name of the split, either 'train', 'validation' or 'test'. Note that these are not proper splits as there is no dataset in this fine-tuning procedure
        @param file_paths (list):       the list of paths to the generated sentences in each batch
        @param trues (list):            all the gold readability level labels
        @param experiment (str):        the name of the experiment (=folder where all logs are saved)
    '''
    readability_levels_mapping = {0:'EasilyReadableText', 1:'StandardReadableText', 2:'HardlyReadableText'}
    with open(f'../results/{experiment}/generations_{split}.tsv', 'a') as all_gens:
        tsv_writer = csv.writer(all_gens, delimiter='\t')
        # iterate through batch files
        for i, path in enumerate(file_paths):
            # store gold labels and generations for this batch
            abs_path = f'../{path}'
            with open(abs_path, 'r') as batch_gens:
                for j, row in enumerate(batch_gens):
                    tsv_writer.writerow([readability_levels_mapping[trues[i+j]], row])

            # remove batch-wise generations file path
            os.remove(abs_path)

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
        itos = json.load(f)

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
            index = splitted_v[3] # the 3rd element when individual_592.592_598_utt_21271711 is splitted by _ is 598, which is the utterance index
            # store the readability class for the current index
            levels_indexes_dict[int(index)] = k

    result_dict = dict(sorted(levels_indexes_dict.items()))

    # DEPRECATED CAUSE NO LONGER WORKING WITH DIALOGUES HERE remove the keys after the 10 first utterances to be consistent with padding strategy
    # if len(result_dict) > 10:
    #     result_dict = {k:result_dict[k] for k in list(result_dict.keys())[:10]}

    return result_dict


def get_readability_levels(indiv_path):
    '''
        TODO update doc
        A généraliser ! Pareil pour la fonction suivante

        @param indiv_path (str):    the path to the batch ontology individual.     
    '''
    # get labels mapping for different readability level
    readability_levels_mapping = {'EasilyReadableText':0, 'StandardReadableText':1, 'HardlyReadableText':2}
    # retrieve the ontology individual
    individual = get_ontology(indiv_path).load()
    # crop first element as it simply corresponds to the class occurence
    utterance_levels = {
        'EasilyReadableText':       [str(i) for i in individual.search(is_a=individual.EasilyReadableText)[1:]],
        'StandardReadableText':     [str(i) for i in individual.search(is_a=individual.StandardReadableText)[1:]],
        'HardlyReadableText':       [str(i) for i in individual.search(is_a=individual.HardlyReadableText)[1:]]
    }
    utterance_levels = parse_indexes(utterance_levels)
    labels = [readability_levels_mapping[v] for v in utterance_levels.values()]

    return labels

def random_prompt(concept, classes, hf=False):
    '''
        This auxiliary function allows to get a prompt that ask for a sentence belonging to a certain class among given classes. It is for now used for readability levels but is meant for a more general purpose.

        @param concept (str):               the name of the ontology concept we want to learn, that is divided into the following classes
        @param classes (list):              a list or strings giving the names of all the classes 

        @return prompt (str):               the randomly selected prompt to use for generation
        @return class_index (int):          the index of the corresponding class that is asked in the prompt 
        @param hf:                          False in case of local model, a huggingface model alias otherwise. Currently only 'llama' is supported
    '''
    start_of_sentence = ['A', 'The', 'Yesterday', 'Hello', 'Here', 'In', 'For', 'Yes', 'Tomorrow', 'Today']
    p = rd.uniform()
    n = len(classes)
    for k in range(1, n+1):
        if (k-1)/n < p < k/n:
            if hf == 'llama' or hf == 'adapters':
                prompt = f"Generate a sentence for which {concept} is {classes[k-1]}: {rd.choice(np.array(start_of_sentence))}"
                return prompt, k-1
            else: 
                prompt = f"A sentence whose {concept} is {classes[k-1]}: The"
                return prompt, k-1


def generate_from_random_prompts(args, model, stoi, itos, hf=False):
    concept = 'ReadabilityLevel'
    classes = ['EasyReadableText', 'StandardReadableText', 'HardlyReadableText']
    # try with random IDs instead of concept names
    concept = 'C6468168'
    classes = [f'{concept}{i}' for i in range(3)]
    batch_labels, batch_generations, batch_ids = [], [], []

    if hf == 'llama' or hf == 'adapters':
        # pipe = args['pipe'] # retrieve pipe (if necessary)
        tokenizer = args['tokenizer'] # retrieve tokenizer (if necessary)
        for i in range(args['batch_size']):
            # get a randomly selected prompt (uniform law)
            prompt, label = random_prompt(concept, classes, hf=hf)
            batch_labels.append(label)
            # perform generation
            # result = pipe(prompt, repetition_penalty=1.5, do_sample=False, temperature=0.1)
            # gen = result[0]['generated_text']
            # generation = gen[len(prompt[prompt.find(':')])+1:gen.find('\n')+1]

            # perform generation (gemma)
            prompt = tokenizer(prompt, return_tensors="pt").to(args['device'])
            output = model.generate(**prompt, max_new_tokens=20, repetition_penalty=1.5)[0] # this contains the prompt and the generated part
            output_ids = get_and_pad_ids(prompt, output, padding_length=20)
            result = tokenizer.decode(output)
            batch_ids.append(output_ids)
            generation = result[result.find(':')+1:result.find('\n')]
            print(f'Sample {i}: \t {generation}')
            # print(output_ids)
            # store result
            batch_generations.append(generation)

    else:
        for _ in range(args['batch_size']):
            # get a randomly selected prompt (uniform law)
            prompt, label = random_prompt(concept, classes, hf=hf)
            batch_labels.append(label)
            # encode the prompt
            prompt = encode(stoi, tokenizer.tokenize(prompt))
            prompt = torch.tensor(prompt, dtype=torch.float16).unsqueeze(-1).to(args['device'])
            # perform generation
            generation = model.generate(prompt, max_new_tokens=20, block_size=args['block_size'])[0] # increase max_new_tokens to generate HardlyReadableText
            generation = decode(generation.tolist(), itos)
            # store result
            batch_generations.append(generation)

    return batch_labels, batch_generations, batch_ids

def get_and_pad_ids(prompt, output, padding_length=20):
    '''
        To be documented.
    '''
    output = output[len(prompt[0]):]
    pad_length = max(padding_length - len(output), 0)
    padded_output = torch.nn.functional.pad(output, (0, pad_length), value=0)
    
    return padded_output

def parse_output_and_deduce_class(output, itos):
    '''
        This function takes as argument the output of a BabyLanguageModel model and parses it to retrieve the predicted value for ReadabilityLevel.

        @param output (tensor): a list of indexes corresponding to the generated tokens
        @param itos (list): the mapping from token indexes to their corresponding strings
    '''
    decoded_output = decode(output, itos)
    predicted_class = 3 # let's say we add a class "unable to classify"

    if 'EasilyReadableText' in decoded_output:
        predicted_class = 0
    elif 'StandardReadableText' in decoded_output:
        predicted_class = 1
    elif 'HardlyReadableText' in decoded_output:
        predicted_class = 2

    return predicted_class

def smooth_input(x_input, max_prop=0.6):
    '''
        This function is used in TrainableHeadAdapters model forward to smooth the one-hot encoding of the input to a 60/20/20 like vector (in its default config). It is meant to be used with n_rl classes and a max_prop which allows an equal repartition of other proportion over other labels.
 
        @param x_input (torch.Tensor):
        @param max_prop (float):
    '''
    n_rl = x_input.size()[1]
    # check if it is possible to use this max-prop and compute the other proportions out of it
    assert int((1-max_prop)*100 % (n_rl-1)) == 0 
    # compute the minimal proportion since it can be computed
    min_prop = (1-max_prop)/(n_rl-1) 
    new_classes_vectors = []
    for classes_one_hot in x_input:
        i = torch.argmax(classes_one_hot)
        x = torch.full((n_rl,), min_prop)
        x[i] = max_prop
        new_classes_vectors.append(x)

    return torch.stack(new_classes_vectors)

# -----------------------------------------------------------------------------------------
# Ontology concept learning utils
# -----------------------------------------------------------------------------------------

def extend_vocab_with_readability_levels(itos, stoi):
    '''
        This function adds the useful ontology concepts in the vocabulary used for generation. 
        /!\ This function overwrites the json files contraining itos and stoi mappings 

        @param itos
        @param stoi

        @return itos
        @return stoi
    '''
    readability_levels = ['EasilyReadableText', 'StandardReadableText', 'HardlyReadableText']

    # update itos
    itos.extend(readability_levels)

    # update stoi
    l, rl = len(itos), len(readability_levels)
    for r in range(rl):
        stoi[readability_levels[r]] = l - rl + r + 1
    
    # dump new stoi dict
    with open("../objects/vocab_stoi.json", "w") as f:
        json.dump(stoi, f)

    # dump new itos dict
    with open("../objects/vocab_itos.json", "w") as f:
        json.dump(itos, f)

    return itos, stoi


def create_batch_individual(batch_index, file_path):
    command = ["../call_ontology.sh", str(batch_index), file_path]

    # Run the command using subprocess
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")

# -----------------------------------------------------------------------------------------
# Fine-tuning utils
# -----------------------------------------------------------------------------------------

def transfer_weights(source, target):
    target.self_attn.q_proj.weight = source.self_attn.q_proj.weight
    target.self_attn.k_proj.weight = source.self_attn.k_proj.weight
    target.self_attn.v_proj.weight = source.self_attn.v_proj.weight
    target.self_attn.o_proj.weight = source.self_attn.o_proj.weight
    target.mlp.gate_proj.weight = source.mlp.gate_proj.weight
    target.mlp.up_proj.weight = source.mlp.up_proj.weight
    target.mlp.down_proj.weight = source.mlp.down_proj.weight
    target.input_layernorm.weight = source.input_layernorm.weight
    target.post_attention_layernorm.weight = source.post_attention_layernorm.weight 

# -----------------------------------------------------------------------------------------
# Results logging utils
# -----------------------------------------------------------------------------------------

def save_epoch_data(target, trues, preds, epoch_or_iter, experiment):
    '''
        This function saves the trues and preds at each epoch for train and validation and for each run on test set (several runs are performed to ensure stability)

        @target (str):              either 'train', 'validation' or 'test'
        @trues (tensor):            the gold labels for sentence readability levels
        @preds (tensor):            the readability levels of the generated sentences
        @epoch_or_iter (int):       the # of the epoch (train and val sets) or the iter (test set)
    '''
    with open(f'../results/{experiment}/predictions_{target}.csv', 'a+', newline='') as f:
        write = csv.writer(f)

        # Write data in columns
        for i in range(len(trues)):
            write.writerow([ trues[i], preds[i], epoch_or_iter ])


def log_all_metrics(experiment):
    '''
        This function browses the files containing trues and preds for each set (train, val, test) and computes standard classification metrics + MCC on it

        @param experiment (str):        name of the experiment

        @return metrics (dict)
    '''
    # train set
    for target in ['train', 'validation', 'test']:

        with open(f'../results/{experiment}/predictions_{target}.csv', 'rb', newline='') as f:
            
            pass
    # val set

    # test set

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