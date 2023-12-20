import os
from owlready2 import *
from termcolor import colored
from pprint import pprint
from datetime import datetime
import torch
import logging
from datasets import Dataset, DatasetDict
from collections import Counter
import json
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_auc_score, f1_score


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
    """
        return the right device depending on GPU availability
    """
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


def set_logger(experiment):
    '''
        -------------------------------- /!\ WARNING /!\ --------------------------------
        Please note that the logger is not yet completetly handled in the main file, therefore you should consider using this function only if you properly define the logger usage in the desired functions.
        ------------------------------------------------------------------------------------

        Set a logger to be used when running experiments. 
    '''
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
        dataset = DatasetDict.load_from_disk(f"./data/{dataset_name}")
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
        dataset.save_to_disk(f'./data/{dataset_name}')
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