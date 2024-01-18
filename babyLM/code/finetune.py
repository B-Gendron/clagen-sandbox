# torch utils
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

# general purpose modules
from glob import glob
import os
import subprocess
import numpy as np
from datasets import load_dataset, load_from_disk
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import argparse
import pandas as pd
from termcolor import colored
from collections import Counter
import logging
from torch.utils.tensorboard import SummaryWriter

# from other scripts
from utils import activate_gpu
from models import BabyLanguageModel


# To be used after: check if the individuals (rdf) has been created
    # if os.path.exists("../../../OntoUttPreprocessing/rdf/wikitalk")


class WikiTalkDataset(torch.utils.data.Dataset):
    '''
        Dataset class for wikitalk pages dataset
    '''
    def __init__(self, data, args):
        self._data = data
        self.args = args

    @property
    def data(self):
        return self._data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = {
            'dial_id':      np.array(self._data[idx]['dial_id']),
            'utt_id':       np.array(self._data[idx]['utt_id']), # penser à padder ça éventuellement
            'embedding':    np.array(self._data[idx]['embedding'])
        }
        return item
    

def get_args_and_dataloaders(dataset, dataset_class):
    '''
        Instantiate the training hyperparameters and the dataloaders.

        @param dataset:                     the data to put in the DataLoader
        @param dataset_class (Dataset):     the consistent dataset class from the datasets.py script to processed data

        @return args (dict):                a dictionary that contains the hyperparameters for training
        @return train_loader (dataloader):  the dataloader that contains the training samples
        @return val_loader (dataloader):    the dataloader that contains the validation samples
        @return test_loader (dataloader):   the dataloader that contains the test samples
    '''
    args = {'train_bsize': 32, 'eval_bsize': 1, 'lr': 0.00001, 'spreading':False}
    train_loader = DataLoader(dataset=dataset_class(dataset["train"], args=args), pin_memory=True, batch_size=args['train_bsize'], shuffle=True, drop_last=True)
    val_loader   = DataLoader(dataset=dataset_class(dataset["validation"], args=args), pin_memory=True, batch_size=args['eval_bsize'], shuffle=True, drop_last=True)
    test_loader  = DataLoader(dataset=dataset_class(dataset["test"], args=args), pin_memory=True, batch_size=args['eval_bsize'], shuffle=True, drop_last=True)
    return args, train_loader, val_loader, test_loader


if __name__ == "__main__":

    wikitalk = load_from_disk("../wikitalk")
    args, train_loader, val_loader, test_loader = get_args_and_dataloaders(wikitalk, WikiTalkDataset)
    print(next(iter(train_loader)))