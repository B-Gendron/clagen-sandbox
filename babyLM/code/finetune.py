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


def train_isolated(args, model, train_loader, optimizer, epoch):
    '''
        Perfom one epoch of model training in the case of the isolated utterance model trained directly on the triplet loss.

        @param args (str):                 the hyperparameters for the training
        @param model:                      the model to train
        @param train_loader (DataLoader):  the dataloader that contains the training samples
        @param optimizer:                  the optimizer to use for training
        @param epoch (int):                the index of the current epoch

        @return loss_it_avg (list):        the list of all the losses on each batch for the epoch
    '''
    model.train()
    device = args['device']
    writer = args['writer']
    loss_it = []
    ce_loss = nn.CrossEntropyLoss()

    for it, batch in tqdm(enumerate(train_loader), desc="Epoch %s: " % (epoch+1), total=train_loader.__len__()):
        batch = {'dial_id': batch['dial_id'].to(device), 'utt_id': batch['utt_id'].to(device), 'embedding' : batch['embedding'].to(device)}
        optimizer.zero_grad()

        # perform training
        classes_probas = model(batch['embedding'])
        loss = ce_loss(A, P, N)
        loss.backward()
        optimizer.step()

        # store loss history
        loss_it.append(loss.item())

    loss_it_avg = sum(loss_it)/len(loss_it)

    # print useful information about the training progress and scores on this training set's full pass
    print("Epoch %s/%s - %s : (%s %s)" % (colored(str(epoch+1), 'blue'),args['max_eps'] , colored('Training', 'blue'), colored('Average loss: ', 'cyan'), sum(loss_it)/len(loss_it)))

	# 🛑 add some metrics to keep with a label and the epoch index
    writer.add_scalar("Loss/train", loss_it_avg, epoch)

    return loss_it_avg

if __name__ == "__main__":

    wikitalk = load_from_disk("../wikitalk")
    args, train_loader, val_loader, test_loader = get_args_and_dataloaders(wikitalk, WikiTalkDataset)
    print(next(iter(train_loader)))