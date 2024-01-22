# torch utils
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

# general purpose modules
from glob import glob
import os
import subprocess
import json
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
from utils import activate_gpu, get_prompt_and_label
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


def train(args, model, train_loader, stoi, optimizer, epoch):
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
    device = 'cpu'
    writer = args['writer']
    loss_it = []
    ce_loss = nn.CrossEntropyLoss()

    for it, batch in tqdm(enumerate(train_loader), desc="Epoch %s: " % (epoch+1), total=train_loader.__len__()):
        batch = {'dial_id': batch['dial_id'].to(device), 'utt_id': batch['utt_id'].to(device), 'embedding' : batch['embedding'].to(device)}
        optimizer.zero_grad()

        # remove padded part
        for idx in range(args['train_bsize']):
            # get the dialog data 
            encoded_dialog = batch['embedding'][idx]
            dialog_without_pad = []
            for utterance in encoded_dialog:
                # select token indexes only
                utt = utterance.tolist()
                utterance_without_pad = utt[:utt.index(-1) if -1 in utt else len(utt)]
                # naturally remove the utterances full of pad
                if utterance_without_pad != []:
                    dialog_without_pad.append(utterance_without_pad)

            # dump dialog encoding without padding in a json file
            with open(f'../objects/batch_{idx}.json', 'w') as f:
                json.dump({'dial_id':batch['dial_id'][idx].item(), 'dial_encoding':dialog_without_pad}, f, indent=2)

        # make a list with all the file names
        files_list = []
        for f in files_list:
            prompt, label = get_prompt_and_label(f, 'train', stoi) # define stoi

    ### NEXT LINES ARE TO BE ADAPTED
    # create a new model class that encapsulates the GPT-like model and adds some extra layers for classification. Copy and paste class and 

    # perform training
    classes_probas = model(batch['embedding'])
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

    print("Getting stoi and itos dicts...")
    with open("../objects/vocab_stoi.json", "r") as f:
        stoi = json.load(f)
    with open("../objects/vocab_itos.json", "r") as f:
        itos = json.load(f)

    model = BabyLanguageModel(args)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args['lr'], foreach=False)
    train(args, model, train_loader, stoi, optimizer, 0)
