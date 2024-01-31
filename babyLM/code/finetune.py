# torch utils
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

# general purpose modules
from glob import glob
import os
import subprocess
import csv
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
from utils import *
from models import BabyLanguageModel


def train(args, model, optimizer, stoi, itos, epoch):
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
    trues, preds = [], []

    for batch_index in tqdm(range(args['max_iters']), desc="Epoch %s: " % (epoch+1), total=args['max_iters']):
        optimizer.zero_grad()

        batch_labels, batch_generations = generate_from_random_prompts(args, model, stoi, itos)
        # save the generated sentences to further look at it
        file_path = save_batch_generations(batch_generations, batch_index)

        # what we call 'trues' here refers to the RL that the generated sentence SHOULD have
        trues.extend(batch_labels)

        create_batch_individual(batch_index, file_path)
        exit()


        ce_loss.backward()
        optimizer.step()

        # update general lists
        # trues.extend(batch_trues)

    loss_it_avg = sum(loss_it)/len(loss_it)

    # print useful information about the training progress and scores on this training set's full pass
    print("Epoch %s/%s - %s : (%s %s)" % (colored(str(epoch+1), 'blue'),args['max_eps'] , colored('Training', 'blue'), colored('Average loss: ', 'cyan'), loss_it_avg))

	# 🛑 add some metrics to keep with a label and the epoch index
    writer.add_scalar("Loss/train", loss_it_avg, epoch)

    return loss_it_avg, trues, preds

def test(args, model, loader):
    pass


def run_epochs(args, model, optimizer):
    pass

def run_exp(args, model):
    # include the layers to freeze and not to freeze

    pass

if __name__ == "__main__":

    args = {'vocab_size':239267, # new vocab size corresponding to the new dataset + add 3 onto concepts
            'batch_size':8,
            'block_size':64, 
            'max_iters':5000,
            'eval_interval':100,
            'lr':1e-3,
            'device':activate_gpu(force_cpu=True),
            'max_eps':10,
            'eval_iters':1000,
            'n_embd':64,
            'n_heads':8,
            'n_layers':24,
            'dropout':0.3,
            'train_bsize':8,
            'writer':SummaryWriter(f"../logs/{get_datetime()}_{64}")
        }

    print("Getting stoi and itos dicts...")
    itos, stoi = load_vocab_mappings()
    print(len(itos))

    print("Load the pretrained model weights...")
    model_path = '../models/babyllm-gptlike_64_22012024223644_nq_params.pt'
    pretrained_model = BabyLanguageModel(args)
    pretrained_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    pretrained_model.to(args['device'])

    print("Start fine-tuning on one epoch...")
    optimizer = torch.optim.AdamW(pretrained_model.parameters(), lr=args['lr'], foreach=False)
    train(args, pretrained_model, optimizer, stoi, itos, 0)


    # OPTION 1: check the readability class of the output. To do so, write an auxiliary function that:
        # - generates a sentence with a readability level instruction given in prompt
        # - add this infividual to a temp rdf file for the batch
        # - perform inference on this file (like it is done in create_individuals.py)
        # - uses the mapping class -> class index to finally output the individual class
    
    # [x] OPTION 2: change the prompt to finish on last utterance by (ReadabilityLevel= which encourages the model to learn the concept of readability (in a final test step we can use OPTION 1 to check of the model actually learnt something). We need a function that:
        # - decodes the output
        # - parses it to deduce the predicted readability level
        # - maps it to the class index, and that's all :)
