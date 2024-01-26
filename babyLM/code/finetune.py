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
from utils import *
from models import BabyLanguageModel


def train(args, model, train_loader, stoi, itos, optimizer, epoch):
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

    for it, batch in tqdm(enumerate(train_loader), desc="Epoch %s: " % (epoch+1), total=train_loader.__len__()):
        batch = {'dial_id': batch['dial_id'].to(device), 'utt_id': batch['utt_id'].to(device), 'embedding' : batch['embedding'].to(device)}
        optimizer.zero_grad()

        batch_trues, batch_preds = [], [] 
        batch_ids = []

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

            if dialog_without_pad != []:
                dial_id = batch['dial_id'][idx].item()
                batch_ids.append(dial_id)
                # dump dialog encoding without padding in a json file
                with open(f'../objects/batch_{idx}.json', 'w') as f:
                    json.dump({'dial_id':dial_id, 'dial_encoding':dialog_without_pad}, f, indent=2)

        # make a list with all the file names
        file_list = [os.path.join("../objects/", filename) for filename in [f"batch_{i}.json" for i in range(args['train_bsize'])]]
        for f in file_list:
            prompt, label = get_prompt_and_label(f, 'train', stoi, args['device'])
            read_level_probas = model.predict_readability_levels(prompt, block_size=args['block_size'])
            batch_preds.append(read_level_probas)
            # update trues/preds lists
            batch_trues.append(label)
        # convert to tensors
        batch_preds, batch_trues = torch.stack(batch_preds), torch.tensor(batch_trues)

        # these 3 elements should be saved in a file during training
        save_batch_info(batch_ids, batch_trues, torch.argmax(batch_preds, dim=-1), batch_preds, output_file='first_test')
        preds.extend(torch.argmax(batch_preds, dim=-1).tolist()) # this should not work.

        # compute and backpropagate MSE loss on batch predictions
        loss = ce_loss(batch_preds.to(device), batch_trues.to(device))
        loss_it.append(loss.item())
        loss.backward()
        optimizer.step()

        # update general lists
        trues.extend(batch_trues)

    loss_it_avg = sum(loss_it)/len(loss_it)

    # print useful information about the training progress and scores on this training set's full pass
    print("Epoch %s/%s - %s : (%s %s)" % (colored(str(epoch+1), 'blue'),args['max_eps'] , colored('Training', 'blue'), colored('Average loss: ', 'cyan'), loss_it_avg))

	# 🛑 add some metrics to keep with a label and the epoch index
    writer.add_scalar("Loss/train", loss_it_avg, epoch)

    return loss_it_avg, trues, preds



def run_epochs(args, model, train_loader, val_loader, test_loader, optimizer):
    pass

if __name__ == "__main__":

    args = {'vocab_size':239270, # new vocab size corresponding to the new dataset + add 3 onto concepts
            'batch_size':8,
            'block_size':64, 
            'max_iters':5000,
            'eval_interval':100,
            'lr':1e-3,
            'device':activate_gpu(),
            'max_eps':10,
            'eval_iters':1000,
            'n_embd':64,
            'n_heads':8,
            'n_layers':24,
            'dropout':0.3,
            'writer':SummaryWriter(f"../logs/{get_datetime()}_{64}")
        }

    print("Getting stoi and itos dicts...")
    itos, stoi = load_vocab_mappings()
    print(len(itos))

    print("Load the pretrained model weights...")
    model_path = '../models/babyllm-gptlike_64_25012024110257_nq_params.pt'
    pretrained_model = BabyLanguageModel(args)
    pretrained_model.load_state_dict(torch.load(model_path))
    pretrained_model.to(args['device'])

    print("Start fine-tuning on one epoch...")
    optimizer = torch.optim.AdamW(pretrained_model.parameters(), lr=args['lr'], foreach=False)
    train(args, pretrained_model, train_loader, stoi, itos, optimizer, 0)


    # OPTION 1: check the readability class of the output. To do so, write an auxiliary function that:
        # - generates a sentence with a readability level instruction given in prompt
        # - add this infividual to a temp rdf file for the batch
        # - perform inference on this file (like it is done in create_individuals.py)
        # - uses the mapping class -> class index to finally output the individual class
    
    # [x] OPTION 2: change the prompt to finish on last utterance by (ReadabilityLevel= which encourages the model to learn the concept of readability (in a final test step we can use OPTION 1 to check of the model actually learnt something). We need a function that:
        # - decodes the output
        # - parses it to deduce the predicted readability level
        # - maps it to the class index, and that's all :)
