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
from numpy import random as rd
from datasets import load_dataset, load_from_disk
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import argparse
import pandas as pd
from termcolor import colored
from collections import Counter
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import logging
from torch.utils.tensorboard import SummaryWriter

# from other scripts
from utils import *
from models import BabyLanguageModel, TrainableHead


def train(args, model, finetuning_model, stoi, itos, epoch):
    '''
        Perfom one epoch of model training in the case of the isolated utterance model trained directly on the triplet loss.

        @param args (str):                 the hyperparameters for the training
        @param model:                      the model to train
        @param optimizer:                  the optimizer to use for training
        @param stoi (dict):                the string-to-index dict from the pretraining vocab
        @param itos (list):                the index-to-string list from the pretraining vocab
        @param epoch (int):                the index of the current epoch

        @return loss_it_avg (list):        the list of all the losses on each batch for the epoch
    '''
    finetuning_model.train()
    optimizer = torch.optim.Adam(finetuning_model.parameters(), lr=args['lr'])

    writer = args['writer']
    loss_it = []
    ce_loss = nn.CrossEntropyLoss()
    trues, preds = [], []

    for batch_index in tqdm(range(args['train_iters']), desc="Epoch %s: " % (epoch+1), total=args['train_iters']):
        batch_labels, batch_generations = generate_from_random_prompts(args, model, stoi, itos) 
        # save the generated sentences to further look at it
        file_path = save_batch_generations(batch_generations, batch_index)

        # what we call 'trues' here refers to the RL that the generated sentence SHOULD have
        trues.extend(batch_labels)

        create_batch_individual(batch_index, file_path)
        generations_rl = get_readability_levels(f'../rdf/individual_batch_{batch_index}.rdf')
        preds.extend(generations_rl)

        # deduce predictions probabilities from predictions
        generations_probas = [[int(j == i) for j in range(3)] for i in generations_rl]
        # pass the "probas" through the finetuning model to compute loss and update main model head
        generations_probas = torch.tensor(generations_probas, dtype=torch.float32, requires_grad=True)
        output_probas = finetuning_model(generations_probas)
        loss = ce_loss(output_probas, torch.tensor(batch_labels))

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss_it.append(loss.item())

    loss_it_avg = sum(loss_it)/len(loss_it)


    # print useful information about the training progress and scores on this training set's full pass
    print("Epoch %s/%s - %s : (%s %s)" % (colored(str(epoch+1), 'blue'),args['max_eps'] , colored('Training', 'blue'), colored('Average loss: ', 'cyan'), loss_it_avg))

	# 🛑 add some metrics to keep with a label and the epoch index
    writer.add_scalar("Loss/train", loss_it_avg, epoch)

    return loss_it_avg, trues, preds


def test(args, model, finetuning_model, stoi, itos, target):
    '''
        Perfom one epoch of model evaluation, either as validation or test.

        @param args (str):                 the hyperparameters for the training
        @param model:                      the model to train
        @param stoi (dict):                the string-to-index dict from the pretraining vocab
        @param itos (list):                the index-to-string list from the pretraining vocab
        @param epoch (int):                the index of the current epoch
        @param target (string):            either 'validation' or 'test', for a better display

        @return loss_it_avg (list):        the list of all the losses on each batch for the epoch
    '''
    finetuning_model.eval()
    writer = args['writer']
    loss_it = []
    ce_loss = nn.CrossEntropyLoss()
    trues, preds = [], []

    for batch_index in tqdm(range(args['eval_iters']), total=args['eval_iters']):

        with torch.no_grad():
            batch_labels, batch_generations = generate_from_random_prompts(args, model, stoi, itos) 
            # save the generated sentences to further look at it
            file_path = save_batch_generations(batch_generations, batch_index)

            # what we call 'trues' here refers to the RL that the generated sentence SHOULD have
            trues.extend(batch_labels)

            create_batch_individual(batch_index, file_path)
            generations_rl = get_readability_levels(f'../rdf/individual_batch_{batch_index}.rdf')
            preds.extend(generations_rl)

            # deduce predictions probabilities from predictions
            generations_probas = [[int(j == i) for j in range(3)] for i in generations_rl]
            # pass the "probas" through the finetuning model to compute loss and update main model head
            generations_probas = torch.tensor(generations_probas, dtype=torch.float32, requires_grad=True)
            output_probas = finetuning_model(generations_probas)
            loss = ce_loss(output_probas, torch.tensor(batch_labels))

            loss_it.append(loss.item())

    loss_it_avg = sum(loss_it)/len(loss_it)

    accuracy = accuracy_score(trues, preds)
    precision = precision_score(trues, preds, average='weighted', zero_division=0.0)
    recall = recall_score(trues, preds, average='weighted')
    f1 = f1_score(trues, preds, average='weighted')

    # print useful information about the training progress and scores on this training set's full pass
    print("%s : (%s %s) (%s %s) (%s %s) (%s %s) (%s %s)" % (colored(f'{target}', 'blue'), colored('Average loss: ', 'cyan'), loss_it_avg, colored('Accuracy: ', 'cyan'), accuracy, colored('Precision: ', 'cyan'), precision, colored('Recall: ', 'cyan'), recall, colored('F1 score: ', 'cyan'), f1))

	# 🛑 add some metrics to keep with a label and the epoch index
    writer.add_scalar(f"Loss/{target}", loss_it_avg)

    return loss_it_avg, trues, preds


def run_epochs(args, model, finetuning_model, stoi, itos, experiment):
    val_losses = []

    for ep in range(args['max_eps']):
        # perform training and validation runs
        _, train_trues, train_preds = train(args, model, finetuning_model, stoi, itos, ep)
        val_loss, val_trues, val_preds = test(args, model, finetuning_model, stoi, itos, 'validation')
 
        # save epoch trues and preds for train and validation
        save_epoch_data('train', train_trues, train_preds, ep, experiment)
        save_epoch_data('validation', val_trues, val_preds, ep, experiment)

        # save val loss for this epoch
        val_losses.append(val_loss)

    return val_losses


def run_on_several_test_sets(args, model, finetuning_model, stoi, itos, experiment, episodes=10):
    '''
        This function accounts for model stability by testing the model on different test sets depending on the number of episodes. Predictions are stored for each episode so all the classification metrics can be computed as well as their mean and standard deviation.

        @param args (dict):                 the hyperparameters for the training
        @param model:                       the pretrained model to use for inference
        @param stoi (dict):                 the string-to-index dict from the pretraining vocab
        @param itos (list):                 the index-to-string list from the pretraining vocab
        @param experiment (str):            the name of the experiment
        @param episoddes (int):             the number of test sets to infer on (default=10)

        @return test_losses (list):         the losses on test sets (length = number of episodes)
    '''
    test_losses = []

    for i in range(episodes):
        test_loss, test_trues, test_preds = test(args, model, finetuning_model, stoi, itos, 'test')
        save_epoch_data('test', test_trues, test_preds, i, experiment)

        test_losses.append(test_loss)

    return test_losses


def run_exp(args, model_path, experiment, episodes=10):
    print(colored(f'Start of the experiment {experiment}', 'green'))

    # create results dir if it doesn't exist
    if not os.path.exists(f'../results/{experiment}/'):
        os.makedirs(f'../results/{experiment}/')

    # Getting stoi and itos dicts
    itos, stoi = load_vocab_mappings()

    # Load the pretrained model weights
    model = BabyLanguageModel(args)
    if args['device'] == 'cpu':
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    else:
        model.load_state_dict(torch.load(model_path))
    model.to(args['device'])

    # freeze all layers except model.lm_head
    for p in model.token_embedding_table.parameters(): p.requires_grad = False
    for p in model.position_embedding_table.parameters(): p.requires_grad = False
    for p in model.blocks.parameters(): p.requires_grad = False
    for p in model.ln_f.parameters(): p.requires_grad = False
    for p in model.lm_head.parameters(): p.requires_grad = True

    # Setup finetuning model
    finetuning_model = TrainableHead(args)
    finetuning_model.lm_head.weight = model.lm_head.weight
    for p in finetuning_model.pool.parameters(): p.requires_grad = False
    for p in finetuning_model.anti_pool.parameters(): p.requires_grad = False

    # run training and validation
    val_losses = run_epochs(args, model, finetuning_model, stoi, itos, experiment)

    # run test 
    test_losses = run_on_several_test_sets(args, model, finetuning_model, stoi, itos, experiment, episodes)

    # log all classification metrics from saved trues/preds
    ## TBC

    return val_losses, test_losses


if __name__ == "__main__":

    args = {'vocab_size':239267, # new vocab size corresponding to the new dataset
            'batch_size':8,
            'block_size':64, 
            'train_iters':10,
            'eval_iters':1,
            'lr':1e-1,
            'device':activate_gpu(force_cpu=True),
            'max_eps':10,
            'n_embd':64,
            'n_heads':8,
            'n_layers':24,
            'dropout':0.3,
            'writer':SummaryWriter(f"../logs/{get_datetime()}_{64}")
        }

    model_path = '../models/babyllm-gptlike_64_22012024223644_nq_params.pt'

    run_exp(args, model_path, 'firstTestOnGPU')


    # [x] OPTION 1: check the readability class of the output. To do so, write an auxiliary function that:
        # - generates a sentence with a readability level instruction given in prompt
        # - add this infividual to a temp rdf file for the batch
        # - perform inference on this file (like it is done in create_individuals.py)
        # - uses the mapping class -> class index to finally output the individual class
    
    # [x] OPTION 2: change the prompt to finish on last utterance by (ReadabilityLevel= which encourages the model to learn the concept of readability (in a final test step we can use OPTION 1 to check of the model actually learnt something). We need a function that:
        # - decodes the output
        # - parses it to deduce the predicted readability level
        # - maps it to the class index, and that's all :)
