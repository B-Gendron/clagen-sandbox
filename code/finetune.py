# torch utils
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from transformers import ( AutoModelForCausalLM, AutoTokenizer,
pipeline,
)

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


def train(args, model, finetuning_model, stoi, itos, epoch, experiment, hf=False):
    '''
        Perfom one epoch of model training in the case of the isolated utterance model trained directly on the triplet loss.

        @param args (str):                 the hyperparameters for the training
        @param model:                      the pretrained model to use for inference
        @param finetuning_model:           the model used for weight updates in fine-tuning
        @param stoi (dict):                the string-to-index dict from the pretraining vocab
        @param itos (list):                the index-to-string list from the pretraining vocab
        @param epoch (int):                the index of the current epoch
        @param experiment (str):           the experiment name
        @param hf:                         False in case of local model, a huggingface model alias otherwise. Currently only 'llama' is supported

        @return loss_it_avg (list):        the list of all the losses on each batch for the epoch
        @return trues (list):              list of gold labels to be stored later
        @return preds (list):              list of the associated predictions to be stored later
    '''
    finetuning_model.train()
    optimizer = torch.optim.Adam(finetuning_model.parameters(), lr=args['lr'])

    writer = args['writer']
    loss_it = []
    ce_loss = nn.CrossEntropyLoss()
    trues, preds = [], []
    file_paths = []

    for batch_index in tqdm(range(args['train_iters']), desc="Epoch %s: " % (epoch+1), total=args['train_iters']):
        batch_labels, batch_generations = generate_from_random_prompts(args, model, stoi, itos, hf=hf) 
        # save the generated sentences to further look at it
        file_path = save_batch_generations(batch_generations, batch_index)
        file_paths.append(file_path)

        # what we call 'trues' here refers to the RL that the generated sentence SHOULD have
        trues.extend(batch_labels)
        create_batch_individual(batch_index, file_path)
        generations_rl = get_readability_levels(f'../rdf/individual_batch_{batch_index}.rdf')
        preds.extend(generations_rl)

        # deduce predictions "probabilities" from predictions
        generations_probas = [[int(j == i) for j in range(3)] for i in generations_rl]
        # pass the "probas" through the finetuning model to compute loss and update main model head
        generations_probas = torch.tensor(generations_probas, dtype=torch.float32, requires_grad=True).to(args['device'])
        output_probas = finetuning_model(generations_probas)
        loss = ce_loss(output_probas, torch.tensor(batch_labels).to(args['device']))

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss_it.append(loss.item())

        # update the weights of the main model
        model.lm_head.weight = finetuning_model.lm_head.weight 

    # append batch generations to split generations
    store_split_generations('train', file_paths, trues, experiment)

    loss_it_avg = sum(loss_it)/len(loss_it)

    # print useful information about the training progress and scores on this training set's full pass
    print("Epoch %s/%s - %s : (%s %s)" % (colored(str(epoch+1), 'blue'),args['max_eps'] , colored('Training', 'blue'), colored('Average loss: ', 'cyan'), loss_it_avg))

	# 🛑 add some metrics to keep with a label and the epoch index
    writer.add_scalar("Loss/train", loss_it_avg, epoch)

    return loss_it_avg, trues, preds


def test(args, model, finetuning_model, stoi, itos, target, experiment, hf=False):
    '''
        Perfom one epoch of model evaluation, either as validation or test.

        @param args (str):                 the hyperparameters for the training
        @param model:                      the pretrained model to use for inference
        @param finetuning_model:           the model used for weight updates in fine-tuning
        @param stoi (dict):                the string-to-index dict from the pretraining vocab
        @param itos (list):                the index-to-string list from the pretraining vocab
        @param target (str):               either 'validation' or 'test', for a better display
        @param experiment (str):           the experiment name
        @param hf:                          False in case of local model, a huggingface model alias otherwise. Currently only 'llama' is supported

        @return loss_it_avg (list):        the list of all the losses on each batch for the epoch
        @return trues (list):              list of gold labels to be stored later
        @return preds (list):              list of the associated predictions to be stored later
    '''
    finetuning_model.eval()
    writer = args['writer']
    loss_it = []
    ce_loss = nn.CrossEntropyLoss()
    trues, preds = [], []
    file_paths = []

    for batch_index in tqdm(range(args['eval_iters']), total=args['eval_iters']):

        batch_labels, batch_generations = generate_from_random_prompts(args, model, stoi, itos, hf=hf) 
        # save the generated sentences to further look at it
        file_path = save_batch_generations(batch_generations, batch_index)
        file_paths.append(file_path)

        # what we call 'trues' here refers to the RL that the generated sentence SHOULD have
        trues.extend(batch_labels)

        create_batch_individual(batch_index, file_path)
        generations_rl = get_readability_levels(f'../rdf/individual_batch_{batch_index}.rdf')
        preds.extend(generations_rl)

        # deduce predictions probabilities from predictions
        generations_probas = [[int(j == i) for j in range(3)] for i in generations_rl]
        # pass the "probas" through the finetuning model to compute loss and update main model head
        generations_probas = torch.tensor(generations_probas, dtype=torch.float32, requires_grad=True).to(args['device'])
        output_probas = finetuning_model(generations_probas)
        loss = ce_loss(output_probas, torch.tensor(batch_labels).to(args['device']))

        loss_it.append(loss.item())

    loss_it_avg = sum(loss_it)/len(loss_it)

    # append batch generations to split generations
    store_split_generations(target, file_paths, trues, experiment)

    accuracy = accuracy_score(trues, preds)
    precision = precision_score(trues, preds, average='weighted', zero_division=0.0)
    recall = recall_score(trues, preds, average='weighted')
    f1 = f1_score(trues, preds, average='weighted')

    # print useful information about the training progress and scores on this training set's full pass
    print("%s : (%s %s) (%s %s) (%s %s) (%s %s) (%s %s)" % (colored(f'{target}', 'blue'), colored('Average loss: ', 'cyan'), loss_it_avg, colored('Accuracy: ', 'cyan'), accuracy, colored('Precision: ', 'cyan'), precision, colored('Recall: ', 'cyan'), recall, colored('F1 score: ', 'cyan'), f1))

	# 🛑 add some metrics to keep with a label and the epoch index
    writer.add_scalar(f"Loss/{target}", loss_it_avg)

    return loss_it_avg, trues, preds


def run_episodes(args, model, finetuning_model, stoi, itos, experiment, hf=False):
    '''
        Run all episodes of the fine-tuning (train + validation).

        @param args (dict):                 the hyperparameters for the training
        @param model:                       the pretrained model to use for inference
        @param finetuning_model:            the model used for weight updates in fine-tuning
        @param stoi (dict):                 the string-to-index dict from the pretraining vocab
        @param itos (list):                 the index-to-string list from the pretraining vocab
        @param experiment (str):            the name of the experiment
        @param hf:                          False in case of local model, a huggingface model alias otherwise. Currently only 'llama' is supported

        @return val_losses (list):          the losses on validation sets (length = number of val_iters)
    '''
    val_losses = []

    for ep in range(args['max_eps']):
        # perform training and validation runs
        _, train_trues, train_preds = train(args, model, finetuning_model, stoi, itos, ep, experiment, hf=hf)
        val_loss, val_trues, val_preds = test(args, model, finetuning_model, stoi, itos, 'validation', experiment, hf=hf)
 
        # save epoch trues and preds for train and validation
        save_epoch_data('train', train_trues, train_preds, ep, experiment)
        save_epoch_data('validation', val_trues, val_preds, ep, experiment)

        # save val loss for this epoch
        val_losses.append(val_loss)

    return val_losses


def run_on_several_test_sets(args, model, finetuning_model, stoi, itos, experiment, episodes=10, hf=False):
    '''
        This function accounts for model stability by testing the model on different test sets depending on the number of episodes. Predictions are stored for each episode so all the classification metrics can be computed as well as their mean and standard deviation.

        @param args (dict):                 the hyperparameters for the training
        @param model:                       the pretrained model to use for inference
        @param finetuning_model:            the model used for weight updates in fine-tuning
        @param stoi (dict):                 the string-to-index dict from the pretraining vocab
        @param itos (list):                 the index-to-string list from the pretraining vocab
        @param experiment (str):            the name of the experiment
        @param episodes (int):              the number of test sets to infer on (default=10)
        @param hf:                          False in case of local model, a huggingface model alias otherwise. Currently only 'llama' is supported

        @return test_losses (list):         the losses on test sets (length = number of episodes)
    '''
    test_losses = []

    for i in range(episodes):
        test_loss, test_trues, test_preds = test(args, model, finetuning_model, stoi, itos, 'test', hf=hf)
        save_epoch_data('test', test_trues, test_preds, i, experiment)

        test_losses.append(test_loss)

    return test_losses


def run_exp(args, model_name, experiment, episodes=10, hf=False):
    '''
        Run an end-to-end finetuning.

        @param args (dict):           the dict containing all the hyperparameters
        @param model_name (str):      either from a local storage (hf=False), or from huggingface hub (hf=True)
        @param experiment (str):      name of the experiment 
        @param episodes (int):        number of times the test step should be performed (to compute descriptive stats on metrics)
        @param hf:                    False in case of local model, a huggingface model alias otherwise. Currently only 'llama' is supported

        @return val_losses (list):    all val losses for all the 'epochs'
        @return test_losses (list):   all test losses from all the episodes
    '''
    print(colored(f'Start of the experiment {experiment}', 'green'))

    # create results dir if it doesn't exist
    if not os.path.exists(f'../results/{experiment}/'):
        os.makedirs(f'../results/{experiment}/')

    # Getting stoi and itos dicts
    itos, stoi = load_vocab_mappings()

    if hf == 'llama':
        # setup model
        model = AutoModelForCausalLM.from_pretrained(  
            model_name,
            low_cpu_mem_usage=True,
            return_dict=True,       # this returns a load_state_dict compatible object ?
            torch_dtype=torch.float16,
            device_map=args['device'],
        )
        # setup tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        # store the pipe to use it in generation
        pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=10)
        args.update({'pipe':pipe})
        # freeze all layers except lm_head (not the better option but just to test)
        for p in model.parameters(): p.requires_grad = False
        for p in model.lm_head.parameters(): p.requires_grad = True

    else:
        # Load the pretrained model weights
        model = BabyLanguageModel(args)
        if args['device'] == 'cpu':
            model.load_state_dict(torch.load(model_name, map_location=torch.device('cpu')))
        else:
            model.load_state_dict(torch.load(model_name))
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
    finetuning_model.to(args['device'])

    # run training and validation
    val_losses = run_episodes(args, model, finetuning_model, stoi, itos, experiment, hf=hf)

    # run test 
    test_losses = run_on_several_test_sets(args, model, finetuning_model, stoi, itos, experiment, episodes, hf=hf)

    # log all classification metrics from saved trues/preds
    ## TBC

    return val_losses, test_losses


if __name__ == "__main__":

    args = {'vocab_size':239267,                    # new vocab size corresponding to the new dataset
            'batch_size':32,                        # size of the batch, the greater bsize the greater number of data samples
            'block_size':64,            # Transformer block size in the language model
            'train_iters':100,                      # number of train batches to consider in one episode
            'eval_iters':10,                        # number of validation/test batches to consider in one episode
            'lr':1e-3,                              # learning rate
            'device':activate_gpu(force_cpu=True),  # set device for training. Desable force_cpu to run on gpu if available
            'max_eps':10,                           # number of episodes (max of episodes in case of early stopping)
            'n_embd':64,                # embedding size
            'n_heads':8,                # number of attention heads for one transformer block
            'n_layers':24,              # number of Transformer layers in the language model
            'dropout':0.3,              # dropout rate
            'writer':SummaryWriter(f"../logs/{get_datetime()}"), # Tensorboard util
            'hf':False,                             # False if BabyLM, otherwise llama, falcon, mistral,... 
        }

    # model_path = '../models/babyllm-gptlike_64_22012024223644_nq_params.pt'
    model_name = "meta-llama/Llama-2-7b-chat-hf"
    args.update({'hf':'llama'})

    run_exp(args, model_name, 'firstTestOnGPU', hf='llama')


    # [x] OPTION 1: check the readability class of the output. To do so, write an auxiliary function that:
        # - generates a sentence with a readability level instruction given in prompt
        # - add this infividual to a temp rdf file for the batch
        # - perform inference on this file (like it is done in create_individuals.py)
        # - uses the mapping class -> class index to finally output the individual class
    
    # [x] OPTION 2: change the prompt to finish on last utterance by (ReadabilityLevel= which encourages the model to learn the concept of readability (in a final test step we can use OPTION 1 to check of the model actually learnt something). We need a function that:
        # - decodes the output
        # - parses it to deduce the predicted readability level
        # - maps it to the class index, and that's all :)
