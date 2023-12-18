# Inspired from: https://colab.research.google.com/drive/1JMLa53HDuA-i7ZBmqV7ZnA3c_fvtXnx-?usp=sharing#scrollTo=nql_1ER53oCf 

import torch
import torch.nn as nn
import torch.nn.functional as F

from glob import glob
import os
import subprocess
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
import argparse
import pandas as pd
from termcolor import colored
from collections import Counter
import logging
from torch.utils.tensorboard import SummaryWriter

from torch.utils.data import Dataset
import numpy as np

from utils import activate_gpu
from models import BabyLanguageModel

class WikiTalkDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = {
          ## TODO add appropriate content
        }
        return item
    

# def read_tinyshakepeare():
#     # wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
#     with open('input.txt', 'r', encoding='utf-8') as f:
#         text = f.read()
#     return text

def read_data(folder_name):
    file_list = glob(os.path.join("/data/", folder_name, "*.txt"))
    print(f'Reading {len(file_list)} files...')
    corpus = []
    for file_path in tqdm(file_list):
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
            corpus.append(text)
    print("Finished!")

    # gather all textual data in a big string
    text_data = " ".join(corpus)
    return text_data


def get_vocab_info(data):
    '''
        Create a vocabulary mapping with training data text processed as a bag of words. Default vocab_size is the number of different words in the whole dataset.
        TODO change vocab size computation (use torch.vocab ?)
        TODO add a feature to choose a vocab size (in order to alleviate data processing for bigger datasets). This might be required for open web corpus.
    '''
    # here are all the unique characters that occur in this text
    chars = sorted(list(set(data)))
    # chars = sorted(list(set(data)))
    vocab_size = len(chars)
    # create a mapping from characters to integers
    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(chars) }
    return vocab_size, stoi, itos


encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string


# Train and test splits
def train_val_split(data, train_ratio=0.9):
    '''
        Processes train/val split according to the given train_ratio (default = 90% of the dataset is used for training)
    '''
    tensor_data = torch.tensor(encode(data), dtype=torch.long)
    n = int(train_ratio*len(tensor_data))
    train_data = tensor_data[:n]
    val_data = tensor_data[n:]
    return train_data, val_data


# data loading
def get_batch(split, device):
    '''
        This function is a kind of manual dataloader.
    '''
    block_size = args['block_size']

    # get the desired data split
    data = train_data if split == 'train' else val_data
    # randomly select some indexes
    ix = torch.randint(len(data) - block_size, (args['batch_size'],))
    # deduce corresponding data
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    # send data to device
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss(device):
    '''
        This function is NOT a training loop. It simply estimates the loss value on each batch and outputs the mean of these loss values.
    '''
    out = {}
    model.eval()
    eval_iters = args['eval_iters']
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split, device)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def train_and_infer(model, args, optimizer, device):
    '''
        To be documented.
    '''
    max_iters = args['max_iters']
    for iter in tqdm(range(max_iters)):

        # every once in a while evaluate the loss on train and val sets
        if iter % args['eval_interval'] == 0 or iter == max_iters - 1:
            losses = estimate_loss(device)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        # sample a batch of data
        xb, yb = get_batch('train', device)

        # evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()


if __name__ == "__main__":

    if not os.path.isfile("../data/utterances.jsonl"): 
        subprocess.call(['sh', 'download_openwebtext.sh'])

    # text = read_data('tinyshakespeare')
    text = read_data('openwebtext')

    vocab_size, stoi, itos = get_vocab_info(text)
    print(vocab_size)

    # hyperparameters default config
    args = {
        'vocab_size':get_vocab_info(text)[0], # to be implemented
        'batch_size':16, # how many independent sequences will we process in parallel?
        'block_size':32, # what is the maximum context length for predictions?
        'max_iters':5000,
        'eval_interval':100,
        'lr':1e-3,
        'device':activate_gpu(),
        'eval_iters':200,
        'n_embd':64,
        'n_heads':4,
        'n_layers':4,
        'dropout':0.0
    }

    torch.manual_seed(42)

    # instantiate parser and retrieve model hyperparameters
    # args dict contains (vocab_size, n_embd, block_size, n_head, n_layer, dropout, device) that have default values but are retrived from argparse
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-v", "--vocab_size", help="The size of the vocabulary. Default is the number of unique characters in the training corpus.", type=int, default=vocab_size)
    parser.add_argument("-e", "--embedding_size", help=f"The embedding size. Default is {args['n_embd']}.", type=int, default=args['n_embd'])
    parser.add_argument("-b", "--block_size", help=f"The size of the Transformer decoder block, i.e. the maximum context length for predictions. Default is {args['block_size']}.", type=int, default=args['block_size'])
    parser.add_argument("-h", "--heads", help=f"Number of attention heads. Default is {args['n_heads']}.", type=int, default=args['n_heads'])
    parser.add_argument("-l", "--layers", help=f"Number of Transformer decoder layers. Default is {args['n_layers']}.", type=int, default=args['n_layers'])
    parser.add_argument("-d", "--dropout", help=f"The dropout rate. Default is {args['dropout']}.", type=int, default=args['dropout'])

    arg = parser.parse_args()

    # update hyperparameters config
    v, e, b, h, l, d = arg.vocab_size, arg.embedding_size, arg.block_size, arg.heads, arg.layers, arg.dropout
    args.update({
        'vocab_size':v,
        'block_size':b,
        'n_embd':e,
        'n_heads':h,
        'n_layers':l,
        'dropout':d
    })

    train_data, val_data = train_val_split(text)

    device = args['device']
    model = BabyLanguageModel(args)
    model.to(device)
    # print the number of parameters in the model
    print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')

    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args['lr'])

    train_and_infer(model, args, optimizer, device)

    # save model
    torch.save(model, './models/babyllm-gptlike.pt')

    # generate from the model
    # context = torch.zeros((1, 1), dtype=torch.long, device=device)
    context = torch.tensor(encode("Hey! How are you?"), dtype=torch.long, device=device)
    print(decode(model.generate(context, max_new_tokens=200)[0].tolist()))