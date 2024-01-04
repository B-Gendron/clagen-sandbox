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
from torchtext.vocab import build_vocab_from_iterator, Vocab
from torchtext.data import get_tokenizer
tokenize = get_tokenizer("basic_english")

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
    
    
def get_data(folder_name):
    file_path = os.path.join(f'../{folder_name}', "data.txt")
    text_data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        # yield tokens from each line
        for line in f:
            # get initial words that are not empty words
            words = [ w for w in line.strip().split() if w != ' ']
            # make simple tokens by lowering words
            tokens = list(map(lambda x: x.lower(), words))
            text_data.extend(tokens)

    return text_data


def token_generator():
    for token in all_tokens:
        yield [token] # put the token inside a list so the vocab is not character-wise


def encode(vocab, text):
    '''
        From token strings to indexes

        @param vocab (torchtext.vocab.Vocab): the vocabulary object to use for stoi mapping
        @param text (str): the text to encode

        @return idxes (list of int): the list of indexes that correspond to the text encoding according to the underlying vocab
    '''
    stoi = vocab.get_stoi()
    # print(f"Token index: {stoi}")
    return [stoi[token.lower()] for token in text if token not in (' ', '\n')]

def decode(vocab, idxes):
    '''
        From vocab indexes to token strings

        @param vocab (torchtext.vocab.Vocab): the vocabulary object to use for itos mapping
        @param idexes (list of int): the list of indexes to be mapped to their associated tokens

        @return tokens (str): the concatenated tokens that form the encoded sentence
    '''
    # get torch vocab itos method
    itos = vocab.get_itos()
    print(f"Token string: {itos}")
    return ' '.join([itos[i] for i in idxes])

# Train and test splits
def train_val_split(vocab, data, train_ratio=0.9):
    '''
        Processes train/val split according to the given train_ratio (default = 90% of the dataset is used for training)
    '''
    tensor_data = torch.tensor(encode(vocab, data), dtype=torch.long)
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


def train_and_infer(model, args):
    '''
        To be documented.
    '''
    device = args['device']
    model.to(device)
    # print the number of parameters in the model
    print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')
    optimizer = torch.optim.AdamW(model.parameters(), lr=args['lr'])

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


def generate_from_prompt(prompt_text):
    print(colored('Generating from a simple prompt', 'green'))
    prompt = torch.tensor(encode(vocab, prompt_text), dtype=torch.long, device=args['device']).unsqueeze(-1)
    print(f'Prompt: {prompt_text}')
    print(decode(model.generate(prompt, max_new_tokens=200, block_size=args['block_size'])[0].tolist()))


if __name__ == "__main__":

    if not os.path.exists("../openwebtext/"): 
        subprocess.call(['sh', '../download_openwebtext.sh'])

    all_tokens = get_data('openwebtext')
    print(f"Number of tokens: {len(all_tokens)}")
    vocab = build_vocab_from_iterator(token_generator(), specials=["<unk>"], special_first=True)
    vocab_size = len(vocab)
    print(f'Vocab size: {vocab_size}')

    # hyperparameters default config
    args = {
        'vocab_size':vocab_size,
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

    # instantiate parser and retrieve model hyperparameters
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-v", "--vocab_size", help="The size of the vocabulary. Default is the number of unique characters in the training corpus.", type=int, default=vocab_size)
    parser.add_argument("-e", "--embedding_size", help=f"The embedding size. Default is {args['n_embd']}.", type=int, default=args['n_embd'])
    parser.add_argument("-b", "--block_size", help=f"The size of the Transformer decoder block, i.e. the maximum context length for predictions. Default is {args['block_size']}.", type=int, default=args['block_size'])
    parser.add_argument("-i", "--iters", help=f"The number of iterations (=epochs) for training. Default is {args['max_iters']}.", type=int, default=args['max_iters'])
    parser.add_argument("-h", "--heads", help=f"Number of attention heads. Default is {args['n_heads']}.", type=int, default=args['n_heads'])
    parser.add_argument("-l", "--layers", help=f"Number of Transformer decoder layers. Default is {args['n_layers']}.", type=int, default=args['n_layers'])
    parser.add_argument("-d", "--dropout", help=f"The dropout rate. Default is {args['dropout']}.", type=int, default=args['dropout'])

    arg = parser.parse_args()

    # update hyperparameters config
    v, e, b, i, h, l, d = arg.vocab_size, arg.embedding_size, arg.block_size, arg.iters, arg.heads, arg.layers, arg.dropout
    args.update({
        'vocab_size':v,
        'n_embd':e,
        'block_size':b,
        'max_iters':i,
        'n_heads':h,
        'n_layers':l,
        'dropout':d
    })

    # setup everything for training
    torch.manual_seed(42)
    train_data, val_data = train_val_split(vocab, all_tokens)
    model = BabyLanguageModel(args)

    # perform training and inference
    train_and_infer(model, args)

    # save model
    torch.save(model, '../models/babyllm-gptlike.pt')

    # generate from the model
    generate_from_prompt("Hey! How are you?")