import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import json
import pickle
import argparse
import io
import pandas as pd
from termcolor import colored
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data import get_tokenizer
tokenize = get_tokenizer("basic_english")

from train import *

class CPU_Unpickler(pickle.Unpickler):
    """
        This class is supposed to help running something on CPU when it is based on objects computed on GPU. 
        From: https://github.com/pytorch/pytorch/issues/16797#issuecomment-633423219 
    """
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)



def generate_from_prompt(prompt_text, model, stoi, itos, args):
    device = args['device']
    block_size = args['block_size']
    print(colored('Generating from a simple prompt', 'green'))
    prompt = torch.tensor(encode(stoi, prompt_text), dtype=torch.long, device=device).unsqueeze(-1)
    prompt.to(device)
    print(f'Prompt: {prompt_text}')
    print(f'Encoded prompt: {prompt}')
    output = model.generate(prompt, max_new_tokens=200, block_size=block_size)[0].tolist()
    print(f'Encoded output: {output}')
    output_text = decode(output, itos)
    print(f'Output text: {output_text}')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--model_name", help="The name of the model to load for generation.", type=str, required=True)
    arguments = parser.parse_args()
    model_name = arguments.model_name

    args = {
        'device':'cuda',
        'block_size':64
    }

    print("Getting stoi and itos dicts...")
    stoi = json.load("../objects/vocab_stoi.json")
    itos = json.load("../objects/vocab_itos.json")

    print("Getting model...")
    model = torch.load(f'../models/{model_name}.pt')

    print("Start generation")
    generate_from_prompt("Once upon a time", model, vocab, args)