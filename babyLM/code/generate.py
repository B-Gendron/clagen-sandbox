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
from torchtext.vocab import build_vocab_from_iterator, Vocab
from torchtext.data import get_tokenizer
tokenize = get_tokenizer("basic_english")

import numpy as np

from train import *


def generate_from_prompt(prompt_text):
    print(colored('Generating from a simple prompt', 'green'))
    prompt = torch.tensor(encode(vocab, prompt_text), dtype=torch.long, device=args['device']).unsqueeze(-1)
    print(f'Prompt: {prompt_text}')
    print(decode(model.generate(prompt, max_new_tokens=200, block_size=args['block_size'])[0].tolist()))


if __name__ == '__main__':
        
    model = torch.load('../models/babyllm-gptlike.pt')
    all_tokens = get_data('openwebtext')
    vocab = build_vocab_from_iterator(token_generator(), specials=["<unk>"], special_first=True)

    generate_from_prompt("Hey, how are you?")