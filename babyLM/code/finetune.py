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
from datasets import load_dataset
from torch.utils.data import Dataset
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

class WikiTalkDataset(torch.utils.data.Dataset):
    '''
        Dataset class for wikitalk pages dataset
    '''
    def __init__(self, data):
        self._data = data

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
    
