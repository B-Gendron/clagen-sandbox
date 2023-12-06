import numpy as np
import pandas as pd
import os
import subprocess
import math
import matplotlib.pyplot as plt
import argparse
import json
import datasets
from pprint import pprint
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split

# remove warning messages
import warnings
warnings.filterwarnings("ignore")

def load_and_process_dataset():
    if not os.path.isfile("./data/utterances.jsonl"): 
        subprocess.call(['sh', 'download_data.sh'])

    # read and format data in the right datatype
    utt_dataset = pd.read_json("./data/utterances.jsonl", lines=True)
    utt_dataset = utt_dataset.drop(columns=['user', 'meta', 'timestamp'])
    utt_dataset['replies'] = utt_dataset['reply-to'].values.astype('int')
    utt_dataset = utt_dataset.drop(columns=['reply-to'])

    return utt_dataset


def build_dialog_dataset(utterance_dataset):
    grouped_dataset = utterance_dataset.groupby(['root'])
    dialog_dataset = grouped_dataset.aggregate(lambda x: list(x))
    return dialog_dataset

def display_utterances_separately(dialog):
    for i in range(len(dialog)):
        print(f'------------ Utterance #{i} -----------\n{l[i]}')
        print('')

def split_and_save_hf_dataset(dataset):
    '''
        This function performs a 60/20/20 train/val/test split and save the output dataset into a DatasetDict Huggingface format.

        @param dataset (pandas DataFrame): the dataFrame to format, supposedly already formatted as expected for dialog analysis

        @return hf_data (DatasetDict): a DatasetDict object with 3 splits (train, validation, test) containing the same data than in `dataset`
    '''


if __name__=="__main__":

    utt_dataset = load_and_process_dataset()
    dialog_dataset = build_dialog_dataset(utt_dataset)

    # l = dialog_dataset['text'][1]
    # display_utterances_separately(l)
    baby_dataset = dialog_dataset.head(1000)

