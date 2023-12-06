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

# -----------------------------------------------------------------------------------------
# HuggingFace datasets loading and saving 
# -----------------------------------------------------------------------------------------

def load_dataset(dataset_name):
    '''
        Load a dataset save in the huggingface format, located in the data folder.

        @param dataset_name (str): the name of the dataset, i.e. the associated folder inside the data folder. This string should $

        @return dataset (DatasetDict): the dataset as a DatasetDict object.
    '''
    path = f"./data/{dataset_name}"
    if os.path.isdir(path):
        dataset = DatasetDict.load_from_disk(f"./data/{dataset_name}")
    else:
        raise Exception("No folder with the such name found in the data folder.")

    return dataset


def save_dataset(dataset, dataset_name, output_format='huggingface'):
    '''
        Save the dataset into a HuggingFace format using the save_to_disk method.

        @param dataset (DatasetDict):      the dataset to save in a DatasetDict format
        @param dataset_name (str):         the name to give to the file
        @param output_format (str):        either 'huggingface' or 'json'. Default is 'huggingface'.
    '''
    if output_format == "huggingface":
        dataset.save_to_disk(f'./data/{dataset_name}')
    elif output_format == "json":
        with open(f"data/{dataset_name}.json", 'w') as f:
            json.dump(dataset, f)
    else:
        print("The given output format is not recognized. Please note that accepted formats are 'huggingface' and 'json")


# -----------------------------------------------------------------------------------------
# Wikidata dataset preprocessing utils 
# -----------------------------------------------------------------------------------------

def load_and_process_dataset():
    if not os.path.isfile("./data/utterances.jsonl"): 
        subprocess.call(['sh', 'download_data.sh'])

    # read and format data in the right datatype
    utt_dataset = pd.read_json("./data/utterances.jsonl", lines=True)
    utt_dataset = utt_dataset.drop(columns=['user', 'meta', 'timestamp'])
    utt_dataset['replies_to'] = utt_dataset['reply-to'].values.astype('int')
    utt_dataset = utt_dataset.drop(columns=['reply-to'])

    return utt_dataset


def build_dialog_dataset(utterance_dataset):
    grouped_dataset = utterance_dataset.groupby(['root'], as_index=False) # as_index=False to keep the root column
    dialog_dataset = grouped_dataset.aggregate(lambda x: list(x))
    dialog_dataset = dialog_dataset.rename(columns={'root':'dial_id', 'id':'utt_id'})
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
    # perform splitting
    train, val = train_test_split(dataset, test_size=0.4, random_state=42)
    val, test = train_test_split(val, test_size=0.5, random_state=42)
    # print(train.shape, val.shape, test.shape) # check each subset size

    hf_data = DatasetDict()
    hf_data['train'] = Dataset.from_dict(train)
    hf_data['validation'] = Dataset.from_dict(val)
    hf_data['test'] = Dataset.from_dict(test)

    # save dataset
    save_dataset(hf_data, 'processed_utterances')


if __name__=="__main__":

    utt_dataset = load_and_process_dataset()
    dialog_dataset = build_dialog_dataset(utt_dataset)

    # l = dialog_dataset['text'][1]
    # display_utterances_separately(l)
    baby_dataset = dialog_dataset.head(1000)

    split_and_save_hf_dataset(baby_dataset)