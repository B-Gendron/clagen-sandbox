import datasets
import os 
import subprocess
import json
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets.dataset_dict import DatasetDict
import argparse

from utils import *
from train_adapters import MODEL_NAME


def apply_lm_tokenizer(entry, tokenizer, args):
    '''
        Uses the desired tokenizer to encode dialogues from wikitalk dataset

        @param entry
        @param stoi
        @param max_length

        @param result
    '''
    # setting preprocessing params
    utterance_limit = 10
    max_length = args['max_length']
    text = entry['dialog']
    emotions = entry['emotion']

    # pad emotions
    n = len(emotions)
    if n < utterance_limit:
        emotions.extend([-1 for _ in range(utterance_limit-n)])
    elif n > utterance_limit:
        emotions = emotions[:utterance_limit]

    # encode sentences
    encoded_dialog = []
    for utterance in text:
        # perform encoding
        encoded_utterance = tokenizer(utterance)['input_ids']

        # pad utterance
        n = len(encoded_utterance)
        if n < max_length:
            encoded_utterance.extend([0 for _ in range(max_length-n)])
        elif n > max_length:
            encoded_utterance = encoded_utterance[:max_length]

        # store the padded utterance representation
        encoded_dialog.append(encoded_utterance)

    # pad at dialog level
    n = len(encoded_dialog)
    if n < utterance_limit:
        encoded_dialog.extend([[0 for _ in range(max_length)] for _ in range(utterance_limit-n)])
    elif n > utterance_limit:
        encoded_dialog = encoded_dialog[:utterance_limit]

    result = {'emotion': emotions, 'embedding': encoded_dialog}
    return result


def prepare_data(dataset, stoi, args):
    '''
        A function to wrap up the preprocessing procedure using sentence transformers (S-BERT);

        @param dataset
        @param stoi
        @param max_length

        @return resulting_dataset
    '''
    for split in ['train', 'validation', 'test']:
        dataset[split] = dataset[split].map(lambda e: apply_lm_tokenizer(e, stoi, args))

    processed_dataset = {
        'train':Dataset.from_dict({
            'emotion': dataset['train']['emotion'],
            'embedding': dataset['train']['embedding']
            }),
        'validation': Dataset.from_dict({
            'emotion': dataset['validation']['emotion'],
            'embedding': dataset['validation']['embedding']
            }),
        'test':Dataset.from_dict({
            'emotion': dataset['test']['emotion'],
            'embedding': dataset['test']['embedding']
            })
        }
    
    resulting_dataset = DatasetDict(processed_dataset)
    return resulting_dataset

if __name__ == '__main__':

    # Load data if it does not already exist
    if not os.path.exists("../../OntoUttPreprocessing/data"):
        subprocess.call(['sh', '../run_ontoUttPreprocessing.sh data'])

    dailydialog = load_dataset('daily_dialog')
    print(dailydialog['train'][0])
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    args = {'max_length':20}

    print(colored(f"Start preprocessing...", 'yellow'))
    tokenized_data = prepare_data(dailydialog, tokenizer, args)
    print(tokenized_data['train'][0])
    tokenized_data.save_to_disk('../dd_llama2_tokenized')