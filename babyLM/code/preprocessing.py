import datasets
import os 
import subprocess
import json
from datasets import Dataset, load_dataset
from datasets.dataset_dict import DatasetDict
from sentence_transformers import SentenceTransformer
import argparse
from nltk.tokenize import TweetTokenizer
tokenizer = TweetTokenizer()

from utils import *
from train import encode, decode, load_vocab_mappings

# ce n'est pas ce qu'on veut ! On veut encoder avec le vocabulaire qui a servi pour le préentrainement de babyLM, comme ça ensuite on peut décoder et fournir le texte en prompt au modèle.
# on prend les entrées 1 par 1
    # - on encode
    # - on pad


def apply_sentence_bert(entry, stoi, max_length):
    '''
        Encode the text of one dialog using the vocabulary mapping that has been used to pretrained the BabyLanguageModel

        @param entry
        @param stoi
        @param max_length

        @param result
    '''
    utterance_limit = 10
    text = entry['text']
    dial_id = entry['dial_id']
    utt_id = entry['utt_id']

    # pad utterance ids
    n = len(utt_id)
    if n < utterance_limit:
        utt_id.extend([-1 for _ in range(utterance_limit-n)])
    elif n > utterance_limit:
        utt_id = utt_id[:utterance_limit]

    # encode sentences
    encoded_dialog = []
    for utterance in text:
        # perform encoding
        tokenized_utterance = tokenizer.tokenize(utterance.lower())
        encoded_utterance = encode(stoi, tokenized_utterance)

        # pad utterance
        n = len(encoded_utterance)
        if n < max_length:
            encoded_utterance.extend([-1 for _ in range(max_length-n)])
        elif n > max_length:
            encoded_utterance = encoded_utterance[:max_length]

        # store the padded utterance representation
        encoded_dialog.append(encoded_utterance)

    # pad at dialog level
    n = len(encoded_dialog)
    if n < utterance_limit:
        encoded_dialog.extend([[-1 for _ in range(max_length)] for _ in range(utterance_limit-n)])
    elif n > utterance_limit:
        encoded_dialog = encoded_dialog[:utterance_limit]

    result = {'dial_id': dial_id, 'utt_id': utt_id, 'embedding': encoded_dialog}
    return result


def prepare_data(dataset, stoi, max_length):
    '''
        A function to wrap up the preprocessing procedure using sentence transformers (S-BERT);

        @param dataset
        @param stoi
        @param max_length

        @return resulting_dataset
    '''

    for split in ['train', 'validation', 'test']:
        dataset[split] = dataset[split].map(lambda e: apply_sentence_bert(e, stoi, max_length))

    processed_dataset = {
        'train':Dataset.from_dict({
            'dial_id': dataset['train']['dial_id'],
            'utt_id': dataset['train']['utt_id'],
            'embedding': dataset['train']['embedding']
            }),
        'validation': Dataset.from_dict({
            'dial_id': dataset['validation']['dial_id'],
            'utt_id': dataset['validation']['utt_id'],
            'embedding': dataset['validation']['embedding']
            }),
        'test':Dataset.from_dict({
            'dial_id': dataset['test']['dial_id'],
            'utt_id': dataset['test']['utt_id'],
            'embedding': dataset['test']['embedding']
            })
        }
    
    resulting_dataset = DatasetDict(processed_dataset)
    return resulting_dataset

if __name__ == '__main__':

    itos, stoi = load_vocab_mappings()

    # Load data if it does not already exist
    if not os.path.exists("../../../OntoUttPreprocessing/data"):
        subprocess.call(['sh', '../run_ontoUttPreprocessing.sh data'])

    wikitalk_utterances = datasets.load_from_disk("../../../OntoUttPreprocessing/data/processed_utterances_wikitalk")

    args = {'device': activate_gpu()}

    max_length = 256

    print(colored(f"Start preprocessing...", 'yellow'))
    tokenized_data = prepare_data(wikitalk_utterances, stoi, max_length)
    tokenized_data.save_to_disk('../wikitalk')