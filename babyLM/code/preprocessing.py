import datasets
import os 
import subprocess
from datasets import Dataset, load_dataset
from datasets.dataset_dict import DatasetDict
from sentence_transformers import SentenceTransformer
import argparse

from utils import *

# To be used after: check if the individuals (rdf) has been created
    # if os.path.exists("../../../OntoUttPreprocessing/rdf/wikitalk")


def apply_sentence_bert(entry, sentence_model, max_length):
    '''
        Apply a sentence transformer (S-BERT) model

        @param entry
        @param sentence_model
        @param max_length

        @param result
    '''
    utterance_limit = 15
    text = entry['text']
    dial_id = entry['dial_id']
    utt_id = entry['utt_id']
    embedding = sentence_model.encode(text, device=args['device']).tolist()

    # pad embeddings
    if len(embedding) < utterance_limit:
        embedding = add_sentencebert_random_vectors(embedding, utterance_limit - len(embedding), max_length)
    elif len(embedding) > utterance_limit:
        embedding = embedding[:utterance_limit]

    final_embedding = custom_flatten(embedding)

    result = {'dial_id': dial_id, 'utt_id': utt_id, 'embedding': final_embedding}
    return result


def prepare_data_sentence_bert(dataset, sentence_model, max_length):
    '''
        A function to wrap up the preprocessing procedure using sentence transformers (S-BERT);

        @param dataset
        @param sentence_model
        @param max_length

        @return resulting_dataset
    '''
    model = SentenceTransformer(f'sentence-transformers/{sentence_model}')

    for split in ['train', 'validation', 'test']:
        dataset[split] = dataset[split].map(lambda e: apply_sentence_bert(e, model, max_length))

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

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--pretrained", help="Name of the pretrained sentence Transformer model to use for data tokenization. Default is 'all-MiniLM-L6-v2'", default='minilm')
    arguments = parser.parse_args()

    # Load data if it does not already exist
    if not os.path.exists("../../../OntoUttPreprocessing/data"):
        subprocess.call(['sh', '../run_ontoUttPreprocessing.sh data'])

    wikitalk_utterances = datasets.load_from_disk("../../../OntoUttPreprocessing/data/processed_utterances_wikitalk")

    args = {'device': activate_gpu()}

    pretrained_mapping = {'minilm':'all-MiniLM-L6-v2', 'roberta':'all-roberta-large-v1', 'mpnet':'all-mpnet-base-v2'}
    max_lengths = {'all-MiniLM-L6-v2':384, 'all-roberta-large-v1':1024, 'all-mpnet-base-v2':768}
    sm = pretrained_mapping[arguments.pretrained]
    max_length = max_lengths[sm]

    print(colored(f"Pretrained model:{sm}", 'yellow'))
    tokenized_data = prepare_data_sentence_bert(wikitalk_utterances, sm, max_length)
    tokenized_data.save_to_disk('../wikitalk')