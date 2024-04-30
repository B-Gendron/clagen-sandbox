import datasets
import argparse
from datasets import Dataset
from transformers import AutoTokenizer
from datasets.dataset_dict import DatasetDict
from sklearn.model_selection import train_test_split
import pandas as pd
from utils import *


def apply_lm_tokenizer(entry, tokenizer, args):
    '''
        Uses the desired tokenizer to encode dialogues from wikitalk dataset

        @param entry
        @param stoi
        @param max_length

        @param result
    '''
    # setting preprocessing params
    max_length = args['max_length']
    text = entry['text']
    emotion = entry['sentiment']

    # perform encoding
    encoded_utterance = tokenizer(text)['input_ids']

    # pad utterance
    n = len(encoded_utterance)
    if n < max_length:
        encoded_utterance.extend([0 for _ in range(max_length-n)])
    elif n > max_length:
        encoded_utterance = encoded_utterance[:max_length]

    result = {'sentiment': 0 if emotion=='negative' else 1, 'embedding': encoded_utterance}
    return result


def prepare_data(dataset, stoi, args):
    '''
        A function to wrap up the preprocessing procedure using sentence transformers (S-BERT);

        @param dataset
        @param stoi
        @param max_length

        @return resulting_dataset
    '''
    # map each utterance to its encoding using llama2 tokenizer
    for split in ['train', 'val']:
        dataset[split] = dataset[split].map(lambda e: apply_lm_tokenizer(e, stoi, args))

    processed_dataset = {
        'train':Dataset.from_dict({
            'sentiment': dataset['train']['sentiment'],
            'embedding': dataset['train']['embedding']
            }),
        'val': Dataset.from_dict({
            'sentiment': dataset['val']['sentiment'],
            'embedding': dataset['val']['embedding']
            })
        }
    
    resulting_dataset = DatasetDict(processed_dataset)
    return resulting_dataset


def get_data_tokenizer_args(model_name):
    # data for binary sentiment analysis
    data = pd.read_csv('/kaggle/input/twitter-airline-sentiment/Tweets.csv')
    data = data[['text','airline_sentiment']]
    data = data[data.airline_sentiment != "neutral"]

    positive_df = data[data['airline_sentiment'] == 'positive']
    negative_df = data[data['airline_sentiment'] == 'negative']

    # Determine the number of samples needed from each DataFrame to achieve 50/50 distribution
    num_samples = min(len(positive_df), len(negative_df))

    # Sample from each DataFrame
    positive_sampled = positive_df.sample(n=num_samples, random_state=42)
    negative_sampled = negative_df.sample(n=num_samples, random_state=42)

    # Concatenate the sampled DataFrames back together
    balanced_df = pd.concat([positive_sampled, negative_sampled])

    # Shuffle the concatenated DataFrame to randomize the order
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

    X_train, X_test, y_train, y_test = train_test_split(balanced_df['text'], balanced_df['airline_sentiment'], random_state=42, test_size=0.2)
    sentiment_dataset = {
    'train':Dataset.from_dict({
        'sentiment': y_train,
        'text':X_train
        }),
    'val': Dataset.from_dict({
        'sentiment': y_test,
        'text': X_test
        })
    }
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    args = {'max_length':20}

    return sentiment_dataset, tokenizer, args

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', help="The name of the model to fine-tune. This is intended to be a path from huggingface models hub.", type=str, default="google-bert/bert-large-uncased")
    arguments = parser.parse_args()

    MODEL_NAME = arguments.model
    PATH = f'../data/{MODEL_NAME}_tokenized'
    sentiment_dataset, tokenizer, args = get_data_tokenizer_args()

    print(colored(f"Start preprocessing...", 'yellow'))
    tokenized_data = prepare_data(sentiment_dataset, tokenizer, args)

    print(colored('Preprocessing finished. Saving dataset...', 'yellow'))
    tokenized_data.save_to_disk(PATH)

    print(colored(f'Dataset saved at {PATH}. An example of preprocessed data sample:', 'yellow'))
    print(tokenized_data['train'][0])