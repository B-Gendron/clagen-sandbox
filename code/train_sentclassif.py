# torch utils
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# general purpose modules
import os
import numpy as np
from datasets import load_from_disk
from torch.utils.data import DataLoader
import argparse
from termcolor import colored

# from other scripts
from utils import *

# set default tensor type
torch.set_default_dtype(torch.float16)
torch.set_printoptions(precision=10)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)

class TweetAirlineLlamaTokenized(torch.utils.data.Dataset):
    '''
        Dataset class for twitter airline sentiment dataset
    '''
    def __init__(self, data, args):
        self._data = data
        self.args = args

    @property
    def data(self):
        return self._data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = {
            'sentiment':    np.array(self._data[idx]['sentiment']), 
            'input_ids':    np.array(self._data[idx]['embedding']) # this embedding is now the one from llama2 tokenizer !
        }
        return item
    

def get_dataloaders(args, dataset, dataset_class):
    '''
        Instantiate the training hyperparameters and the dataloaders.

        @param dataset:                     the data to put in the DataLoader
        @param dataset_class (Dataset):     the consistent dataset class from the datasets.py script to processed data

        @return args (dict):                a dictionary that contains the hyperparameters for training
        @return train_loader (dataloader):  the dataloader that contains the training samples
        @return val_loader (dataloader):    the dataloader that contains the validation samples
        @return test_loader (dataloader):   the dataloader that contains the test samples
    '''
    train_loader = DataLoader(dataset=dataset_class(dataset["train"], args=args), pin_memory=True, batch_size=args['train_bsize'], shuffle=True, drop_last=True)
    val_loader   = DataLoader(dataset=dataset_class(dataset["val"], args=args), pin_memory=True, batch_size=args['eval_bsize'], shuffle=True, drop_last=True)
    # test_loader  = DataLoader(dataset=dataset_class(dataset["test"], args=args), pin_memory=True, batch_size=args['eval_bsize'], shuffle=True, drop_last=True)
    return train_loader, val_loader


def train(args, finetuning_model, train_loader, ep):
    finetuning_model.train()
    device = args['device']
    loss_it, all_trues, all_preds = [], [], []
    ce_loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(finetuning_model.parameters(), lr=args['lr'], momentum=0.9, weight_decay=0.001)

    for it, batch in tqdm(enumerate(train_loader), desc="Epoch %s: " % (ep+1), total=train_loader.__len__()):

        batch = {'input_ids':batch['input_ids'].to(device), 'sentiment':batch['sentiment'].to(device)}

        print(batch['input_ids'].size(), batch['sentiment'].size())
        output_probas = finetuning_model(batch['input_ids']).logits
        loss = ce_loss(output_probas, batch['sentiment'])
        loss.backward()
        optimizer.step()

        trues = batch['sentiment'].tolist()
        preds = torch.argmax(output_probas, dim=1).tolist()
        all_trues.extend(trues), all_preds.extend(preds)

        loss_it.append(loss.item())
        optimizer.zero_grad()

    acc = accuracy_score(all_trues, all_preds)
    loss_it_avg = sum(loss_it)/len(loss_it)

    # print useful information about the training progress and scores on this training set's full pass
    print("Epoch %s/%s - %s : (%s %s) (%s %s)" % (colored(str(ep+1), 'blue'),args['max_eps'] , colored('Training', 'blue'), colored('Average loss: ', 'cyan'), loss_it_avg, colored('Accuracy: ', 'cyan'), acc))

def test(args, finetuning_model, loader, target):
    finetuning_model.eval()
    device = args['device']
    loss_it = []
    ce_loss = nn.CrossEntropyLoss()
    
    for it, batch in tqdm(enumerate(loader), total=loader.__len__()):

        batch = {'input_ids':batch['input_ids'].to(device), 'sentiment':batch['sentiment'].to(device)}

        output_probas = finetuning_model(batch['input_ids']).logits
        loss = ce_loss(output_probas, batch['sentiment'])
        loss_it.append(loss.item())

    loss_it_avg = sum(loss_it)/len(loss_it)

    # print useful information about the training progress and scores on this training set's full pass
    print("%s : (%s %s)" % (colored(target, 'blue'), colored('Average loss: ', 'cyan'), loss_it_avg))

def run_epochs(args, finetuning_model, train_loader, val_loader, experiment):
    val_losses = []

    for ep in range(args['max_eps']):
        # perform training and validation runs
        train(args, finetuning_model, train_loader, ep)
        test(args, finetuning_model, val_loader, 'Validation')

    return val_losses

def run_exp(args, model_name, train_loader, val_loader, experiment, episodes=10):
    '''
        Run an end-to-end finetuning.

        @param args (dict):           the dict containing all the hyperparameters
        @param model_name (str):      either from a local storage (hf=False), or from huggingface hub (hf=True)
        @param experiment (str):      name of the experiment 
        @param episodes (int):        number of times the test step should be performed (to compute descriptive stats on metrics)

        @return val_losses (list):    all val losses for all the 'epochs'
        @return test_losses (list):   all test losses from all the episodes
    '''
    print(colored(f'Start of the experiment {experiment}', 'green'))

    # create results dir if it doesn't exist
    if not os.path.exists(f'../results/{experiment}/'):
        os.makedirs(f'../results/{experiment}/')

    model = AutoModelForSequenceClassification.from_pretrained(  
        model_name,
        low_cpu_mem_usage=True,         # recommanded param
        return_dict=True,               # not used for now
        torch_dtype=torch.bfloat16,     # bfloat instead of float because it may help
        device_map=args['device'],      # send to the right device
    )
    model.to(args['device'])

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    args.update({'tokenizer':tokenizer})
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id
    tokenizer.padding_side = "right"

    args.update({'base_model': model}) # save the initial pretrained model without the adapters. This model will NOT be updated
    args.update({'model':model}) # save the model with the adapters that will be updated in fine-tuning
    args.update({'max_new_tokens':20}) # set max new tokens (TODO uniformizer args keys)

    # run training and validation
    val_losses = run_epochs(args, model, train_loader, val_loader, experiment)

    return val_losses


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", '--model', help="The name of the model to fine-tune. This is intended to be a path from huggingface models hub.", type=str, default="google-bert/bert-large-uncased")
    arg = parser.parse_args()

    MODEL_NAME = arg.model
    dd_llama_tokenized = load_from_disk("../data/tweet_airline_llama2_tokenized")

    args = {
            'train_bsize':32,
            'eval_bsize':8,
            'lr':1e-3,                          # learning rate
            'device':activate_gpu(),            # set device for training. Desable force_cpu to run on gpu if available
            'max_eps':40,                       # number of episodes (max of episodes in case of early stopping)
        }
    
    train_loader, val_loader = get_dataloaders(args, dd_llama_tokenized, TweetAirlineLlamaTokenized)
    run_exp(args, MODEL_NAME, train_loader, val_loader, 'finetune_bert_sentclassif')
