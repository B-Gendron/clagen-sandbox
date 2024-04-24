import os
from owlready2 import *
from termcolor import colored
from pprint import pprint
from datetime import datetime
import torch
import numpy as np
import logging
from datasets import Dataset, DatasetDict
from collections import Counter
import multiprocess as mp
import threading
import json
import re
import csv
from numpy import random as rd
import random
from math import sqrt
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_auc_score, f1_score
from nltk.tokenize import TweetTokenizer
tokenizer = TweetTokenizer()

import warnings
warnings.filterwarnings('ignore')

# set default tensor type
torch.set_default_dtype(torch.float16)

from models import BabyLanguageModel, TrainableHead

# -----------------------------------------------------------------------------------------
# General purpose auxiliary functions
# -----------------------------------------------------------------------------------------

def get_datetime():
    '''
        This function gets the current date time and returns it as a string.

        @returns dt_string (str): current time formatted in a string
    '''
    now = datetime.now()
    dt_string = now.strftime("%d%m%Y%H%M%S")
    return dt_string


def args2filename(dico):
    '''
        This function builds a file name regarding the parameters of the experiment given in a dictionnary

        @param dico (dict): a parameters dictionnary

        @returns filename (str): a string to name a file regarding the parameters nature and values
    '''
    filename = "_".join([f"{k}{v}" for k,v in dico.items()])
    return filename


def custom_flatten(ll):
    '''
        A function to flatten a list of lists where sub lists are of heterogeneous sizes.
        @param ll (list): the input list of lists
        @return l (list): the flattened list   
    '''
    l = []
    for sl in ll:
        if type(sl) is not list:
            sl = [sl]
        l.extend(sl)
    return l


def custom_flatten_sep(ll):
    '''
        A function to flatten a list of lists where sub lists are of heterogeneous sizes. This variant of the above function keeps a track of where the sublists have been concatenated by adding a [SEP] marker as a float 0.0

        @param ll (list): the input list of lists

        @return l (list): the flattened list   
    '''
    l = []
    for sl in ll:
        l.extend(sl + [0.0])
    return l[:len(l)-1]


def concatenate(iterable, sep=""):
    sentence = iterable[0]
    for word in iterable[1:]:
        sentence += (sep + word)
    return sentence


def activate_gpu(force_cpu=False):
    '''
        A function to return the right device depending on GPU availability
    '''
    device = "cpu"
    if not force_cpu:
        if torch.cuda.is_available():
            device = 'cuda'
            print('DEVICE = ', colored(torch.cuda.get_device_name(0), "green" ) )
        elif torch.backends.mps.is_available():
            device = 'mps'
            print('DEVICE = ', colored("mps", "green" ) )
        else:
            device = 'cpu'
            print('DEVICE = ', colored('CPU', "blue"))
    return device

def pick_list(layers_list):
    if layers_list == 1:
        return [20, 22, 24, 26]
    elif layers_list == 2:
        return [13, 17, 21, 25]
    else:
        return [7, 12, 17, 22]
    
def hf_model_name(s):
    '''
        This function takes the name of a huggingface model and isolate the first part of the name of the model, after the first slash (/) and before the first dash (-).
        The result is given in lower case.

        This function is useful to generate a parameter name that can be evaluated during fine-tuning steps when processing differs depending on the model. For instance, when doing weight transfer, the layers are not named the same in different models.

        @s (str):       the string of the huggingface model name
    '''
    return s.split('/')[1].split('-')[0].lower()

# -----------------------------------------------------------------------------------------
# Preprocessing utils for training
# -----------------------------------------------------------------------------------------

def encode(stoi, text):
    '''
        From token strings to indexes

        @param stoi (dict):             string to index mapping from the vocab
        @param text (str):              the text to encode

        @return idxes (list of int):    the list of indexes that correspond to the text encoding according to the underlying vocab
    '''
    encoding = []
    for token in text:
        if token not in (' ', '\n'):
            try:
                stoi[token.lower()]
            except KeyError:
                encoding.append(stoi['<unk>']) 
            else:
                encoding.append(stoi[token.lower()])
                
    return encoding

def decode(idxes, itos):
    '''
        From vocab indexes to token strings

        @param idexes (list of int): the list of indexes to be mapped to their associated tokens
        @param itos (dict): index to string mapping from the vocab

        @return tokens (str): the concatenated tokens that form the encoded sentence
    '''
    return ' '.join([itos[i] for i in idxes])


def load_vocab_mappings():
    with open("../objects/vocab_itos.json") as f:
        itos = json.load(f)

    with open("../objects/vocab_stoi.json") as f:
        stoi = f.read()
    stoi = json.loads(stoi)

    return itos, stoi

# -----------------------------------------------------------------------------------------
# Fine-tuning utils
# -----------------------------------------------------------------------------------------

def select_target_modules(target_modules, selection):
    # Create a dictionary mapping first letters to full module names
    module_dict = {module[0]: module for module in target_modules}
    
    # Filter modules based on the first letters in the param
    subset = [module_dict[letter] for letter in selection if letter in module_dict]
    
    return subset


def parse_indexes(levels_dict):
    '''
        This auxiliary function parses and retrieves the utterance indexes from the readability level dictionnary when an utterance is reffered to using the ontology identifier. It return a dictionary with the same format than the input, containing only the indexes instead of the complete identifier.

        @param levels_dict (dict):          the dict from the ontology individual search.
        
        @return levels_indexes_dict (dict): a dict that maps the readability levels to the associated utterance indexes in the current dialog.
    '''
    levels_indexes_dict = {}

    # iterate through the input dict
    for k in levels_dict.keys():
        for v in levels_dict[k]:
            # parse the utterance full name to retrieve its index
            splitted_v = v.split('_')[::-1]
            index = splitted_v[2] # the 3rd element when individual_592.592_598_utt_21271711 is splitted by _ is 598, which is the utterance index
            # store the readability class for the current index
            levels_indexes_dict[int(index)] = k

    result_dict = dict(sorted(levels_indexes_dict.items()))

    # DEPRECATED CAUSE NO LONGER WORKING WITH DIALOGUES HERE remove the keys after the 10 first utterances to be consistent with padding strategy
    # if len(result_dict) > 10:
    #     result_dict = {k:result_dict[k] for k in list(result_dict.keys())[:10]}

    return result_dict


def get_readability_levels(indiv_path):
    '''
        TODO update doc
        A généraliser ! Pareil pour la fonction suivante

        @param indiv_path (str):    the path to the batch ontology individual.     
    '''
    # get labels mapping for different readability level
    readability_levels_mapping = {'EasilyReadableText':0, 'StandardReadableText':1, 'HardlyReadableText':2}
    # retrieve the ontology individual
    individual = get_ontology(indiv_path).load()
    # crop first element as it simply corresponds to the class occurence
    utterance_levels = {
        'EasilyReadableText':       [str(i) for i in individual.search(is_a=individual.EasilyReadableText)[1:]],
        'StandardReadableText':     [str(i) for i in individual.search(is_a=individual.StandardReadableText)[1:]],
        'HardlyReadableText':       [str(i) for i in individual.search(is_a=individual.HardlyReadableText)[1:]]
    }
    utterance_levels = parse_indexes(utterance_levels)
    labels = [readability_levels_mapping[v] for v in utterance_levels.values()]

    individual.destroy()

    return labels


def get_sentence_length(indiv_path):
    '''
        TODO update doc
        A généraliser ! Pareil pour la fonction suivante

        @param indiv_path (str):    the path to the batch ontology individual.     
    '''
    # get labels mapping for different sentence lengths
    sentence_length_mapping = {'ShortFullText':0, 'LongFullText':1}
    # retrieve the ontology individual
    individual = get_ontology(indiv_path).load()
    # crop first element as it simply corresponds to the class occurence
    utterance_levels = {
        'ShortFullText':        [str(i) for i in individual.search(is_a=individual.ShortFullText)[1:]],
        'LongFullText':         [str(i) for i in individual.search(is_a=individual.LongFullText)[1:]]
    }
    utterance_levels = parse_indexes(utterance_levels)
    labels = [sentence_length_mapping[v] for v in utterance_levels.values()]

    individual.destroy()

    return labels

def random_prompt(concept, classes, hf=False):
    '''
        This auxiliary function allows to get a prompt that ask for a sentence belonging to a certain class among given classes. It is for now used for readability levels but is meant for a more general purpose.

        @param concept (str):               the name of the ontology concept we want to learn, that is divided into the following classes
        @param classes (list):              a list or strings giving the names of all the classes 

        @return prompt (str):               the randomly selected prompt to use for generation
        @return class_index (int):          the index of the corresponding class that is asked in the prompt 
        @param hf:                          False in case of local model, a huggingface model alias otherwise. Currently only 'llama' is supported
    '''
    start_of_sentence = ['A', 'The', 'Yesterday', 'Hello', 'Here', 'In', 'For', 'Yes', 'Tomorrow', 'Today']
    p = rd.uniform()
    n = len(classes)
    for k in range(1, n+1):
        if (k-1)/n < p < k/n:
            if hf:
                prompt = f"This is a sentence for which concept {concept} is {classes[k-1]}: {rd.choice(np.array(start_of_sentence))}"
                # prompt = f"INSTRUCTION: Give an example sentence for which concept {concept} is {classes[k-1]}. ANSWER: Here is an example sentence for the given concept: '{rd.choice(np.array(start_of_sentence))}"
                return prompt, k-1
            else: 
                prompt = f"A sentence whose {concept} is {classes[k-1]}: The"
                return prompt, k-1


def generate_from_random_prompts(args, hf=False):
    # concept = 'ReadabilityLevel'
    # classes = ['EasyReadableText', 'StandardReadableText', 'HardlyReadableText']
    # classes = ['LongFullText', 'ShortFullText']

    # try with random IDs instead of concept names
    concept = 'C6468168'
    classes = [f'{concept}{i}' for i in range(2)]
    batch_labels, batch_generations, batch_ids = [], [], []

    if hf:
        tokenizer = args['tokenizer']
        model = args['gen_model']
        for i in range(args['batch_size']):
            # get a randomly selected prompt (uniform law)
            prompt, label = random_prompt(concept, classes, hf=hf)
            batch_labels.append(label)
            # perform generation
            prompt = tokenizer(prompt, return_tensors="pt").to(args['device'])
            output = model.generate(**prompt, max_new_tokens=args['max_new_tokens'], repetition_penalty=1.5)[0] # this contains the prompt and the generated part
            result = tokenizer.decode(output, skip_special_tokens=True)
            # generation = result[result.find('concept:')+len('concept'):]
            generation = result[result.find(':')+1:result.find('\n')]
            # try to give the classifier model both prompt and generated sentence to access adequacy between RLv and sentence --> bad idea (OOM) so back to initial setup
            output_ids = get_and_pad_ids(tokenizer(generation, return_tensors="pt").to(args['device'])['input_ids'], args, padding_length=40)
            # print(output_ids)
            batch_ids.append(output_ids)
            print(f'Sample {i}: \t | Class {label} | \t {generation}')
            # store result
            batch_generations.append(generation)

    else:
        itos, stoi = load_vocab_mappings()
        model = args['model']
        for _ in range(args['batch_size']):
            # get a randomly selected prompt (uniform law)
            prompt, label = random_prompt(concept, classes, hf=hf)
            batch_labels.append(label)
            # encode the prompt
            prompt = encode(stoi, tokenizer.tokenize(prompt))
            prompt = torch.tensor(prompt, dtype=torch.float16).unsqueeze(-1).to(args['device'])
            # perform generation
            generation = model.generate(prompt, max_new_tokens=20, block_size=args['block_size'])[0] # increase max_new_tokens to generate HardlyReadableText
            generation = decode(generation.tolist(), itos)
            # store result
            batch_generations.append(generation)

    return batch_labels, batch_generations, batch_ids

def get_and_pad_ids(output, args, padding_length=40):
    '''
        To be documented.
    '''
    current_length = output.shape[1]
    
    if current_length >= padding_length:
        return output[:, :padding_length]
    
    padding_size = padding_length - current_length
    padding = torch.zeros((1, padding_size), dtype=output.dtype, device=args['device'])
    
    padded_output = torch.cat((output, padding), dim=1)
    return padded_output


def is_same(trues, preds):
    '''
        This function is intented to be used a finetuning time. 
        It compares the desired labels (i.e. readability levels in our case) to the actual class of the generated sentences. 
        It returns, for each element of the batch, 1 if true/pred classes are identical, 0 otherwise.

        @param trues (list): the expected readability levels in our case
        @param preds (list): the readability levels of generated sentences in our case
    '''
    return [1 if p == t else 0 for p, t in zip(preds, trues)]


def update_adapter_weights(args, g, c):
    lora_config = args['config']
    layers_from_config = lora_config.layers_to_transform
    layers = layers_from_config if layers_from_config is not None else range(args['n_layers'])

    if args['hf_model'] == 'llama':
        for i in layers:
            if 'q' in args['target_modules']:
                g.base_model.model.model.layers[i].self_attn.q_proj.lora_A.default.weight = c.base_model.model.model.layers[i].self_attn.q_proj.lora_A.default.weight
                g.base_model.model.model.layers[i].self_attn.q_proj.lora_B.default.weight = c.base_model.model.model.layers[i].self_attn.q_proj.lora_B.default.weight
            if 'k' in args['target_modules']:
                g.base_model.model.model.layers[i].self_attn.k_proj.lora_A.default.weight = c.base_model.model.model.layers[i].self_attn.k_proj.lora_A.default.weight
                g.base_model.model.model.layers[i].self_attn.k_proj.lora_B.default.weight = c.base_model.model.model.layers[i].self_attn.k_proj.lora_B.default.weight
            if 'v' in args['target_modules']:
                g.base_model.model.model.layers[i].self_attn.v_proj.lora_A.default.weight = c.base_model.model.model.layers[i].self_attn.v_proj.lora_A.default.weight
                g.base_model.model.model.layers[i].self_attn.v_proj.lora_B.default.weight = c.base_model.model.model.layers[i].self_attn.v_proj.lora_B.default.weight
            if 'o' in args['target_modules']:
                g.base_model.model.model.layers[i].self_attn.o_proj.lora_A.default.weight = c.base_model.model.model.layers[i].self_attn.o_proj.lora_A.default.weight
                g.base_model.model.model.layers[i].self_attn.o_proj.lora_B.default.weight = c.base_model.model.model.layers[i].self_attn.o_proj.lora_B.default.weight

    elif args['hf_model'] == 'flan':
        for i in layers:
            if 'q' in args['target_modules']:
                g.base_model.model.encoder.block[i].layer[0].SelfAttention.q.lora_A.default.weight = c.base_model.model.transformer.encoder.block[i].layer[0].SelfAttention.q.lora_A.default.weight
                g.base_model.model.encoder.block[i].layer[0].SelfAttention.q.lora_B.default.weight = c.base_model.model.transformer.encoder.block[i].layer[0].SelfAttention.q.lora_B.default.weight
            if 'k' in args['target_modules']:
                g.base_model.model.encoder.block[i].layer[0].SelfAttention.k.lora_A.default.weight = c.base_model.model.transformer.encoder.block[i].layer[0].SelfAttention.k.lora_A.default.weight
                g.base_model.model.encoder.block[i].layer[0].SelfAttention.k.lora_B.default.weight = c.base_model.model.transformer.encoder.block[i].layer[0].SelfAttention.k.lora_B.default.weight
            if 'v' in args['target_modules']:
                g.base_model.model.encoder.block[i].layer[0].SelfAttention.v.lora_A.default.weight = c.base_model.model.transformer.encoder.block[i].layer[0].SelfAttention.v.lora_A.default.weight
                g.base_model.model.encoder.block[i].layer[0].SelfAttention.v.lora_B.default.weight = c.base_model.model.transformer.encoder.block[i].layer[0].SelfAttention.v.lora_B.default.weight
            if 'o' in args['target_modules']:
                g.base_model.model.encoder.block[i].layer[0].SelfAttention.o.lora_A.default.weight = c.base_model.model.transformer.encoder.block[i].layer[0].SelfAttention.o.lora_A.default.weight
                g.base_model.model.encoder.block[i].layer[0].SelfAttention.o.lora_B.default.weight = c.base_model.model.transformer.encoder.block[i].layer[0].SelfAttention.o.lora_B.default.weight


def setup_model_babylm(args, model_name):

    # load pretrained model weights
    model = BabyLanguageModel(args)
    if args['device'] == 'cpu':
        model.load_state_dict(torch.load(model_name, map_location=torch.device('cpu')))
    else:
        model.load_state_dict(torch.load(model_name))
    model.to(args['device'])

    # freeze all layers except model.lm_head
    for p in model.token_embedding_table.parameters(): p.requires_grad = False
    for p in model.position_embedding_table.parameters(): p.requires_grad = False
    for p in model.blocks.parameters(): p.requires_grad = False
    for p in model.ln_f.parameters(): p.requires_grad = False
    for p in model.lm_head.parameters(): p.requires_grad = True
    
    # store model
    args.update({'model':model})

    # setup finetuning model
    finetuning_model = TrainableHead(args)
    for p in finetuning_model.pool.parameters(): p.requires_grad = False
    for p in finetuning_model.anti_pool.parameters(): p.requires_grad = False
    for p in finetuning_model.lm_head.parameters(): p.requires_grad = True
    finetuning_model.lm_head.weight = model.lm_head.weight

    return finetuning_model


def create_batch_individual(batch_index, file_path, experiment):
    command = ["../call_ontology.sh", str(batch_index), file_path, experiment]

    # Run the command using subprocess
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")


def remove_zero_filled_subtensors(tensor, labels):
    non_zero_indices = torch.nonzero(~torch.all(tensor == 0, dim=1)).squeeze()
    non_zero_labels = [labels[i] for i in non_zero_indices]
    return tensor[non_zero_indices], non_zero_labels
