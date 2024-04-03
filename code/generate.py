import torch
import torch.nn as nn

from transformers import AutoModelForCausalLM, AutoTokenizer, logging
logging.set_verbosity_error()
import os
from tqdm import tqdm
from termcolor import colored
import argparse

# from other scripts
from utils import *
from logging_utils import *

# disable hf tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# set default tensor type
torch.set_default_dtype(torch.float16)
# torch.backends.cuda.enable_mem_efficient_sdp(False)
# torch.backends.cuda.enable_flash_sdp(False)

def run_generations(args, experiment):
    '''
        A sequence of fine-tuning iterations for ontology-validation based fine-tuning. The aim is to understand and accurately generate different readability levels (RL) 

        @param args (str):                 the hyperparameters for the training
        @param experiment (str):           the experiment name

        @return trues (list):              list of gold labels to be stored later
        @return preds (list):              list of the associated predictions to be stored later
    '''
    trues, preds = [], []
    file_paths = []

    for i in tqdm(range(args['n_iter'])):
        # generate sentences with a specific RL
        batch_labels, batch_generations, _ = generate_from_random_prompts(args, hf=args['hf'])
        file_path = save_batch_generations(batch_generations, i, experiment)
        file_paths.append(file_path)

        # trues are the RL that the generated sentence should have
        trues.extend(batch_labels)
        create_batch_individual(i, file_path, experiment)
        generations_rl = get_readability_levels(f'../rdf/individual_batch_{i}_{experiment}.rdf')
        preds.extend(generations_rl)

    # save generated texts
    store_split_generations(f"{args['n_iter']*args['batch_size']}iters", file_paths, trues, experiment)


def run_exp(args, model_name, experiment):
    '''
        Run an end-to-end finetuning.

        @param args (dict):           the dict containing all the hyperparameters
        @param model_name (str):      either from a local storage (hf=False), or from huggingface hub (hf=True)
        @param experiment (str):      name of the experiment 
    '''
    print(colored(f'Experiment: {experiment}', 'green'))

    # create results dir if it doesn't exist
    if not os.path.exists(f'../results/{experiment}/'):
        os.makedirs(f'../results/{experiment}/')

    # instantiate 2 Llama models: one for generation and one for classification
    generation_model = AutoModelForCausalLM.from_pretrained(  
        model_name,
        low_cpu_mem_usage=True,         # recommanded param
        return_dict=True,               # not used for now
        torch_dtype=torch.bfloat16,     # bfloat instead of float because it may help
        device_map=args['device'],      # send to the right device
    )
    generation_model.post_init()

    # freeze both models
    for p in generation_model.parameters(): p.requires_grad = False

    # setup + save tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    generation_model.config.pad_token_id = generation_model.config.eos_token_id
    tokenizer.padding_side = "right"
    args.update({'max_new_tokens':20})
    args.update({'tokenizer':tokenizer})
    args.update({'gen_model': generation_model})
    generation_model.to(args['device'])

    # run training, val and test
    run_generations(args, experiment)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', help='The model name.', type=str, default="meta-llama/Llama-2-7b-chat-hf")
    arguments = parser.parse_args()

    args = {'vocab_size':239267,                # new vocab size corresponding to the new dataset
            'n_iter':1000,                      # number of generations
            'batch_size':10,                    # number of generations per iterations
            'device':activate_gpu(),            # set device for training. Desable force_cpu to run on gpu if available
            'n_embd':64,                        # embedding size
            'n_layers':24,                      # number of Transformer layers in the language model    
            'hf':True,                          # True is the model is loaded from huggingface models hub, false otherwise
        }
    
    model_name = arguments.model
    experiment_name = f"{model_name[model_name.find('/')+1:]}_generation_{get_datetime()}"

    run_exp(args, model_name, experiment_name)
