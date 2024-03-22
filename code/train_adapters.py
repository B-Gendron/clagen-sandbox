# torch utils
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model, peft_model, TaskType

# general purpose modules
import os
import numpy as np
from datasets import load_from_disk
from torch.utils.data import DataLoader
import argparse
from termcolor import colored

# from other scripts
from utils import *
from models import TrainableHeadAdapters

# set default tensor type
# torch.set_default_dtype(torch.float32)
# torch.set_printoptions(precision=10)
# torch.backends.cuda.enable_mem_efficient_sdp(False)
# torch.backends.cuda.enable_flash_sdp(False)

MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"

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
    

def get_args_and_dataloaders(dataset, dataset_class):
    '''
        Instantiate the training hyperparameters and the dataloaders.

        @param dataset:                     the data to put in the DataLoader
        @param dataset_class (Dataset):     the consistent dataset class from the datasets.py script to processed data

        @return args (dict):                a dictionary that contains the hyperparameters for training
        @return train_loader (dataloader):  the dataloader that contains the training samples
        @return val_loader (dataloader):    the dataloader that contains the validation samples
        @return test_loader (dataloader):   the dataloader that contains the test samples
    '''
    args = {'train_bsize': 128, 'eval_bsize': 8}
    train_loader = DataLoader(dataset=dataset_class(dataset["train"], args=args), pin_memory=True, batch_size=args['train_bsize'], shuffle=True, drop_last=True)
    val_loader   = DataLoader(dataset=dataset_class(dataset["val"], args=args), pin_memory=True, batch_size=args['eval_bsize'], shuffle=True, drop_last=True)
    # test_loader  = DataLoader(dataset=dataset_class(dataset["test"], args=args), pin_memory=True, batch_size=args['eval_bsize'], shuffle=True, drop_last=True)
    return args, train_loader, val_loader


def train(args, finetuning_model, train_loader, ep):
    finetuning_model.train()
    device = args['device']
    loss_it, all_trues, all_preds = [], [], []
    ce_loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(finetuning_model.parameters(), lr=args['lr'], momentum=0.9, weight_decay=0.001)

    for it, batch in tqdm(enumerate(train_loader), desc="Epoch %s: " % (ep+1), total=train_loader.__len__()):

        batch = {'input_ids':batch['input_ids'].to(device), 'sentiment':batch['sentiment'].to(device)}

        # check if there is at least one emotional labels, otherviwe NaN issues
        # if not torch.all(batch['emotion'] == 0).item():
        
        # remove all zeros utterances (and the associated labels)
        # non_zero_indexes = torch.nonzero(~torch.all(batch['input_ids']==0, dim=1)).squeeze()
        # batch['input_ids'] = batch['input_ids'][non_zero_indexes]
        # batch['emotion'] = batch['emotion'][non_zero_indexes]

        # print("="*20, "batch")
        # print(batch)
        output_probas = finetuning_model(batch['input_ids'])#.logits
        # print(output_probas.logits.shape)
        loss = ce_loss(output_probas, batch['sentiment'])
        loss.backward()
        optimizer.step()

        trues = batch['sentiment'].tolist()
        preds = torch.argmax(output_probas, dim=1).tolist()
        all_trues.extend(trues), all_preds.extend(preds)

        loss_it.append(loss.item())
        optimizer.zero_grad()
    # print(loss_it)

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

        output_probas = finetuning_model(batch['input_ids'])#.logits
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

    # model = AutoModelForSequenceClassification.from_pretrained(  
    #     model_name,
    #     low_cpu_mem_usage=True,         # recommanded param
    #     return_dict=True,               # not used for now
    #     torch_dtype=torch.bfloat16,     # bfloat instead of float because it may help
    #     device_map=args['device'],      # send to the right device
    # )

    model = TrainableHeadAdapters(args, nb_classes=2)
    model.to(args['device'])

    for p in model.parameters(): p.requires_grad = False
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    args.update({'tokenizer':tokenizer})
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id
    tokenizer.padding_side = "right"

    target_modules = select_target_modules(["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head"], args['target_modules'])
    config = LoraConfig(
            r=args['rank'],                             # rank of lora module
            lora_alpha=2*args['rank'],                  # resclaling weights parameters, therefore here alpha = 2*rank ("yelling at the model very loud"). Some suggest alpha = rank
            target_modules=target_modules,
            # layers_to_transform=[2, 6,10, 14, 18, 22, 26, 30],  # avoid top layers, this modifies the representation too much
            bias="lora_only",               # should be better than default setting in our case
            lora_dropout=0.05,              # conventional setting
            task_type=TaskType.SEQ_CLS
        )
    print(40*"-")
    print("LoRA config:")
    print(f"\t- rank = {config.r}")
    print(f"\t- alpha = {config.lora_alpha}")
    print(f"\t- target modules = {config.target_modules}")
    print(f"\t- layers to transform = {config.layers_to_transform}")
    print(f"\t- bias = {config.bias}")
    print(f"\t- dropout = {config.lora_dropout}")
    print(40*"-")

    args.update({'base_model': model}) # save the initial pretrained model without the adapters. This model will NOT be updated
    model = get_peft_model(model, config)
    args.update({'model':model}) # save the model with the adapters that will be updated in fine-tuning
    args.update({'max_new_tokens':20}) # set max new tokens (TODO uniformizer args keys)

    # run training and validation
    val_losses = run_epochs(args, model, train_loader, val_loader, experiment)

    return val_losses


if __name__ == "__main__":

    dd_llama_tokenized = load_from_disk("../tweet_airline_llama2_tokenized")
    args, train_loader, val_loader = get_args_and_dataloaders(dd_llama_tokenized, TweetAirlineLlamaTokenized)

    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--rank", help="Rank of LoRA layers", type=int, default=4)
    parser.add_argument("-m", "--target-modules", help="a string that points attention layers where to put LoRA adapters. The string concatenates the first letter of each desired module.", type=str, default='qv')
    arg = parser.parse_args()

    rank = arg.rank
    target_modules = arg.target_modules

    args = {'vocab_size':32000,                # new vocab size corresponding to the new dataset
            'batch_size':3,                     # size of the batch, the greater bsize the greater number of data samples
            'block_size':64,                    # Transformer block size in the language model
            'train_iters':100,                    # number of train batches to consider in one episode
            'eval_iters':10,                    # number of validation/test batches to consider in one episode
            'lr':1e-3,                          # learning rate
            'rank':rank,                        # rank in LoRA config
            'target_modules':target_modules,    # target modules in LoRA config
            'device':activate_gpu(),            # set device for training. Desable force_cpu to run on gpu if available
            'max_eps':40,                       # number of episodes (max of episodes in case of early stopping)
            'n_embd':4096,                        # embedding size
            'n_heads':8,                        # number of attention heads for one transformer block   
            'n_layers':33,                      # number of Transformer layers in the language model    
            'dropout':0.3,                      # dropout rate  
            'hf':False,                         # False if BabyLM, otherwise llama, falcon, mistral,..  . 
        }
    
    run_exp(args, MODEL_NAME, train_loader, val_loader, 'test_lora_usual_conditions')
