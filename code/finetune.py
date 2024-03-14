import torch
import torch.nn as nn

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, peft_model
import os
from tqdm import tqdm
import argparse
from termcolor import colored
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.tensorboard import SummaryWriter

# from other scripts
from utils import *
from logging_utils import *
from models import BabyLanguageModel, TrainableHead, TrainableHeadAdapters, BinaryTrainableModel

# disable hf tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# set default tensor type
torch.set_default_dtype(torch.float16)
torch.set_printoptions(precision=10)


def train(args, finetuning_model, epoch, experiment, hf=False):
    '''
        A sequence of fine-tuning iterations for ontology-validation based fine-tuning. The aim is to understand and accurately generate different readability levels (RL) 

        @param args (str):                 the hyperparameters for the training
        @param finetuning_model:           the model used for weight updates in fine-tuning
        @param epoch (int):                the index of the current epoch
        @param experiment (str):           the experiment name
        @param hf:                         False in case of local model, a huggingface model alias otherwise. Currently only 'llama' is supported

        @return loss_it_avg (list):        the list of all the losses on each batch for the epoch
        @return trues (list):              list of gold labels to be stored later
        @return preds (list):              list of the associated predictions to be stored later
    '''
    # setup training config + outputs storage
    finetuning_model.train()
    for p in finetuning_model.pool.parameters(): p.requires_grad = False
    optimizer = torch.optim.AdamW(finetuning_model.parameters(), lr=args['lr'], fused=torch.float16)
    weights = torch.tensor([1.0, 2.0], device=args['device']) # double proba of getting 1 (same class) as there are 3 different classes, therefore twice more chances to be right when saying they are not same
    ce_loss = nn.CrossEntropyLoss(weight=weights)
    writer = args['writer']
    loss_it = []
    trues, preds = [], []
    file_paths = []

    for batch_index in tqdm(range(args['train_iters']), desc="Epoch %s: " % (epoch+1), total=args['train_iters']):
        # generate sentences with a specific RL
        batch_labels, batch_generations, batch_ids = generate_from_random_prompts(args, hf=hf)
        file_path = save_batch_generations(batch_generations, batch_index)
        file_paths.append(file_path)

        # trues are the RL that the generated sentence should have
        trues.extend(batch_labels)
        create_batch_individual(batch_index, file_path)
        generations_rl = get_readability_levels(f'../rdf/individual_batch_{batch_index}.rdf')
        preds.extend(generations_rl)

        # prediction probabilities go through finetuning model
        generations_probas = [[int(j == i) for j in range(3)] for i in generations_rl]
        generations_probas = torch.tensor(generations_probas, dtype=torch.float16, requires_grad=True).to(args['device'])
        gold_label = is_same(batch_labels, generations_rl)
        output = finetuning_model(input_ids=torch.stack(batch_ids).squeeze())
        # print(output) # this to check if the output is as desired

        # training step
        loss = ce_loss(output, torch.tensor(gold_label).to(args['device'])) 
        loss.backward()
        optimizer.step()
        loss_it.append(loss.item())
        optimizer.zero_grad()
        print(loss_it)

    # append batch generations to split generations
    store_split_generations('train', file_paths, trues, experiment)
    loss_it_avg = sum(loss_it)/len(loss_it)

    # print useful information about the training progress and scores on this training set's full pass
    print("Epoch %s/%s - %s : (%s %s)" % (colored(str(epoch+1), 'blue'),args['max_eps'] , colored('Training', 'blue'), colored('Average loss: ', 'cyan'), loss_it_avg))

	# 🛑 add some metrics to keep with a label and the epoch index
    writer.add_scalar("Loss/train", loss_it_avg, epoch)

    return loss_it_avg, trues, preds


def test(args, finetuning_model, target, experiment, hf=False):
    '''
        Perfom one epoch of model evaluation, either as validation or test.

        @param args (str):                 the hyperparameters for the training
        @param finetuning_model:           the model used for weight updates in fine-tuning
        @param target (str):               either 'validation' or 'test', for a better display
        @param experiment (str):           the experiment name
        @param hf:                          False in case of local model, a huggingface model alias otherwise. Currently only 'llama' is supported

        @return loss_it_avg (list):        the list of all the losses on each batch for the epoch
        @return trues (list):              list of gold labels to be stored later
        @return preds (list):              list of the associated predictions to be stored later
    '''
    finetuning_model.eval()
    writer = args['writer']
    loss_it = []
    weights = torch.tensor([1.0, 2.0], device=args['device']) # double proba of getting 1 (same class) as there are 3 different classes, therefore twice more chances to be right when saying they are not same
    ce_loss = nn.CrossEntropyLoss(weight=weights)
    trues, preds = [], []
    file_paths = []

    for batch_index in tqdm(range(args['eval_iters']), total=args['eval_iters']):
        # generate sentences with a specific RL
        batch_labels, batch_generations, batch_ids = generate_from_random_prompts(args, hf=hf)
        file_path = save_batch_generations(batch_generations, batch_index)
        file_paths.append(file_path)

        # trues are the RL that the generated sentence should have
        trues.extend(batch_labels)
        create_batch_individual(batch_index, file_path)
        generations_rl = get_readability_levels(f'../rdf/individual_batch_{batch_index}.rdf')
        preds.extend(generations_rl)

        # prediction probabilities go through finetuning model
        generations_probas = [[int(j == i) for j in range(3)] for i in generations_rl]
        generations_probas = torch.tensor(generations_probas, dtype=torch.float16, requires_grad=True).to(args['device'])
        gold_label = is_same(batch_labels, generations_rl)
        output = finetuning_model(input_ids=torch.stack(batch_ids).squeeze())
        # print(output) # this to check if the output is as desired

        # training step
        loss = ce_loss(output, torch.tensor(gold_label).to(args['device'])) 
        loss_it.append(loss.item())
        print(loss_it)

    loss_it_avg = sum(loss_it)/len(loss_it)

    # append batch generations to split generations
    store_split_generations(target, file_paths, trues, experiment)

    accuracy = accuracy_score(trues, preds)
    precision = precision_score(trues, preds, average='weighted', zero_division=0.0)
    recall = recall_score(trues, preds, average='weighted')
    f1 = f1_score(trues, preds, average='weighted')

    # print useful information about the training progress and scores on this training set's full pass
    print("%s : (%s %s) (%s %s) (%s %s) (%s %s) (%s %s)" % (colored(f'{target}', 'blue'), colored('Average loss: ', 'cyan'), loss_it_avg, colored('Accuracy: ', 'cyan'), accuracy, colored('Precision: ', 'cyan'), precision, colored('Recall: ', 'cyan'), recall, colored('F1 score: ', 'cyan'), f1))

	# 🛑 add some metrics to keep with a label and the epoch index
    writer.add_scalar(f"Loss/{target}", loss_it_avg)

    return loss_it_avg, trues, preds


def run_episodes(args, finetuning_model, experiment, hf=False):
    '''
        Run all episodes of the fine-tuning (train + validation).

        @param args (dict):                 the hyperparameters for the training
        @param model:                       the pretrained model to use for inference
        @param finetuning_model:            the model used for weight updates in fine-tuning
        @param stoi (dict):                 the string-to-index dict from the pretraining vocab
        @param itos (list):                 the index-to-string list from the pretraining vocab
        @param experiment (str):            the name of the experiment
        @param hf:                          False in case of local model, a huggingface model alias otherwise. Currently only 'llama' is supported

        @return val_losses (list):          the losses on validation sets (length = number of val_iters)
    '''
    val_losses = []

    for ep in range(args['max_eps']):
        # perform training and validation runs
        _, train_trues, train_preds = train(args, finetuning_model, ep, experiment, hf=hf)
        val_loss, val_trues, val_preds = test(args, finetuning_model, 'validation', experiment, hf=hf)
 
        # save epoch trues and preds for train and validation
        save_epoch_data('train', train_trues, train_preds, ep, experiment)
        save_epoch_data('validation', val_trues, val_preds, ep, experiment)

        # save val loss for this epoch
        val_losses.append(val_loss)

    return val_losses


def run_on_several_test_sets(args, finetuning_model, experiment, episodes=5, hf=False):
    '''
        This function accounts for model stability by testing the model on different test sets depending on the number of episodes. Predictions are stored for each episode so all the classification metrics can be computed as well as their mean and standard deviation.

        @param args (dict):                 the hyperparameters for the training
        @param model:                       the pretrained model to use for inference
        @param finetuning_model:            the model used for weight updates in fine-tuning
        @param stoi (dict):                 the string-to-index dict from the pretraining vocab
        @param itos (list):                 the index-to-string list from the pretraining vocab
        @param experiment (str):            the name of the experiment
        @param episodes (int):              the number of test sets to infer on (default=10)
        @param hf:                          False in case of local model, a huggingface model alias otherwise. Currently only 'llama' is supported

        @return test_losses (list):         the losses on test sets (length = number of episodes)
    '''
    test_losses = []

    for i in range(episodes):
        test_loss, test_trues, test_preds = test(args, finetuning_model, 'test', experiment, hf=hf)
        save_epoch_data('test', test_trues, test_preds, i, experiment)

        test_losses.append(test_loss)

    return test_losses


def run_exp(args, model_name, experiment, episodes=10, hf=False):
    '''
        Run an end-to-end finetuning.

        @param args (dict):           the dict containing all the hyperparameters
        @param model_name (str):      either from a local storage (hf=False), or from huggingface hub (hf=True)
        @param experiment (str):      name of the experiment 
        @param episodes (int):        number of times the test step should be performed (to compute descriptive stats on metrics)
        @param hf:                    False in case of local model, a huggingface model alias otherwise. Currently only 'llama' is supported

        @return val_losses (list):    all val losses for all the 'epochs'
        @return test_losses (list):   all test losses from all the episodes
    '''
    print(colored(f'Start of the experiment {experiment}', 'green'))

    # create results dir if it doesn't exist
    if not os.path.exists(f'../results/{experiment}/'):
        os.makedirs(f'../results/{experiment}/')

    if hf:
        # TODO encapsuler tout ça dans des fonctions pour plus de clarté et pour pouvoir paramétrer tout ça !
        model = AutoModelForCausalLM.from_pretrained(  
            model_name,
            low_cpu_mem_usage=True,         # recommanded param
            return_dict=True,               # not used for now
            torch_dtype=torch.bfloat16,     # bfloat instead of float because it may help
            device_map=args['device'],      # send to the right device
        )
        for p in model.parameters(): p.requires_grad = False
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        args.update({'tokenizer':tokenizer})
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
        tokenizer.padding_side = "right"

        config = LoraConfig(
                r=32,
                lora_alpha=64,
                target_modules=[
                    # "q_proj",
                    "k_proj",
                    "v_proj",
                    # "o_proj",
                    # "gate_proj",
                    # "up_proj",
                    # "down_proj",
                    # "lm_head",
                ],
                layers_to_transform=[3, 4, 5, 29],  # avoid top layers, this modifies the representation too much
                bias="lora_only",               # should be better than default setting in our case
                lora_dropout=0.05,              # conventional setting
                # task_type=TaskType.SEQ_CLS,
            )
        args.update({'base_model': model}) # save the initial pretrained model without the adapters. This model will NOT be updated
        model = get_peft_model(model, config)
        args.update({'model':model}) # save the model with the adapters that will be updated in fine-tuning
        args.update({'max_new_tokens':20}) # set max new tokens (TODO uniformizer args keys)
        finetuning_model = TrainableHeadAdapters(args)
        finetuning_model.to(args['device'])

    else:
        
        finetuning_model = setup_model_babylm(args, model_name)

    # run training and validation
    val_losses = run_episodes(args, finetuning_model, experiment, hf=hf)

    # run test 
    test_losses = run_on_several_test_sets(args, finetuning_model, experiment, episodes, hf=hf)

    # log all classification metrics from saved trues/preds
    ## TBC

    return val_losses, test_losses


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--dblock", help="The index of the Transfomer docoder block to update.", type=int, default=0)
    arg = parser.parse_args()
    d_block = arg.dblock
    # print(f"Decoder block #{d_block} will be updated")

    args = {'vocab_size':239267,        # new vocab size corresponding to the new dataset
            'batch_size':32,            # size of the batch, the greater bsize the greater number of data samples
            'block_size':64,            # Transformer block size in the language model
            'train_iters':100,          # number of train batches to consider in one episode
            'eval_iters':10,            # number of validation/test batches to consider in one episode
            'lr':1e-4,                 # learning rate
            'device':activate_gpu(),    # set device for training. Desable force_cpu to run on gpu if available
            'max_eps':10,               # number of episodes (max of episodes in case of early stopping)
            'n_embd':64,                # embedding size
            'n_heads':8,                # number of attention heads for one transformer block
            'n_layers':24,              # number of Transformer layers in the language model
            'dropout':0.3,              # dropout rate
            'writer':SummaryWriter(f"../logs/{get_datetime()}"), # Tensorboard util
            'hf':False,                 # False if BabyLM, otherwise llama, falcon, mistral,... 
            'd_block':d_block
        }

    # model_path = '../models/babyllm-gptlike_64_22012024223644_nq_params.pt'
    model_name = "meta-llama/Llama-2-7b-chat-hf"
    # model_name = "google/gemma-2b-it"
    # update args to run finetuning trainable head with appropriate dimensions
    args.update({'hf':'adapters', 'vocab_size':32000, 'n_embd':4096, 'n_layers':33}) # for llama
    # args.update({'hf':'adapters', 'vocab_size':256000, 'n_embd':2048}) # for gemma

    run_exp(args, model_name, '1403_llama2_finetuning_binary', hf=True)



# ---

# [x] OPTION 1: check the readability class of the output. To do so, write an auxiliary function that:
    # - generates a sentence with a readability level instruction given in prompt
    # - add this infividual to a temp rdf file for the batch
    # - perform inference on this file (like it is done in create_individuals.py)
    # - uses the mapping class -> class index to finally output the individual class

# [x] OPTION 2: change the prompt to finish on last utterance by (ReadabilityLevel= which encourages the model to learn the concept of readability (in a final test step we can use OPTION 1 to check of the model actually learnt something). We need a function that:
    # - decodes the output
    # - parses it to deduce the predicted readability level
    # - maps it to the class index, and that's all :)
