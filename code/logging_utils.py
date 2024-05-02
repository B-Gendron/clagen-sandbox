import os
import torch
from datasets import DatasetDict
import json
import csv

import warnings
warnings.filterwarnings('ignore')

# set default tensor type
torch.set_default_dtype(torch.float16)


# -----------------------------------------------------------------------------------------
# Data management
# -----------------------------------------------------------------------------------------

def load_dataset(dataset_name):
    '''
        Load a dataset save in the huggingface format, located in the data folder.

        @param dataset_name (str): the name of the dataset, i.e. the associated folder inside the data folder. This string should correspond to an existing dataset in the folder, otherwise an exception will be raised.

        @return dataset (DatasetDict): the dataset as a DatasetDict object.
    '''
    path = f"./data/{dataset_name}"
    if os.path.isdir(path):
        dataset = DatasetDict.load_from_disk(f"../data/{dataset_name}")
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
        dataset.save_to_disk(f'../data/{dataset_name}')
    elif output_format == "json":
        with open(f"data/{dataset_name}.json", 'w') as f:
            json.dump(dataset, f)
    else:
        print("The given output format is not recognized. Please note that accepted formats are 'huggingface' and 'json")


# -----------------------------------------------------------------------------------------
# Save info from fine-tuning
# -----------------------------------------------------------------------------------------

def save_batch_generations(batch_generations, batch_index, experiment):
    file_path = f'../objects/batch_generations_{batch_index}_{experiment}.tsv'
    with open(file_path, 'w', newline='') as f:
        tsv_writer = csv.writer(f, delimiter='\t')

        for row in batch_generations:
            row = row.replace('\n', '\\n')
            tsv_writer.writerow([row])

    return file_path[3:] # remove ../ to make it relative

def store_split_generations(split, file_paths, trues, experiment):
    '''
        This function concatenate generated sentences from all batches of the split 'split' into one tsv file. 

        @param split (str):             the name of the split, either 'train', 'validation' or 'test'. Note that these are not proper splits as there is no dataset in this fine-tuning procedure
        @param file_paths (list):       the list of paths to the generated sentences in each batch
        @param trues (list):            all the gold readability level labels
        @param experiment (str):        the name of the experiment (=folder where all logs are saved)
    '''
    # readability_levels_mapping = {0:'EasilyReadableText', 1:'StandardReadableText', 2:'HardlyReadableText'}
    sentence_length_mapping = {0:'Negative', 1:'Positive'}
    with open(f'../results/{experiment}/generations_{split}.tsv', 'a') as all_gens:
        tsv_writer = csv.writer(all_gens, delimiter='\t')
        # iterate through batch files
        for i, path in enumerate(file_paths):
            # store gold labels and generations for this batch
            abs_path = f'../{path}'
            with open(abs_path, 'r') as batch_gens:
                for j, row in enumerate(batch_gens):
                    tsv_writer.writerow([sentence_length_mapping[trues[i+j]], row])

            # remove batch-wise generations file path
            os.remove(abs_path)


def save_epoch_data(target, trues, preds, epoch_or_iter, experiment):
    '''
        This function saves the trues and preds at each epoch for train and validation and for each run on test set (several runs are performed to ensure stability)

        @target (str):              either 'train', 'validation' or 'test'
        @trues (tensor):            the gold labels for sentence readability levels
        @preds (tensor):            the readability levels of the generated sentences
        @epoch_or_iter (int):       the # of the epoch (train and val sets) or the iter (test set)
    '''
    with open(f'../results/{experiment}/predictions_{target}.csv', 'a+', newline='') as f:
        write = csv.writer(f)

        # Write data in columns
        for i in range(len(trues)):
            write.writerow([ trues[i], preds[i], epoch_or_iter ])


def display_lora_config(config):
    print(40*"-")
    print("LoRA config:")
    print(f"\t- rank = {config.r}")
    print(f"\t- alpha = {config.lora_alpha}")
    print(f"\t- target modules = {config.target_modules}")
    print(f"\t- layers to transform = {config.layers_to_transform}")
    print(f"\t- bias = {config.bias}")
    print(f"\t- dropout = {config.lora_dropout}")
    print(40*"-")


def display_finetuning_args(args):
    print(40*"-")
    print("Fine-tuning config:")
    for k, v in args.items():
        if k not in ['rank', 'target_modules']:
            print(f"\t- {k} = {v}")
    print(40*"-")
