import torch
import torch.nn as nn

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model 
import os
from tqdm import tqdm
import argparse
from termcolor import colored
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# from other scripts
from utils import *
from logging_utils import *

# set default tensor type
torch.set_default_dtype(torch.float16)
torch.set_printoptions(precision=10)

# this helps on the following error: RuntimeError: CUDA error: invalid configuration argument.
# source: https://stackoverflow.com/questions/77343471/pytorch-cuda-error-invalid-configuration-argument 
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

# to display loss values for each iterations
all_train_losses, all_val_losses = [], []


def train(args, epoch, experiment):
    '''
        A sequence of fine-tuning iterations for classification-generation fine-tuning

        @param args (str):                 the hyperparameters for the training
        @param epoch (int):                the index of the current epoch
        @param experiment (str):           the experiment name

        @return loss_it (list):            the list of all the losses on each batch for the epoch
        @return trues (list):              list of gold labels to be stored later
        @return preds (list):              list of the associated predictions to be stored later
    '''
    classification_model, generation_model = args['clf_model'], args['gen_model']
    classification_model.train()
    optimizer = torch.optim.AdamW(classification_model.parameters(), lr=args['lr'], fused=torch.float16)
    ce_loss = nn.CrossEntropyLoss()
    loss_it = []
    trues, preds = [], []
    file_paths = []

    for batch_index in tqdm(range(args['train_iters']), desc="Epoch %s: " % (epoch+1), total=args['train_iters']):
        # generate sentences with a specific RL
        batch_labels, batch_generations, batch_ids = generate_from_random_prompts(args)
        file_path = save_batch_generations(batch_generations, batch_index, experiment)
        file_paths.append(file_path)

        gen_sentiments = get_sentiment_labels(file_path, args)
        # trues are the sentiments that the generated sentences have according to the annotator model
        trues.extend(gen_sentiments)

        # get classification model output
        model_input = torch.stack(batch_ids).squeeze()
        output_logits = classification_model(input_ids=model_input).logits
        batch_preds = torch.argmax(output_logits, dim=-1)
        preds.extend([bp.item() for bp in batch_preds])

        # for i, s in enumerate(batch_generations):
        #     print(f"Sample #{i} | Asked {batch_labels[i]} | Got {gen_sentiments[i]} | Predicted {batch_preds[i]}")

        # training step (loss computation w/ autocast to handle tensor type consistency)
        with torch.autocast('cuda'):
            loss = ce_loss(output_logits, torch.tensor(gen_sentiments, device=args['device'])) 

        loss.backward()
        optimizer.step()
        loss_it.append(loss.item())
        optimizer.zero_grad()

        # at this point, the weights of the adapters in clf_models have been updated. The generation model should contain new weights
        update_adapter_weights(args, generation_model, classification_model)

        # update the models where they are stored, otherwise it is always the initial model that is used for generation
        args['clf_model'], args['gen_model'] = classification_model, generation_model

    # append batch generations to split generations
    store_split_generations('train', file_paths, trues, experiment)
    all_train_losses.extend(loss_it) # save all the losses of this epoch
    loss_it_avg = sum(loss_it)/len(loss_it)
    acc = accuracy_score(trues, preds)

    # # print useful information about the training progress and scores on this training set's full pass
    print("Epoch %s/%s - %s : (%s %s) (%s %s)" % (colored(str(epoch+1), 'blue'),args['max_eps'] , colored('Training', 'blue'), colored('Average loss: ', 'cyan'), loss_it_avg, colored('Accuracy: ', 'cyan'), acc))

    return loss_it, trues, preds


def test(args, target, experiment):
    '''
        Perfom one epoch of model evaluation, either as validation or test.

        @param args (str):                 the hyperparameters for the training
        @param target (str):               either 'validation' or 'test', for a better display
        @param experiment (str):           the experiment name

        @return loss_it_avg (list):        the list of all the losses on each batch for the epoch
        @return trues (list):              list of gold labels to be stored later
        @return preds (list):              list of the associated predictions to be stored later
    '''
    classification_model = args['clf_model']
    ce_loss = nn.CrossEntropyLoss()
    loss_it = []
    trues, preds = [], []
    file_paths = []

    with torch.no_grad():

        for batch_index in tqdm(range(args['eval_iters']), total=args['eval_iters']):
            # generate sentences with a specific RL
            batch_labels, batch_generations, batch_ids = generate_from_random_prompts(args)
            file_path = save_batch_generations(batch_generations, batch_index, experiment)
            file_paths.append(file_path)

            gen_sentiments = get_sentiment_labels(file_path, args)
            # trues are the sentiments that the generated sentences have according to the annotator model
            trues.extend(gen_sentiments)

            # get classification model output
            model_input = torch.stack(batch_ids).squeeze()
            output_logits = classification_model(input_ids=model_input).logits
            batch_preds = torch.argmax(output_logits, dim=-1)
            preds.extend([bp.item() for bp in batch_preds])

            # training step (loss computation w/ autocast to handle tensor type consistency)
            with torch.autocast('cuda'):
                loss = ce_loss(output_logits, torch.tensor(gen_sentiments, device=args['device']))
            loss_it.append(loss.item())

    # all_val_losses.extend(loss_it) # save all the losses of this epoch
    loss_it_avg = sum(loss_it)/len(loss_it)

    # append batch generations to split generations
    store_split_generations(target, file_paths, trues, experiment)

    # trues, preds = trues.cpu(), preds.cpu()
    accuracy = accuracy_score(trues, preds)
    precision = precision_score(trues, preds, average='weighted', zero_division=0.0)
    recall = recall_score(trues, preds, average='weighted')
    f1 = f1_score(trues, preds, average='weighted')
    all_val_losses.extend(loss_it)

    # print useful information about the training progress and scores on this training set's full pass
    print("%s : (%s %s) (%s %s) (%s %s) (%s %s) (%s %s)" % (colored(f'{target}', 'blue'), colored('Average loss: ', 'cyan'), loss_it_avg, colored('Accuracy: ', 'cyan'), accuracy, colored('Precision: ', 'cyan'), precision, colored('Recall: ', 'cyan'), recall, colored('F1 score: ', 'cyan'), f1))

    return loss_it, trues, preds


def run_episodes(args, experiment):
    '''
        Run all episodes of the fine-tuning (train + validation).

        @param args (dict):                 the hyperparameters for the training
        @param experiment (str):            the name of the experiment

        @return val_losses (list):          the losses on validation sets (length = number of val_iters)
    '''
    val_losses = []

    for ep in range(args['max_eps']):
        # perform training and validation runs
        _, train_trues, train_preds = train(args, ep, experiment)
        print(all_train_losses)
        val_loss, val_trues, val_preds = test(args, 'validation', experiment)
        print(all_val_losses)
 
        # save epoch trues and preds for train and validation
        save_epoch_data('train', train_trues, train_preds, ep, experiment)
        save_epoch_data('validation', val_trues, val_preds, ep, experiment)

        # save val loss for this epoch
        val_losses.append(val_loss)

    return val_losses


def run_on_several_test_sets(args, experiment, episodes=10):
    '''
        This function accounts for model stability by testing the model on different test sets depending on the number of episodes. Predictions are stored for each episode so all the classification metrics can be computed as well as their mean and standard deviation.

        @param args (dict):                 the hyperparameters for the training
        @param experiment (str):            the name of the experiment
        @param episodes (int):              the number of test sets to infer on (default=10)

        @return test_losses (list):         the losses on test sets (length = number of episodes)
    '''
    test_losses = []

    for i in range(episodes):
        test_loss, test_trues, test_preds = test(args, 'test', experiment)
        save_epoch_data('test', test_trues, test_preds, i, experiment)

        test_losses.append(test_loss)

    return test_losses


def run_exp(args, model_name, annotator_model_name, experiment, episodes=10):
    '''
        Run an end-to-end finetuning.

        @param args (dict):                 the dict containing all the hyperparameters
        @param model_name (str):            either from a local storage (hf=False), or from huggingface hub (hf=True)
        @param annotator_model_name (str):  the sentiment classifier model to use to annotate generated sentences
        @param experiment (str):            name of the experiment 
        @param episodes (int):              number of times the test step should be performed (to compute descriptive stats on metrics)

        @return val_losses (list):          all val losses for all the 'epochs'
        @return test_losses (list):         all test losses from all the episodes
    '''
    print(colored(f'Start of the experiment {experiment}', 'green'))

    # create results dir if it doesn't exist
    if not os.path.exists(f'../results/{experiment}/'):
        os.makedirs(f'../results/{experiment}/')

    # get our SOTA model for sentiment annotation + associated tokenizer
    annotator_model = AutoModelForSequenceClassification.from_pretrained(  
        annotator_model_name,
        low_cpu_mem_usage=True,         # recommanded param
        return_dict=True,               # not used for now
        torch_dtype=torch.bfloat16,     # bfloat instead of float because it may help
        device_map=args['device'],      # send to the right device
    )
    annotator_model.load_state_dict(torch.load(f'../models/{annotator_model_name}_best.pt'))
    annotator_tokenizer = AutoTokenizer.from_pretrained(annotator_model_name, trust_remote_code=True)

    # save both annotator model and tokenizer
    args.update({'ann_model':annotator_model})
    args.update({'ann_tokenizer':annotator_tokenizer})

    # instantiate 2 Llama models: one for generation and one for classification
    generation_model = AutoModelForCausalLM.from_pretrained(  
        model_name,
        low_cpu_mem_usage=True,         # recommanded param
        return_dict=True,               # not used for now
        torch_dtype=torch.bfloat16,     # bfloat instead of float because it may help
        device_map=args['device'],      # send to the right device
    )
    generation_model.post_init()
    classification_model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        low_cpu_mem_usage=True,         # recommanded param
        return_dict=True,               # not used for now
        torch_dtype=torch.bfloat16,     # bfloat instead of float because it may help
        device_map=args['device'],      # send to the right device
    )
    classification_model.post_init()

    # freeze both models
    for p in generation_model.parameters(): p.requires_grad = False
    for p in classification_model.parameters(): p.requires_grad = False

    # setup + save tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    generation_model.config.pad_token_id = generation_model.config.eos_token_id
    classification_model.config.pad_token_id = classification_model.config.eos_token_id
    tokenizer.padding_side = "right"
    args.update({'max_new_tokens':20})
    args.update({'tokenizer':tokenizer})

    # setup LoRA config for the adapters of both models (they NEED to be the same!)
    if args['hf_model'] in ['llama', 'zephyr', 'mistral', 'gemma']:
        target_modules = select_target_modules(["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head"], args['target_modules']) # for decoder-only model
    if args['hf_model'] in ['flan']:
        target_modules = select_target_modules(["q", "k", "v", "o", "lm_head"], args['target_modules']) # for encoder-decoder model

    config = LoraConfig(
        r=args['rank'],                       # rank of lora module
        lora_alpha=(1/4)*args['rank'],        # resclaling factor for weights parameters
        target_modules=target_modules,        # which attention modules to put adapters in
        bias="lora_only",                     # should be better than default setting in our case
        lora_dropout=0.1,                     # conventional setting
        # task_type=TaskType.SEQ_CLS,         # I don't think this is useful
        # use_rslora=True                     # things get worse when I enable this
    )

    # display config details
    args.update({'config': config})
    display_lora_config(config)

    # plug adapters + save both models = here I remove adapters because I want to use raw llama model
    generation_model = get_peft_model(generation_model, config)
    classification_model = get_peft_model(classification_model, config)
    args.update({'gen_model': generation_model})
    args.update({'clf_model': classification_model}) 

    # train the classification layer in the classifier
    for p in classification_model.base_model.model.score.parameters(): p.requires_grad = True

    # send models to device
    generation_model.to(args['device'])
    classification_model.to(args['device'])

    # run training and validation
    val_losses = run_episodes(args, experiment)

    # run test 
    test_losses = run_on_several_test_sets(args, experiment, episodes)

    return val_losses, test_losses


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--rank", help="Rank of LoRA layers", type=int, default=4)
    parser.add_argument("-m", "--target-modules", help="a string that points attention layers where to put LoRA adapters. The string concatenates the first letter of each desired module.", type=str, default='qv')
    parser.add_argument("-l", "--layers-list", help="pick a list of layers between 1, 2 and 3", type=int, default=1)
    arg = parser.parse_args()

    rank = arg.rank
    target_modules = arg.target_modules
    layers_list = arg.layers_list

    args = {
            'batch_size':3,                     # size of the batch, the greater bsize the greater number of data samples
            'train_iters':100,                  # number of train batches to consider in one episode
            'eval_iters':10,                    # number of validation/test batches to consider in one episode
            'lr':1e-4,                          # learning rate
            'rank':rank,                        # rank in LoRA config
            'target_modules':target_modules,    # target modules in LoRA config
            'device':activate_gpu(),            # set device for training. Desable force_cpu to run on gpu if available
            'max_eps':10,                       # number of episodes (max of episodes in case of early stopping)
            'max_length':20,                    # the maximum length of generated sentences for sentiment analysis annotation
            'n_layers':24,                      # number of Transformer layers in the language model    
        }
    
    # ANNOTATOR MODEL
    annotator_model_name = 'google-bert/bert-base-uncased'
    
    # HUGGINGFACE MODELS
    model_name = "meta-llama/Llama-2-7b-chat-hf"
    # model_name = "google/flan-t5-xl"
    # model_name = "meta-llama/Llama-2-13b-chat-hf"
    # model_name = "HuggingFaceH4/zephyr-7b-beta"
    # model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    # model_name = "google/gemma-2b-it"

    # ARGS SETUPS FOR THE DIFFERENT MODELS
    # args.update({'hf_model':hf_model_name(model_name), 'n_layers':12}) # for flan
    args.update({'hf_model':hf_model_name(model_name), 'n_layers':32}) # for llama
    # args.update({'hf':'adapters', 'vocab_size':256000, 'n_embd':2048}) # for gemma

    display_finetuning_args(args)

    run_exp(args, model_name, annotator_model_name, f"{args['hf_model']}_{args['rank']}_{args['target_modules']}")
