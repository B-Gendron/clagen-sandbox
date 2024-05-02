from termcolor import colored
from datetime import datetime
import torch
import numpy as np
from numpy import random as rd

import warnings
warnings.filterwarnings('ignore')

# set default tensor type
torch.set_default_dtype(torch.float16)

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

    
def hf_model_name(s):
    '''
        This function takes the name of a huggingface model and isolate the first part of the name of the model, after the first slash (/) and before the first dash (-).
        The result is given in lower case.

        This function is useful to generate a parameter name that can be evaluated during fine-tuning steps when processing differs depending on the model. For instance, when doing weight transfer, the layers are not named the same in different models.

        @s (str):       the string of the huggingface model name
    '''
    return s.split('/')[1].split('-')[0].lower()


# -----------------------------------------------------------------------------------------
# Fine-tuning utils
# -----------------------------------------------------------------------------------------

def select_target_modules(target_modules, selection):
    # Create a dictionary mapping first letters to full module names
    module_dict = {module[0]: module for module in target_modules}
    
    # Filter modules based on the first letters in the param
    subset = [module_dict[letter] for letter in selection if letter in module_dict]
    
    return subset


def random_prompt(concept, classes, hf=False):
    '''
        This auxiliary function allows to get a prompt that ask for a sentence belonging to a certain class among given classes. It is for now used for readability levels but is meant for a more general purpose.

        @param concept (str):               the name of the ontology concept we want to learn, that is divided into the following classes
        @param classes (list):              a list or strings giving the names of all the classes 

        @return prompt (str):               the randomly selected prompt to use for generation
        @return class_index (int):          the index of the corresponding class that is asked in the prompt 
        @param hf:                          False in case of local model, a huggingface model alias otherwise. Currently only 'llama' is supported
    '''
    start_of_sentence = ['A', 'The', 'For', 'Yes', 'No', 'I']
    p = rd.uniform()
    n = len(classes)
    for k in range(1, n+1):
        if (k-1)/n < p < k/n:
            if hf:
                prompt = f"You are a client who recently puchased a new product and you want to write a review of this product to express your feelings. Here is the review with concept {concept} being {classes[k-1]}: {rd.choice(np.array(start_of_sentence))}"
                # prompt = f"INSTRUCTION: Give an example sentence for which concept {concept} is {classes[k-1]}. ANSWER: Here is an example sentence for the given concept: '{rd.choice(np.array(start_of_sentence))}"
                return prompt, k-1
            else: 
                prompt = f"A sentence whose {concept} is {classes[k-1]}: The"
                return prompt, k-1


def generate_from_random_prompts(args, hf=False):
    concept = 'Sentiment'
    classes = [f'{concept}{i}' for i in range(2)]
    batch_labels, batch_generations, batch_ids = [], [], []

    tokenizer = args['tokenizer']
    model = args['gen_model']

    for i in range(args['batch_size']):
        # get a randomly selected prompt (uniform law)
        prompt, label = random_prompt(concept, classes, hf=hf)
        batch_labels.append(label)

        # perform generation
        prompt = tokenizer(prompt, return_tensors="pt").to(args['device'])
        output = model.generate(**prompt, max_new_tokens=args['max_new_tokens'], repetition_penalty=1.5)[0] # contains prompt + generated part
        result = tokenizer.decode(output, skip_special_tokens=True)

        # store result
        generation = result[result.find(':')+1:result.find('\n')]
        output_ids = get_and_pad_ids(tokenizer(generation, return_tensors="pt").to(args['device'])['input_ids'], args, padding_length=40)
        batch_ids.append(output_ids)
        batch_generations.append(generation)
        print(f"Sample {i}: \t | Asked {label} | \t {generation}")

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


def get_sentiment_labels(file_path, args):
    '''
        This function takes a file with the sentences generated by the LLM for the current batch and uses a SOTA model in sentiment analysis to annotate the sentences with their actual sentiment (0=negative, 1=positive).
    '''
    batch_sentiments = []
    ml = args['max_length']
    # get annotator model and tokenizer
    annotator = args['ann_model']
    tok = args['ann_tokenizer']

    # read the file containing the generations
    with open(f'../{file_path}', 'r') as f:
        # for each utterance in the file
        for i, utterance in enumerate(f):
            utterance = utterance.strip()
            # process sentence (tokenization + padding)
            encoded_utterance = tok(utterance)['input_ids']
            n = len(encoded_utterance)
            if n < ml:
                encoded_utterance.extend([0 for _ in range(ml-n)])
            elif n > ml:
                encoded_utterance = encoded_utterance[:ml]
            # apply model and store label
            tensor_utterance = torch.tensor(encoded_utterance, device=args['device']).unsqueeze(0)
            sentiment_probas = annotator(tensor_utterance).logits
            sentiment = torch.argmax(sentiment_probas, dim=-1)
            batch_sentiments.append(sentiment.item())

    return batch_sentiments


def update_adapter_weights(args, g, c):
    '''
        À priori pas en cause étant donné que la classification ne fonctionne déjà pas (avant le transfert de poids, donc)
    '''
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
