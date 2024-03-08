# IN FINETUNING 
# This setting is now useless because we finetune llama only with adapters

    # if hf == 'llama':
    #     # setup model
    #     model = AutoModelForCausalLM.from_pretrained(  
    #         model_name,
    #         low_cpu_mem_usage=True,
    #         return_dict=True,       # this returns a load_state_dict compatible object ?
    #         torch_dtype=torch.float16,
    #         device_map=args['device'],
    #     )
    #     args.update({'config':model.config})
    #     # setup tokenizer
    #     tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    #     # tokenizer.pad_token = tokenizer.eos_token
    #     tokenizer.pad_token = tokenizer.eos_token
    #     tokenizer.padding_side = "right"

    #     # store the pipe to use it in generation
    #     pipe = pipeline(task="text-generation", 
    #                     model=model, 
    #                     tokenizer=tokenizer, 
    #                     max_new_tokens=20) # increase max_new_tokens to generate HardlyReadableText
    #     args.update({'pipe':pipe})

    #     # freeze all model layers 
    #     for p in model.parameters(): p.requires_grad = False

    #     # Setup finetuning model
    #     finetuning_model = TrainableHead(args)
    #     # initiate TrainableHead weights with the correct decoder layer weights
    #     transfer_weights(model.model.layers[args['d_block']], finetuning_model.decoder)

    #     # require grad at right positions
    #     for p in finetuning_model.parameters(): p.requires_grad = False
    #     for p in finetuning_model.decoder.parameters(): p.requires_grad = True

    #     for n, p in finetuning_model.named_parameters(): print(n, p.requires_grad)

    #     finetuning_model.to(args['device'])


# When the input ids were random
# input_ids0 = torch.randint(0, 32000, (32,20)).to(args['device'])

# We are no longer using a pipe for generation

        # store the pipe to use it in generation
        # pipe = pipeline(task="text-generation", 
        #                 model=model, 
        #                 tokenizer=tokenizer, 
        #                 max_new_tokens=20)
        # args.update({'pipe':pipe})


# This utils function is for tuning pre-trained decoders directly (without adapters, and only for llama models)
# def transfer_weights(source, target):
#     target.self_attn.q_proj.weight = source.self_attn.q_proj.weight
#     target.self_attn.k_proj.weight = source.self_attn.k_proj.weight
#     target.self_attn.v_proj.weight = source.self_attn.v_proj.weight
#     target.self_attn.o_proj.weight = source.self_attn.o_proj.weight
#     target.mlp.gate_proj.weight = source.mlp.gate_proj.weight
#     target.mlp.up_proj.weight = source.mlp.up_proj.weight
#     target.mlp.down_proj.weight = source.mlp.down_proj.weight
#     target.input_layernorm.weight = source.input_layernorm.weight
#     target.post_attention_layernorm.weight = source.post_attention_layernorm.weight 


# Old function from utils. This is not usefull as it doesn't proves that the model learns the concepts. It actually demonstrates that vocab extension works...

# def extend_vocab_with_readability_levels(itos, stoi):
#     '''
#         This function adds the useful ontology concepts in the vocabulary used for generation. 
#         /!\ This function overwrites the json files contraining itos and stoi mappings 

#         @param itos
#         @param stoi

#         @return itos
#         @return stoi
#     '''
#     readability_levels = ['EasilyReadableText', 'StandardReadableText', 'HardlyReadableText']

#     # update itos
#     itos.extend(readability_levels)

#     # update stoi
#     l, rl = len(itos), len(readability_levels)
#     for r in range(rl):
#         stoi[readability_levels[r]] = l - rl + r + 1
    
#     # dump new stoi dict
#     with open("../objects/vocab_stoi.json", "w") as f:
#         json.dump(stoi, f)

#     # dump new itos dict
#     with open("../objects/vocab_itos.json", "w") as f:
#         json.dump(itos, f)

#     return itos, stoi


# Old utils function to match class name to class index
# def parse_output_and_deduce_class(output, itos):
#     '''
#         This function takes as argument the output of a BabyLanguageModel model and parses it to retrieve the predicted value for ReadabilityLevel.

#         @param output (tensor): a list of indexes corresponding to the generated tokens
#         @param itos (list): the mapping from token indexes to their corresponding strings
#     '''
#     decoded_output = decode(output, itos)
#     predicted_class = 3 # let's say we add a class "unable to classify"

#     if 'EasilyReadableText' in decoded_output:
#         predicted_class = 0
#     elif 'StandardReadableText' in decoded_output:
#         predicted_class = 1
#     elif 'HardlyReadableText' in decoded_output:
#         predicted_class = 2

#     return predicted_class


# def get_vocab_info(data):
#     '''
#         Create a vocabulary mapping with training data text processed as a bag of words. Default vocab_size is the number of different words in the whole dataset.
#         TODO change vocab size computation (use torch.vocab ?)
#         TODO add a feature to choose a vocab size (in order to alleviate data processing for bigger datasets). This might be required for open web corpus.
#     '''
#     # here are all the unique characters that occur in this text
#     chars = sorted(list(set(data)))
#     # chars = sorted(list(set(data)))
#     vocab_size = len(chars)
#     # create a mapping from characters to integers
#     stoi = { ch:i for i,ch in enumerate(chars) }
#     itos = { i:ch for i,ch in enumerate(chars) }
#     return vocab_size, stoi, itos


# encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
# decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# def get_vocab(folder_name):
# '''
#     This fuction works, but has been replaced by a more optimized way to both build vocabulary and tokenize data.
    
#     Build a vocabulary of type torch.Vocab from the text dataset stored in folder_name folder.
    
#     @param folder_name (str): the name of the folder where data is stored. Expects a data.txt file.

#     @return vocab (torch.Vocab): the inferred vocabulary, with lowercase tokens, excluding empty tokens
# '''
# file_path = os.path.join(f'../{folder_name}', "data.txt")

# def yield_tokens():
#     with open(file_path, 'r', encoding='utf-8') as f:
#         # yield tokens from each line
#         for line in f:
#             # get initial words that are not empty words
#             words = [ w for w in line.strip().split() if w != ' ']
#             # make simple tokens by lowering words
#             tokens = list(map(lambda x: x.lower(), words))
#             yield tokens

# # build the vocab using the generator
# token_generator = yield_tokens()
# print(type(token_generator))
# vocab = build_vocab_from_iterator(token_generator, specials=["<unk>"])

# return vocab
    
# def read_tinyshakepeare():
#     # wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
#     with open('input.txt', 'r', encoding='utf-8') as f:
#         text = f.read()
#     return text


# TENTATIVE DE FAIRE TOURNER SUR CPU ALORS QUE L'ENTRAINEMENT A LIEU SUR GPU, MAIS C'EST PAS ENCORE ÇA
# with open('../objects/vocab.pt', 'rb') as f:
#     vocab = CPU_Unpickler(f).load()

# with open('../models/babyllm-gptlike.pt', 'rb') as f:
#     model = CPU_Unpickler(f).load()


# Why did I use sentence BERT for that ?

# def apply_sentence_bert(entry, sentence_model, max_length):
#     '''
#         Apply a sentence transformer (S-BERT) model

#         @param entry
#         @param sentence_model
#         @param max_length

#         @param result
#     '''
#     utterance_limit = 15
#     text = entry['text']
#     dial_id = entry['dial_id']
#     utt_id = entry['utt_id']
#     embedding = sentence_model.encode(text, device=args['device']).tolist()

#     # pad utterance ids
#     if len(utt_id) < utterance_limit:
#         utt_id.extend([-1 for _ in range(utterance_limit-len(utt_id))])
#     elif len(utt_id) > utterance_limit:
#         utt_id = utt_id[:utterance_limit]

#     # pad embeddings
#     if len(embedding) < utterance_limit:
#         embedding = add_sentencebert_random_vectors(embedding, utterance_limit - len(embedding), max_length)
#     elif len(embedding) > utterance_limit:
#         embedding = embedding[:utterance_limit]

#     final_embedding = custom_flatten(embedding)

#     result = {'dial_id': dial_id, 'utt_id': utt_id, 'embedding': final_embedding}
#     return result


# def prepare_data_sentence_bert(dataset, sentence_model, max_length):
#     '''
#         A function to wrap up the preprocessing procedure using sentence transformers (S-BERT);

#         @param dataset
#         @param sentence_model
#         @param max_length

#         @return resulting_dataset
#     '''
#     model = SentenceTransformer(f'sentence-transformers/{sentence_model}')

#     for split in ['train', 'validation', 'test']:
#         dataset[split] = dataset[split].map(lambda e: apply_sentence_bert(e, model, max_length))

#     processed_dataset = {
#         'train':Dataset.from_dict({
#             'dial_id': dataset['train']['dial_id'],
#             'utt_id': dataset['train']['utt_id'],
#             'embedding': dataset['train']['embedding']
#             }),
#         'validation': Dataset.from_dict({
#             'dial_id': dataset['validation']['dial_id'],
#             'utt_id': dataset['validation']['utt_id'],
#             'embedding': dataset['validation']['embedding']
#             }),
#         'test':Dataset.from_dict({
#             'dial_id': dataset['test']['dial_id'],
#             'utt_id': dataset['test']['utt_id'],
#             'embedding': dataset['test']['embedding']
#             })
#         }
    
#     resulting_dataset = DatasetDict(processed_dataset)
#     return resulting_dataset


# generate from randomly selected prompts using multiprocessing

# def generate_from_random_prompts(args, model, stoi, itos, batch_size, n_threads=None):
#     '''
#         Version parallélisée (non fonctionnelle)
#     '''
#     batch_labels, batch_generations = [], []
#     def process_set(args, model, stoi, itos):
#         p = rd.uniform()
#         if p < 1/3:
#             prompt = f"A EasilyReadableText sentence: "
#             batch_labels.append(0)
#         elif p > 1/3 and p < 2/3:
#             prompt = f"A StandardReadableText sentence: "
#             batch_labels.append(1)
#         else:
#             prompt = f"A HardlyReadableText sentence: "
#             batch_labels.append(2)
#         prompt = encode(stoi, tokenizer.tokenize(prompt))
#         prompt = torch.tensor(prompt, dtype=torch.long).unsqueeze(-1).to(args['device'])
#         generation = model.generate(prompt, max_new_tokens=20, block_size=args['block_size'])[0].tolist()
#         generation = decode(generation, itos)
#         batch_generations.append(generation)

#     # paralellize model calls
#     processes = [mp.Process(target=process_set, args=(args, model, stoi, itos)) for _ in range(batch_size)]
#     for process in processes:
#         process.start()
#     for process in processes:
#         process.join() 


# to add in BabyLanguageModel to proceed option 2 in finetune.py

    # def predict_readability_levels(self, idx, block_size):
    #     '''
    #     This function is usefull only for option 2 (cf. bottom of finetune.py) which is no longer used in the code.
    #     '''
    #     idx_cond = idx[:, -block_size:]
    #     # handle out-of-range indices
    #     idx_cond = torch.clamp(idx_cond, max=self.token_embedding_table.num_embeddings - 1)

    #     logits, _ = self(idx_cond)
    #     logits = logits[:, -1, :] # becomes (B, C)
    #     probas = F.softmax(logits, dim=-1)

    #     # return token index with max proba among the 3 readability levels
    #     last_token_probas = probas[0]
    #     v_size = probas[0].size()[0]

    #     rl_probas = [last_token_probas[k] for k in [v_size - i for i in range(3, 0, -1)]]
        
    #     return torch.stack(rl_probas)
    