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