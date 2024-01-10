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