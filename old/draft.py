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