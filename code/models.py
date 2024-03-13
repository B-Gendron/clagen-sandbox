# Inspired from: https://colab.research.google.com/drive/1JMLa53HDuA-i7ZBmqV7ZnA3c_fvtXnx-?usp=sharing#scrollTo=nql_1ER53oCf 

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaForCausalLM
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.generation import utils

# set default tensor type
torch.set_default_dtype(torch.float16)


def smooth_input(x_input, max_prop=0.6):
    '''
        This function is used in TrainableHeadAdapters model forward to smooth the one-hot encoding of the input to a 60/20/20 like vector (in its default config). It is meant to be used with n_rl classes and a max_prop which allows an equal repartition of other proportion over other labels.
 
        @param x_input (torch.Tensor):
        @param max_prop (float):
    '''
    n_rl = x_input.size()[1]
    # check if it is possible to use this max-prop and compute the other proportions out of it
    assert int((1-max_prop)*100 % (n_rl-1)) == 0 
    # compute the minimal proportion since it can be computed
    min_prop = (1-max_prop)/(n_rl-1) 
    new_classes_vectors = []
    for classes_one_hot in x_input:
        i = torch.argmax(classes_one_hot)
        x = torch.full((n_rl,), min_prop)
        x[i] = max_prop
        new_classes_vectors.append(x)

    return torch.stack(new_classes_vectors)


class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size, n_embd, block_size, dropout):
        super(Head, self).__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)   # (B,T,C)
        q = self.query(x) # (B,T,C)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,C)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size, n_embd, block_size, dropout):
        super(MultiHeadAttention, self).__init__()
        # nn.ModuleList bc we don't want to run them here, just to define a list of attention heads
        self.heads = nn.ModuleList([Head(head_size, n_embd, block_size, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ A block following classic MLP architecture: Linear/ReLU/Linear/Dropout block """

    def __init__(self, n_embd, dropout):
        super(FeedFoward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class DecoderBlock(nn.Module):
    """ A simple decoder block """

    def __init__(self, n_embd, n_heads, block_size, dropout):
        # n_embd: embedding dimension
        # n_head: the number of heads we'd like
        super(DecoderBlock, self).__init__()
        head_size = n_embd // n_heads
        self.sa = MultiHeadAttention(n_heads, head_size, n_embd, block_size, dropout)
        self.ffwd = FeedFoward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class BabyLanguageModel(nn.Module):
    """ A basic and very small GPT-like model for fast training and inference."""

    def __init__(self, args):
        super(BabyLanguageModel, self).__init__()
        # get all values from args dict
        vocab_size = args['vocab_size']
        n_embd = args['n_embd']
        block_size = args['block_size']
        n_heads = args['n_heads']
        n_layers = args['n_layers']
        dropout = args['dropout']

        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[DecoderBlock(n_embd, n_heads, block_size, dropout) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

        self.device = args['device']

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=self.device)) # (T,C)

        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens, block_size):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx


class TrainableHead(nn.Module):
    '''
        An auxiliary model to update the babylm model lm_head layer using generation redability levels predictions.
    '''
    def __init__(self, args, n_rl=3):
        super(TrainableHead, self).__init__()
        n_embd = args['n_embd']
        vocab_size = args['vocab_size']
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False) # bias=False to be consistent with llama
        self.anti_pool = nn.Linear(n_rl, n_embd)
        self.pool = nn.Linear(vocab_size, n_rl)
        self.anti_pool_llama = nn.Linear(n_rl, 4096)
        self.pool_llama = nn.Linear(4096, n_rl)
        self.softmax = nn.Softmax(dim=1)
        self.batch_norm = nn.BatchNorm1d(n_rl)

        self.decoder = LlamaDecoderLayer(config=args['config'])

        self.args = args
        self.penalty = 1e-2

    def forward(self, x_input):
        if self.args['hf'] == 'llama':
            x = self.anti_pool_llama(x_input)
            x = self.decoder(x)
            x = self.pool_llama(x[0].squeeze())
            x = self.penalty*self.softmax(x) + x_input
        else:
            # x = self.softmax(x_input)
            x = self.anti_pool(x_input)
            x = self.lm_head(x)
            x = self.pool(x)
            x = self.penalty*self.softmax(x) + x_input
            # x = self.batch_norm(x) + x_input

        return x
    
class TrainableHeadAdapters(nn.Module):
    '''
        An auxiliary model to update the babylm model lm_head layer using generation redability levels predictions.
    '''
    def __init__(self, args):
        super(TrainableHeadAdapters, self).__init__()
        vocab_size = args['vocab_size']
        max_new_tokens = args['max_new_tokens']
        self.pool = nn.Linear(max_new_tokens*vocab_size, vocab_size) 
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.classification_layer = nn.Linear(vocab_size, 2) # [32, vocab_size] --> [32, 2] binary clf seen as multiclass clf to avoid num approx problems
        self.softmax = nn.Softmax(dim=-1)

        self.model = args['model']
        self.args = args
        self.penalty = 1e-1

    def forward(self, input_ids):
        '''
            b = batch_size (default=32)
            t = max_new_tokens (default=20)
            v = vocab_size (default=32000 if llama, another number I don't remeber if gemma)
        '''
        # forward path inside the model
        x = self.model(input_ids)

        # dimension reduction to match with classification layer
        x = self.pool(x[0].view(x[0].size(0), -1).half()) # [b, t, v] --> [b, v]
        x = self.relu(x)
        x = self.dropout(x)

        # classification layer with output softmax
        x = self.classification_layer(x) # [b, v] --> [b, 2]
        x = self.softmax(x)

        return x


class BinaryTrainableModel(nn.Module):
    '''
        TODO put this in draft
        The "auxiliary model" used for the binary classification approach, which is unfortunately not adapted to our usecase :/
    '''
    def __init__(self, args):
        super(BinaryTrainableModel, self).__init__()
        n_embd = args['n_embd']
        self.linear = nn.Linear(args['n_layers']*n_embd, n_embd)
        self.classification_layer = nn.Linear(n_embd, 2) # binary classification scenario using 2 output neurons + softmax (1 neuron + sigmoid seem to cause problems to colleagues...)
        self.softmax = nn.Softmax(dim=-1)
        self.args = args

    def forward(self, hidden_states):
        '''
            Fake a forward path through the model using a compliant input (a tensor of completely random input_ids)
        '''
        hs = torch.squeeze(hidden_states).half() # [32, 33, 1, 4096] --> [32, 33, 4096]
        x = self.linear(hs.view(self.args['batch_size'], -1)) # [32, 33, 4096] --> [32, 4096]
        x = self.classification_layer(x) # [32, 4096] --> [32, 2]
        x = self.softmax(x)

        return x