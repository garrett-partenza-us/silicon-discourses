import io
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data import get_tokenizer
from torch.utils.data import DataLoader
import torch.nn.functional as F
import re
import torch
import string

def yield_tokens(file_path, tokenizer):
    with io.open(file_path, encoding = 'utf-8') as file:
        for line in file:
            if re.search('[a-zA-Z]', line):
                line = line.lower()
                line = line.translate(str.maketrans('', '', string.punctuation))
                line = line.replace('\n', '')
                line = line.strip()
                line = tokenizer(line)
                line.insert(0, "<sos>")
                line.append("<eos>")
                yield line
                
def padding(arr, seq_len, mode="right"):
    assert mode=="right" or mode=="left", "invalid padding mode"
    if mode=="right":
        return F.pad(arr, (0, seq_len-arr.nelement()))
    else:
        return F.pad(arr, (seq_len-arr.nelement()), 0)
                    
textfile = "data/discourses.mb.txt"
tokenizer = get_tokenizer("spacy")
contains_chars = lambda x: re.search('[a-zA-Z]', x)
vocab = build_vocab_from_iterator(yield_tokens(textfile, tokenizer), specials=["<unk>", "<eos>", "<sos>", "<pad>"])
vocab.set_default_index(-1)     
num_tokens = len(vocab)
d_model = 768
d_embed = 1024
dropout = 0.1
seq_len = 32
batch_size = 128
sequences, targets = [], []

with io.open(textfile, encoding = 'utf-8') as file:
    for line in file.readlines():
        if contains_chars(line):
            line = line.lower()
            line = line.translate(str.maketrans('', '', string.punctuation))
            line = line.replace('\n', '')
            line = line.strip()
            words = tokenizer(line)
            words.insert(0, "<sos>")
            words.append("<eos>")
            tokens = [vocab[word] for word in words]
            for i in range(1, len(tokens)): 
                n_gram_seqs = tokens[:i+1]
                targets.append(n_gram_seqs.pop(-1))
                sequences.append(padding(torch.tensor(n_gram_seqs), seq_len))
                
dataset = DataLoader(list(zip(sequences, targets)), shuffle=True, batch_size=batch_size, drop_last=True)


import torch
import torch.nn as nn
from torch.nn.functional import log_softmax
import numpy as np
import matplotlib.pyplot as plt
import copy
import math

def positional_encoding(seq_len, d, n=10000):
    p = np.zeros((seq_len, d))
    for k in range(seq_len):
        for i in np.arange(int(d/2)):
            denominator = np.power(n, 2*i/d)
            p[k, 2*i] = np.sin(k/denominator)
            p[k, 2*i+1] = np.cos(k/denominator)
    return p

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

# literally the same high level wrapper as encoder, where differences occur within the actual decoder layer class
class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)
    
class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        # attention over itself
        self.self_attn = self_attn
        # attention over encoder
        self.feed_forward = feed_forward
        # three sublayers now due to encoder-decoder attention
        self.sublayer = clones(SublayerConnection(size, dropout), 2)

    def forward(self, x):
        "Follow Figure 1 (right) for connections."
        # same as the encoder layer, plus an additional sublayer for the encoder-decoder attention
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x))
        return self.sublayer[1](x, self.feed_forward)
    
class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        # all h heads are calculated in a single matrix multiplication (better parallelization)
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [
            # reshapes into the individual attention heads of shape h x d_model // h
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]
        

        # 2) Apply attention on all the projected vectors in batch.
        # calculate attention using paper formula (scaled dot product attention to prevent small gradients)
        x, self.attn = attention(
            query, key, value, mask=mask, dropout=self.dropout
        )
        
        # 3) "Concat" using a view and apply a final linear.
        # concats all attention layers back into d_model size feature vector
        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(nbatches, -1, self.h * self.d_k)
        )
        
            
        del query
        del key
        del value
        
        # applys fourth linear transformation
        return self.linears[-1](x)
    
# wrapper to apply residual connection addition as well as layer normalization
class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        # bunch of math, tldr; rescales features with mean of 0 and std of 1 (paper -> https://arxiv.org/abs/1607.06450)
        # not sure what the function of eps is here
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    
# simple single hidden layer feed forward nueral network with relu acitvation funcftion
class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(self.w_1(x).relu()))
    
class Generator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        # matmul(1xd_model, d_modelxvocav) gives a 1 dimensional array of softmaxed values for each vocab word
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return log_softmax(self.proj(x), dim=-1)
    
h = 12
N = 12
d_ff = d_model*4
model = Decoder(
    DecoderLayer(
        d_model,
        MultiHeadedAttention(h, d_model),
        PositionwiseFeedForward(d_model, d_ff, dropout),
        dropout
    ), 
    N
)
luc = nn.Embedding(num_tokens, d_model)
generator = Generator(d_model, num_tokens)
P = torch.tensor(positional_encoding(seq_len, d_model, n=10000)).repeat(batch_size, 1, 1)

for param in model.parameters():
    if param.dim() > 1:
        nn.init.xavier_uniform_(param)
        
from torch import optim 
from tqdm import tqdm

def probs2words(probs):
    _, next_words = torch.max(probs, dim=1)
    next_words = (vocab.lookup_token(x) for x in next_words)
    return list(next_words)
    
embedding_func = lambda x: luc(torch.tensor(x, dtype=torch.long))
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-6)

model.train()
luc.train()

train_loss = []

c = 0
for batch in tqdm(dataset):
    optimizer.zero_grad()
    x, y = batch
    x = luc(x)
    x = torch.add(x, P)
    out = model(x.float())
    probs = generator(out[:, -1])
    loss = loss_func(probs, y)
    loss.backward()
    optimizer.step()
    
    train_loss.append(loss.item())
    
    print(probs2words(probs))
    if c == 5:
        break
    else:
        c+=1
    
import matplotlib.pyplot as plt

plt.plot(train_loss, label="cross entropy loss", c="b")
plt.savefig("loss.png")