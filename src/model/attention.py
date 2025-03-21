# Multi-Head Attention implementation
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from src.model.utils import clones



def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1) 
    scores = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):

    def __init__(self, n_heads, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super().__init__()
        assert d_model % n_heads == 0 # d_model / n_heads = d_k = d_v
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1) # Same mask applied to all n_heads heads
        n_batches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => (n_heads x d_k)
        # Apply linear transformation to query, key, value
        query = self.linears[0](query) 
        key = self.linears[1](key)
        value = self.linears[2](value)
        ''' 
        Split the d_model into n_heads heads: (n_batches, seq_len, d_model) -> (n_batches, seq_len, n_heads, d_k)
        Then transpose the result to (n_batches, n_heads, seq_len, d_k) to prepare for the subsequent matrix multiplication.
        '''
        query = query.view(n_batches, -1, self.n_heads, self.d_k).transpose(1, 2)
        key = key.view(n_batches, -1, self.n_heads, self.d_k).transpose(1, 2)
        value = value.view(n_batches, -1, self.n_heads, self.d_k).transpose(1, 2)

        # 2) Apply attention on all the projected vectors in batch
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout) # x shape: (n_batches, n_heads, seq_len, d_k)

        # 3) Concat using a view and apply a final linear
        x = x.transpose(1, 2).contiguous().view(n_batches, -1, self.n_heads * self.d_k)

        del query, key, value
        return self.linears[-1](x)