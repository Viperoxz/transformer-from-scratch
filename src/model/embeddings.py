import torch.nn as nn
import math


class Embeddings(nn.Module):
    
    def __init__(self, d_model, vocab):
        super().__init__()
        self.word_embedding = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.word_embedding(x) * math.sqrt(self.d_model)