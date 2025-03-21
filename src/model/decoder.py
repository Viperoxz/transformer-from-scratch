# Transformer Decoder implementation
import torch.nn as nn
from src.model.utils import clones, LayerNorm
from src.model.utils import SublayerConnection



class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, N):
        super().__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super().__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3) # 3 sub-layers: self-attn, src-attn, feed forward

    def forward(self, x, memory, src_mask, tgt_mask):
        # memory is the output of the encoder
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask)) # masked self-attention
        x = self.sublayer[1](x, lambda x: self.src_attn(x, memory, memory, src_mask)) # src-attention
        return self.sublayer[2](x, self.feed_forward) # feed forward



