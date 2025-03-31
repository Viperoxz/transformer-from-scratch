# Full Transformer model
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from src.model.utils import clones, LayerNorm, SublayerConnection, subsequent_mask
from src.model.attention import MultiHeadedAttention
from src.model.feedforward import FeedForward
from src.model.positional_encoding import PositionalEncoding
from src.model.encoder import Encoder, EncoderLayer
from src.model.decoder import Decoder, DecoderLayer
from src.model.embeddings import Embeddings


class Generator(nn.Module):
    """Generator network."""

    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask,
                            tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)
    
class DecoderOnlyTransformer(nn.Module):
    """A Decoder-only Transformer for language modeling."""
    def __init__(self, decoder, tgt_embed, generator):
        super().__init__()
        self.decoder = decoder
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, tgt, tgt_mask):
        """Process target sequence and predict next tokens."""
        embedded = self.tgt_embed(tgt)
        output = self.decoder(embedded, tgt_mask)
        return self.generator(output)
    

# def make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
#     c = copy.deepcopy
#     attn = MultiHeadedAttention(h, d_model)
#     ff = FeedForward(d_model, d_ff, dropout)
#     position = PositionalEncoding(d_model, dropout)
#     model = EncoderDecoder(
#         Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
#         Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
#         nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
#         nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
#         Generator(d_model, tgt_vocab)
#     )

#     for p in model.parameters():
#         if p.dim() > 1:
#             nn.init.xavier_uniform_(p)
#     return model

def make_model(vocab_size, N=2, d_model=256, d_ff=1024, h=4, dropout=0.1):
    """Create a smaller Decoder-only Transformer."""
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = FeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    
    model = DecoderOnlyTransformer(
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, vocab_size), c(position)),
        Generator(d_model, vocab_size)
    )

    # Khởi tạo tham số
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model

if __name__ == "__main__":
    vocab_size = 10000  
    model = make_model(vocab_size)
    x = torch.randint(0, vocab_size, (8, 128))  # Batch size 8, seq len 128
    mask = subsequent_mask(128).unsqueeze(0).repeat(8, 1, 1)
    out = model(x, mask)
    print(out.shape)
    