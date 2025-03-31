# Script to train the model
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.train.trainer import train_model
from src.model.transformer import make_model
from src.data_utils.dataloader import get_dataloader
import torch
from pathlib import Path


# class Batch:
#     ''' 
#     Object for holding a batch of data with mask during training.
#     '''

#     def __init__(self, src, tgt=None, pad=2):
#         self.src = src
#         self.src_mask = (src != pad).unsqueeze(-2)
#         if tgt is not None:
#             self.tgt = tgt[:, :-1]
#             self.tgt_y = tgt[:, 1:]
#             self.tgt_mask = self.make_std_mask(self.tgt, pad) 
#             self.ntokens = (self.tgt_y != pad).data.sum() # Number of tokens in target excluding padding

#     @staticmethod
#     def make_std_mask(tgt, pad):
#         ''' Create a mask to hide padding and future words. '''
#         tgt_mask = (tgt != pad).unsqueeze(-2) 
#         '''
#         tgt != pad: Creates a boolean tensor with True for non-padding tokens and False for padding tokens. 
#                     Example: tgt = [[1, 2, 3, 2, 2]] -> [[True, True, True, False, False]] (assuming pad=2).
#         unsqueeze(-2): Adds a dimension at the second-to-last position, transforming the shape from (batch_size, tgt_len) 
#                     to (batch_size, 1, tgt_len) to match the attention mechanism's requirements.
#         '''
#         tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(tgt_mask)
#         return tgt_mask
    

if __name__ == "__main__":
    # Cấu hình
    data_dir = "data"
    vocab_path = Path("data/vocab/vocab.pt")
    batch_size = 8
    epochs = 3
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load vocab và dữ liệu
    vocab = torch.load(vocab_path)
    dataloader = get_dataloader(data_dir, vocab, batch_size=batch_size)

    # Khởi tạo mô hình
    model = make_model(vocab_size=len(vocab))

    # Huấn luyện
    train_model(model, dataloader, vocab, epochs=epochs, device=device)