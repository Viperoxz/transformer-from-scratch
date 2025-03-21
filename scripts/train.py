# Script to train the model
import torch
import torch.nn as nn
import torch.optim as optim
import math
import time
from src.model.transformer import make_model
from src.model.utils import subsequent_mask
from typing import Tuple, Optional


class Batch:
    ''' 
    Object for holding a batch of data with mask during training.
    '''

    def __init__(self, src, tgt=None, pad=2):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if tgt is not None:
            self.tgt = tgt[:, :-1]
            self.tgt_y = tgt[:, 1:]
            self.tgt_mask = self.make_std_mask(self.tgt, pad) 
            self.ntokens = (self.tgt_y != pad).data.sum() # Number of tokens in target excluding padding

    @staticmethod
    def make_std_mask(tgt, pad):
        ''' Create a mask to hide padding and future words. '''
        tgt_mask = (tgt != pad).unsqueeze(-2) 
        '''
        tgt != pad: Creates a boolean tensor with True for non-padding tokens and False for padding tokens. 
                    Example: tgt = [[1, 2, 3, 2, 2]] -> [[True, True, True, False, False]] (assuming pad=2).
        unsqueeze(-2): Adds a dimension at the second-to-last position, transforming the shape from (batch_size, tgt_len) 
                    to (batch_size, 1, tgt_len) to match the attention mechanism's requirements.
        '''
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(tgt_mask)
        return tgt_mask
    

class TrainState:
    ''' Track number of steps, examples, loss and start time of training. '''

    def __init__(self, step: int = 0, accum_step: int = 0, samples: int = 0, tokens: int = 0):
        self.step: int = step
        self.accum_step: int = accum_step
        self.samples: int = samples
        self.tokens: int = tokens
        

def run_epoch(
    data_iter,
    model,
    loss_compute,
    optimizer,
    scheduler,
    mode: str = "train",
    accum_iter: int = 1,
    train_state: Optional[TrainState] = None,
) -> tuple[float, TrainState]:
    """Train a single epoch"""
    if train_state is None:
        train_state = TrainState()
    
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    n_accum = 0
    
    for i, batch in enumerate(data_iter):
        out = model.forward(batch.src, batch.tgt, batch.src_mask, batch.tgt_mask)
        loss, loss_node = loss_compute(out, batch.tgt_y, batch.ntokens)
        
        if mode == "train" or mode == "train+log":
            loss_node.backward()
            train_state.step += 1
            train_state.samples += batch.src.shape[0]
            train_state.tokens += batch.ntokens.item()
            
            if i % accum_iter == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                n_accum += 1
                train_state.accum_step += 1
            scheduler.step()

        total_loss += loss
        total_tokens += batch.ntokens.item()
        tokens += batch.ntokens.item()
        
        if i % 40 == 1 and (mode == "train" or mode == "train+log"):
            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - start
            print(
                (
                    "Epoch Step: %6d | Accumulation Step: %3d | Loss: %6.2f "
                    "| Tokens / Sec: %7.1f | Learning Rate: %6.1e"
                ) % (i, n_accum, loss / batch.ntokens, tokens / elapsed, lr)
            )
            start = time.time()
            tokens = 0
        
        del loss
        del loss_node
    
    return total_loss / total_tokens, train_state