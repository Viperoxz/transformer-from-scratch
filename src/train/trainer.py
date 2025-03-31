# src/train/trainer.py
import torch
import torch.nn as nn
import torch.optim as optim
import time
from src.model.transformer import subsequent_mask
from typing import Tuple, Optional
from pathlib import Path

class TrainState:
    """Track number of steps, examples, loss and start time of training."""
    def __init__(self, step: int = 0, accum_step: int = 0, samples: int = 0, tokens: int = 0):
        self.step: int = step
        self.accum_step: int = accum_step
        self.samples: int = samples
        self.tokens: int = tokens

def compute_loss(output, targets, vocab):
    """Compute loss for language modeling."""
    criterion = nn.CrossEntropyLoss(ignore_index=vocab["<blank>"])
    loss = criterion(output.view(-1, len(vocab)), targets.view(-1))
    return loss

def run_epoch(
    data_iter,
    model,
    optimizer,
    scheduler,
    vocab,
    device: str,
    mode: str = "train",
    accum_iter: int = 1,
    train_state: Optional[TrainState] = None,
) -> Tuple[float, TrainState]:
    """Train a single epoch for Decoder-only Transformer."""
    if train_state is None:
        train_state = TrainState()
    
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    n_accum = 0
    
    for i, (x, y) in enumerate(data_iter):
        x, y = x.to(device), y.to(device)
        tgt_mask = subsequent_mask(x.size(1)).to(device)
        
        out = model(x, tgt_mask)
        loss = compute_loss(out, y, vocab)
        
        if mode == "train" or mode == "train+log":
            loss.backward()
            train_state.step += 1
            train_state.samples += x.shape[0]
            train_state.tokens += (y != vocab["<blank>"]).sum().item()
            
            if i % accum_iter == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                n_accum += 1
                train_state.accum_step += 1
            scheduler.step()

        total_loss += loss.item()
        total_tokens += (y != vocab["<blank>"]).sum().item()
        tokens += (y != vocab["<blank>"]).sum().item()
        
        if i % 40 == 1 and (mode == "train" or mode == "train+log"):
            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - start
            print(
                (
                    "Epoch Step: %6d | Accumulation Step: %3d | Loss: %6.2f "
                    "| Tokens / Sec: %7.1f | Learning Rate: %6.1e"
                ) % (i, n_accum, loss.item(), tokens / elapsed, lr)
            )
            start = time.time()
            tokens = 0
    
    return total_loss / total_tokens if total_tokens > 0 else 0.0, train_state

def train_model(model, dataloader, vocab, epochs=3, lr=0.0003, device="cuda"):
    """Train the model over multiple epochs."""
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        print("CUDA not available, falling back to CPU")
    
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min(step ** -0.5, step * 2000 ** -1.5)
    )
    train_state = TrainState()

    for epoch in range(epochs):
        print(f"Starting Epoch {epoch + 1}/{epochs}")
        loss, train_state = run_epoch(
            dataloader, model, optimizer, scheduler, vocab, device, mode="train", train_state=train_state
        )
        print(f"Epoch {epoch + 1} Loss: {loss:.4f}")
        checkpoint_path = Path("checkpoints") / f"epoch_{epoch}.pt"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), checkpoint_path)