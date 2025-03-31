# DataLoader for batching
import torch
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import glob
import json
import os
import spacy

class TinyStoriesDataset(Dataset):
    def __init__(self, data_dir, vocab, max_length=128):
        self.vocab = vocab
        self.max_length = max_length
        self.spacy_en = spacy.load("en_core_web_sm")
        self.data = []
        
        # Đọc tất cả file JSON từ thư mục raw
        for file in glob.glob(str(data_dir / "raw/*.json")):
            with open(file, "r") as f:
                stories = json.load(f)
            for story in stories:
                tokens = self.tokenize(story["story"])
                if len(tokens) > 0:  # Chỉ thêm nếu có token
                    self.data.append(tokens[:max_length])

    def tokenize(self, text):
        tokens = [self.vocab["<s>"]] + [self.vocab.get(token.text, self.vocab["<unk>"]) 
                                        for token in self.spacy_en(text)] + [self.vocab["</s>"]]
        return tokens

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq = self.data[idx]
        # Padding nếu ngắn hơn max_length
        if len(seq) < self.max_length:
            seq += [self.vocab["<blank>"]] * (self.max_length - len(seq))
        x = torch.tensor(seq[:-1], dtype=torch.long)  # Input
        y = torch.tensor(seq[1:], dtype=torch.long)   # Target
        return x, y

def get_dataloader(data_dir, vocab, batch_size=8, max_length=128):
    dataset = TinyStoriesDataset(Path(data_dir), vocab, max_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

if __name__ == "__main__":
    from pathlib import Path
    vocab = torch.load(Path("data/vocab/vocab.pt"))
    dataloader = get_dataloader("data", vocab)
    for x, y in dataloader:
        print(x.shape, y.shape)
        break