from pathlib import Path
import torch
import spacy
from torchtext.vocab import build_vocab_from_iterator
import os
import json
import glob
import requests
from tqdm import tqdm

DATA_DIR = Path("data")
VOCAB_DIR = DATA_DIR / "vocab"
VOCAB_PATH = VOCAB_DIR / "vocab.pt"
TINY_STORIES_DIR = DATA_DIR / "TinyStories_all_data"
TINY_STORIES_TAR = DATA_DIR / "TinyStories_all_data.tar.gz"

def download_tiny_stories():
    """Tải Tiny Stories nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not TINY_STORIES_TAR.exists():
        url = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories_all_data.tar.gz"
        print(f"Downloading Tiny Stories to {TINY_STORIES_TAR}...")
        resp = requests.get(url, stream=True)
        total = int(resp.headers.get("content-length", 0))
        with open(TINY_STORIES_TAR, "wb") as file, tqdm(total=total, unit="iB", unit_scale=True) as bar:
            for data in resp.iter_content(chunk_size=1024):
                size = file.write(data)
                bar.update(size)
    else:
        print(f"{TINY_STORIES_TAR} already exists, skipping download...")

    if not TINY_STORIES_DIR.exists():
        print(f"Unpacking {TINY_STORIES_TAR}...")
        os.system(f"tar -xzf {TINY_STORIES_TAR} -C {DATA_DIR}")
    else:
        print(f"{TINY_STORIES_DIR} already exists, skipping unpacking...")

def load_tokenizer():
    """Tải tokenizer SpaCy tiếng Anh."""
    model = "en_core_web_sm"
    try:
        spacy.load(model)
    except OSError:
        print(f"Downloading {model}...")
        os.system(f"python -m spacy download {model}")
    return spacy.load(model)

def tokenize(text: str, tokenizer) -> list[str]:
    """Chia nhỏ văn bản thành token."""
    return [token.text for token in tokenizer(text)]

def yield_tokens(data_dir, tokenizer):
    """Lấy token từ Tiny Stories."""
    shard_filenames = sorted(glob.glob(str(data_dir / "*.json")))
    for shard in shard_filenames:
        with open(shard, "r") as f:
            data = json.load(f)
        for example in data:
            yield tokenize(example["story"], tokenizer)

def build_vocab(spacy_en):
    """Xây dựng từ điển từ Tiny Stories."""
    tokenizer_en = lambda x: tokenize(x, spacy_en)

    print("Building English vocabulary from Tiny Stories...")
    vocab = build_vocab_from_iterator(
        yield_tokens(TINY_STORIES_DIR, tokenizer_en),
        min_freq=2,
        specials=['<s>', '</s>', '<unk>', '<blank>']
    )
    vocab.set_default_index(vocab['<unk>'])
    return vocab

def load_vocab(spacy_en):
    """Tải từ điển từ file hoặc xây dựng nếu chưa có."""
    VOCAB_DIR.mkdir(parents=True, exist_ok=True)
    if not VOCAB_PATH.exists():
        print("Vocabulary not found, building new one...")
        download_tiny_stories()  # Tải dữ liệu nếu chưa có
        vocab = build_vocab(spacy_en)
        torch.save(vocab, VOCAB_PATH)
        print(f"Vocabulary saved to {VOCAB_PATH}")
    else:
        print("Loading vocabulary from disk...")
        vocab = torch.load(VOCAB_PATH)

    print(f"Vocabulary size: {len(vocab)} (EN)")
    return vocab

if __name__ == "__main__":
    print("Starting tokenization and vocabulary building...")
    spacy_en = load_tokenizer()
    vocab = load_vocab(spacy_en)
    print("Tokenization and vocabulary building completed.")