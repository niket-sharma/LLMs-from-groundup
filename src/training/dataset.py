import torch
from torch.utils.data import Dataset, DataLoader
import os
import pickle
import numpy as np


class TextDataset(Dataset):
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        chunk = self.data[idx:idx + self.block_size + 1]
        x = torch.from_numpy((chunk[:-1]).astype(np.int64))
        y = torch.from_numpy((chunk[1:]).astype(np.int64))
        return x, y


class SimpleTokenizer:
    def __init__(self):
        self.vocab = {}
        self.inverse_vocab = {}
        self.vocab_size = 0

    def build_vocab(self, text):
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        self.vocab = {ch: i for i, ch in enumerate(chars)}
        self.inverse_vocab = {i: ch for i, ch in enumerate(chars)}

    def encode(self, text):
        return [self.vocab[c] for c in text]

    def decode(self, tokens):
        return ''.join([self.inverse_vocab[t] for t in tokens])

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump({
                'vocab': self.vocab,
                'inverse_vocab': self.inverse_vocab,
                'vocab_size': self.vocab_size
            }, f)

    def load(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.vocab = data['vocab']
            self.inverse_vocab = data['inverse_vocab']
            self.vocab_size = data['vocab_size']


def prepare_data(text_file, block_size=1024, train_split=0.9):
    with open(text_file, 'r', encoding='utf-8') as f:
        text = f.read()

    tokenizer = SimpleTokenizer()
    tokenizer.build_vocab(text)

    # Encode the entire text
    data = np.array(tokenizer.encode(text), dtype=np.uint16)

    # Train/validation split
    n = int(train_split * len(data))
    train_data = data[:n]
    val_data = data[n:]

    # Create datasets
    train_dataset = TextDataset(train_data, block_size)
    val_dataset = TextDataset(val_data, block_size)

    return train_dataset, val_dataset, tokenizer


def create_dataloaders(train_dataset, val_dataset, batch_size=64, num_workers=0):
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader