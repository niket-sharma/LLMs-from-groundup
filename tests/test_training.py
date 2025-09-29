import pytest
import torch
import tempfile
import os
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from training.dataset import TextDataset, SimpleTokenizer, prepare_data, create_dataloaders
from training.trainer import GPTTrainer, estimate_loss
from models.gpt import SmallGPT


class TestSimpleTokenizer:
    def test_tokenizer_build_vocab(self):
        text = "hello world"
        tokenizer = SimpleTokenizer()
        tokenizer.build_vocab(text)

        assert tokenizer.vocab_size == len(set(text)) + 1  # +1 for UNK token
        assert 'h' in tokenizer.vocab
        assert 'w' in tokenizer.vocab
        assert ' ' in tokenizer.vocab

    def test_tokenizer_encode_decode(self):
        text = "hello world"
        tokenizer = SimpleTokenizer()
        tokenizer.build_vocab(text)

        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)

        assert decoded == text
        assert len(encoded) == len(text)

    def test_tokenizer_save_load(self):
        text = "hello world"
        tokenizer = SimpleTokenizer()
        tokenizer.build_vocab(text)

        with tempfile.NamedTemporaryFile(delete=False) as f:
            tokenizer.save(f.name)

            new_tokenizer = SimpleTokenizer()
            new_tokenizer.load(f.name)

            assert new_tokenizer.vocab_size == tokenizer.vocab_size
            assert new_tokenizer.vocab == tokenizer.vocab

        os.unlink(f.name)


class TestTextDataset:
    def test_dataset_creation(self):
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        block_size = 3

        dataset = TextDataset(data, block_size)

        assert len(dataset) == len(data) - block_size
        assert len(dataset) == 7

    def test_dataset_getitem(self):
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        block_size = 3

        dataset = TextDataset(data, block_size)
        x, y = dataset[0]

        assert x.tolist() == [1, 2, 3]
        assert y.tolist() == [2, 3, 4]

    def test_prepare_data(self):
        # Create a temporary text file
        text_content = "hello world this is a test text for the dataset preparation"

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write(text_content)
            temp_file = f.name

        try:
            train_dataset, val_dataset, tokenizer = prepare_data(temp_file, block_size=5)

            assert len(train_dataset) > 0
            assert len(val_dataset) > 0
            assert tokenizer.vocab_size > 0

            # Test that we can get items from datasets
            x, y = train_dataset[0]
            assert len(x) == 5
            assert len(y) == 5

        finally:
            os.unlink(temp_file)

    def test_create_dataloaders(self):
        # Create dummy datasets
        data = list(range(100))
        train_dataset = TextDataset(data[:80], block_size=5)
        val_dataset = TextDataset(data[80:], block_size=5)

        train_loader, val_loader = create_dataloaders(
            train_dataset, val_dataset, batch_size=4
        )

        # Test that we can iterate through dataloaders
        train_batch = next(iter(train_loader))
        val_batch = next(iter(val_loader))

        assert len(train_batch) == 2  # x, y
        assert len(val_batch) == 2    # x, y
        assert train_batch[0].shape[0] <= 4  # batch size
        assert train_batch[1].shape[0] <= 4  # batch size


class TestGPTTrainer:
    def test_trainer_init(self):
        # Create a small model and dummy data
        model = SmallGPT(vocab_size=100, d_model=64, n_heads=4, n_layers=2)

        # Create dummy datasets
        data = list(range(200))
        train_dataset = TextDataset(data[:160], block_size=5)
        val_dataset = TextDataset(data[160:], block_size=5)

        train_loader, val_loader = create_dataloaders(
            train_dataset, val_dataset, batch_size=4
        )

        trainer = GPTTrainer(model, train_loader, val_loader)

        assert trainer.model is not None
        assert trainer.optimizer is not None
        assert trainer.scheduler is not None
        assert trainer.best_val_loss == float('inf')

    def test_estimate_loss(self):
        model = SmallGPT(vocab_size=100, d_model=64, n_heads=4, n_layers=2)

        # Create dummy dataset
        data = list(range(200))
        dataset = TextDataset(data, block_size=5)
        loader = torch.utils.data.DataLoader(dataset, batch_size=4)

        device = 'cpu'
        loss = estimate_loss(model, loader, device, eval_iters=5)

        assert isinstance(loss.item(), float)
        assert loss.item() > 0