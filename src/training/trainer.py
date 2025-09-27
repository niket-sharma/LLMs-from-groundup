import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import os
import time
import math
from tqdm import tqdm


class GPTTrainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        lr=3e-4,
        weight_decay=0.1,
        betas=(0.9, 0.95),
        device='cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=betas
        )

        # Learning rate scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=len(train_loader) * 10,  # 10 epochs by default
            eta_min=lr * 0.1
        )

        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc="Training")
        for batch_idx, (data, targets) in enumerate(pbar):
            data, targets = data.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            logits, loss = self.model(data, targets)

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()
            num_batches += 1

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss/num_batches:.4f}',
                'lr': f'{self.scheduler.get_last_lr()[0]:.2e}'
            })

        return total_loss / num_batches

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_loss = 0
        num_batches = 0

        for data, targets in tqdm(self.val_loader, desc="Validation"):
            data, targets = data.to(self.device), targets.to(self.device)

            logits, loss = self.model(data, targets)

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches

    def train(self, epochs, save_dir=None, save_every=5):
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            print("-" * 50)

            # Train
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)

            # Validate
            val_loss = self.validate()
            self.val_losses.append(val_loss)

            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                if save_dir:
                    self.save_checkpoint(os.path.join(save_dir, 'best_model.pt'))
                    print(f"New best model saved with val loss: {val_loss:.4f}")

            # Save checkpoint every save_every epochs
            if save_dir and (epoch + 1) % save_every == 0:
                self.save_checkpoint(
                    os.path.join(save_dir, f'checkpoint_epoch_{epoch + 1}.pt')
                )

        return self.train_losses, self.val_losses

    def save_checkpoint(self, path):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_loss = checkpoint['best_val_loss']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']


def estimate_loss(model, data_loader, device, eval_iters=200):
    model.eval()
    losses = torch.zeros(eval_iters)

    for k in range(eval_iters):
        try:
            X, Y = next(iter(data_loader))
            X, Y = X.to(device), Y.to(device)
            with torch.no_grad():
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        except StopIteration:
            break

    return losses.mean()