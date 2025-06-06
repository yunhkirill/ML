import os

import torch
import torchvision
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from model import SimCLR
from hparams import simclr_hyp
from dataset import SimCLRDataset
from utils import info_nce_loss, load_cifar10


np.random.seed(42)


class SimCLRTrainer:
    def __init__(self, config, dataset):
        """Initialize the SimCLR trainer."""
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.train_loader = None
        self._setup(dataset)

    def _setup_data(self, dataset):
        """Set up the data loader for SimCLR training."""
        simclr_dataset = SimCLRDataset(dataset)
        self.train_loader = DataLoader(
            simclr_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=0,  # Windows CPU compatibility
            pin_memory=(self.device.type == 'cuda')
        )

    def _setup_model(self):
        """Initialize the SimCLR model, optimizer, and scheduler."""
        encoder = torchvision.models.resnet18(weights=None)
        self.model = SimCLR(encoder, projection_dim=128).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config['lr'],
            weight_decay=self.config['weight_decay']
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config['epochs']
        )

    def _setup(self, dataset):
        """Perform all setup steps."""
        self._setup_data(dataset)
        self._setup_model()

    def train_epoch(self, epoch):
        """Run one training epoch."""
        self.model.train()
        total_loss = 0.0
        pbar = tqdm(
            self.train_loader,
            desc=f'SimCLR Epoch {epoch}/{self.config["epochs"]}'
        )
        for view1, view2 in pbar:
            view1, view2 = view1.to(self.device), view2.to(self.device)
            self.optimizer.zero_grad()
            _, z1 = self.model(view1)
            _, z2 = self.model(view2)
            features = torch.cat([z1, z2], dim=0)
            loss = info_nce_loss(
                features,
                batch_size=view1.size(0),
                temperature=self.config['temperature']
            )
            loss.backward()
            self.optimizer.step()
            batch_loss = loss.item()
            total_loss += batch_loss
            pbar.set_postfix(loss=batch_loss)
        self.scheduler.step()
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss

    def train(self):
        """Run the full SimCLR training loop."""
        losses = []
        for epoch in range(1, self.config['epochs'] + 1):
            loss = self.train_epoch(epoch)
            losses.append(loss)
            print(f"Epoch {epoch}/{self.config['epochs']}, Loss: {loss:.4f}")
        os.makedirs("data", exist_ok=True)
        torch.save(self.model.encoder.state_dict(), "data/simclr_encoder.pth")
        return losses, self.model.encoder


def simclr_train(config, dataset):
    """Train a SimCLR model and return losses and encoder."""
    trainer = SimCLRTrainer(config, dataset)
    losses, encoder = trainer.train()
    return losses, encoder


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_cifar10()
    losses, encoder = simclr_train(simclr_hyp, X_train)