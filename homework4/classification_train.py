import os

import wandb
import torch
import torchvision
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from model import Classifier
from utils import load_cifar10
from dataset import CustomDataset
from simclr_train import simclr_train
from hparams import simclr_hyp, cls_hyp


np.random.seed(42)


class ClassificationTrainer:
    def __init__(self, config, encoder, train_data, val_data, class_names, use_simclr=True):
        """Initialize the classification trainer."""
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.encoder = encoder.to(self.device)
        self.class_names = class_names
        self.use_simclr = use_simclr
        self.model = None
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.train_loader = None
        self.val_loader = None
        self.best_loss = float('inf')
        self.best_f1 = 0.0
        self.current_epoch = 0
        self._setup(train_data, val_data)

    def _setup_data(self, train_data, val_data):
        """Set up data loaders for training and validation."""
        X_train, y_train = train_data
        X_val, y_val = val_data
        train_dataset = CustomDataset(X_train, y_train, self.class_names)
        val_dataset = CustomDataset(X_val, y_val, self.class_names)
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=0,  # Windows CPU compatibility
            pin_memory=(self.device.type == 'cuda')
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=0,
            pin_memory=(self.device.type == 'cuda')
        )

    def _setup_model(self):
        """Initialize model, loss, optimizer, and scheduler."""
        self.model = Classifier(self.encoder, num_classes=len(self.class_names), input_shape=(3, 32, 32)).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        params = self.model.classifier.parameters() if self.use_simclr else self.model.parameters()
        self.optimizer = torch.optim.Adam(
            params,
            lr=self.config['lr'],
            weight_decay=self.config['weight_decay']
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            patience=3,
            factor=0.5
        )

    def _setup(self, train_data, val_data):
        """Perform all setup steps."""
        self._setup_data(train_data, val_data)
        self._setup_model()

    def _compute_f1(self, logits, labels):
        """Compute macro F1 score."""
        probs, preds = torch.softmax(logits, dim=1).topk(k=1)
        return f1_score(
            labels.cpu().numpy(),
            preds.cpu().numpy().squeeze(),
            average='macro'
        )

    def train_epoch(self):
        """Run one training epoch."""
        self.model.train()
        total_loss = 0.0
        total_f1 = 0.0
        pbar = tqdm(
            self.train_loader,
            desc=f'Train Epoch {self.current_epoch}/{self.config["epochs"]}'
        )
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                logits = self.model(images)
                loss = self.criterion(logits, labels)
                loss.backward()
                self.optimizer.step()
            batch_loss = loss.item()
            batch_f1 = self._compute_f1(logits, labels)
            total_loss += batch_loss
            total_f1 += batch_f1
            pbar.set_postfix(loss=batch_loss, f1=batch_f1)
        avg_loss = total_loss / len(self.train_loader)
        avg_f1 = total_f1 / len(self.train_loader)
        return avg_loss, avg_f1

    def validate_epoch(self):
        """Run one validation epoch."""
        self.model.eval()
        total_loss = 0.0
        total_f1 = 0.0
        pbar = tqdm(
            self.val_loader,
            desc=f'Valid Epoch {self.current_epoch}/{self.config["epochs"]}'
        )
        with torch.no_grad():
            for images, labels in pbar:
                images, labels = images.to(self.device), labels.to(self.device)
                logits = self.model(images)
                loss = self.criterion(logits, labels)
                batch_loss = loss.item()
                batch_f1 = self._compute_f1(logits, labels)
                total_loss += batch_loss
                total_f1 += batch_f1
                pbar.set_postfix(loss=batch_loss, f1=batch_f1)
        avg_loss = total_loss / len(self.val_loader)
        avg_f1 = total_f1 / len(self.val_loader)
        return avg_loss, avg_f1

    def save_checkpoint(self, val_loss, val_f1):
        """Save model checkpoints based on validation metrics."""
        os.makedirs("data", exist_ok=True)
        prefix = "" if self.use_simclr else "_no_simclr"
        if val_f1 >= self.best_f1:
            self.best_f1 = val_f1
            torch.save(self.model.state_dict(), f"data/cls_best_model{prefix}.pth")
            artifact = wandb.Artifact(f'cls_best_model{prefix}_epoch_{self.current_epoch}', type='model')
            artifact.add_file(f"data/cls_best_model{prefix}.pth")
            wandb.log_artifact(artifact)
        if val_loss <= self.best_loss:
            self.best_loss = val_loss
            torch.save(self.model.state_dict(), f"data/cls_final_model{prefix}.pth")
            artifact = wandb.Artifact(f'cls_final_model{prefix}_epoch_{self.current_epoch}', type='model')
            artifact.add_file(f"data/cls_final_model{prefix}.pth")
            wandb.log_artifact(artifact)

    def train(self):
        """Run the full training loop."""
        train_losses, train_f1s = [], []
        val_losses, val_f1s = [], []
        for epoch in range(self.config['epochs']):
            self.current_epoch = epoch + 1
            train_loss, train_f1 = self.train_epoch()
            val_loss, val_f1 = self.validate_epoch()
            train_losses.append(train_loss)
            train_f1s.append(train_f1)
            val_losses.append(val_loss)
            val_f1s.append(val_f1)
            self.scheduler.step(val_loss)
            self.save_checkpoint(val_loss, val_f1)
            prefix = "SimCLR" if self.use_simclr else "No SimCLR"
            print(f"{prefix} Epoch {self.current_epoch}/{self.config['epochs']}, "
                  f"Train Loss: {train_loss:.4f}, Train F1: {train_f1:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}")
            wandb.log({
                f'{prefix.lower().replace(" ", "_")}_epoch': self.current_epoch,
                f'{prefix.lower().replace(" ", "_")}_train_loss': train_loss,
                f'{prefix.lower().replace(" ", "_")}_train_f1': train_f1,
                f'{prefix.lower().replace(" ", "_")}_val_loss': val_loss,
                f'{prefix.lower().replace(" ", "_")}_val_f1': val_f1
            })
        return train_losses, train_f1s, val_losses, val_f1s


def plot_comparison(simclr_metrics, no_simclr_metrics):
    """Plot comparison of SimCLR vs No SimCLR."""
    _, _, simclr_val_losses, simclr_val_f1s = simclr_metrics
    _, _, no_simclr_val_losses, no_simclr_val_f1s = no_simclr_metrics

    epochs = range(1, len(simclr_val_losses) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, simclr_val_losses, label='SimCLR Val Loss', color='blue')
    plt.plot(epochs, no_simclr_val_losses, label='No SimCLR Val Loss', color='red')
    plt.title('Validation Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, simclr_val_f1s, label='SimCLR Val F1', color='blue')
    plt.plot(epochs, no_simclr_val_f1s, label='No SimCLR Val F1', color='red')
    plt.title('Validation F1 Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    save_path = "results/comparison_plot.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    wandb.log({"comparison_plot": wandb.Image(save_path)})
    plt.show()
    plt.close()


if __name__ == "__main__":
    wandb.init(
        project="mipt_ml_homework4",
        config={
            "simclr_hyp": simclr_hyp,
            "cls_hyp": cls_hyp,
            "architecture": "ResNet18+Classifier",
            "dataset": "CIFAR-10",
            "device": "cpu"
        }
    )

    X_train_full, y_train_full, X_test, y_test = load_cifar10()

    indices = np.random.permutation(len(X_test))
    train_size = 0.8
    split_idx = int(train_size * len(X_test))
    train_indices, val_indices = indices[:split_idx], indices[split_idx:]
    X_train = X_test[train_indices]
    y_train = y_test[train_indices]
    X_val = X_test[val_indices]
    y_val = y_test[val_indices]
    class_names = ['airplane', 'softmax', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    # Train without SimCLR
    encoder_no_simclr = torchvision.models.resnet18()
    encoder_no_simclr.fc = nn.Identity()
    trainer_no_simclr = ClassificationTrainer(
        config=cls_hyp,
        encoder=encoder_no_simclr,
        train_data=(X_train, y_train),
        val_data=(X_val, y_val),
        class_names=class_names,
        use_simclr=False
    )
    no_simclr_metrics = trainer_no_simclr.train()

    # Train with SimCLR
    model_path = "data/simclr_encoder.pth"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if os.path.exists(model_path):
        encoder = torchvision.models.resnet18()
        encoder.fc = nn.Identity()
        encoder.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    else:
        _, encoder = simclr_train(simclr_hyp, X_train_full)

    trainer_simclr = ClassificationTrainer(
        config=cls_hyp,
        encoder=encoder,
        train_data=(X_train, y_train),
        val_data=(X_val, y_val),
        class_names=class_names,
        use_simclr=True
    )
    simclr_metrics = trainer_simclr.train()
    

    plot_comparison(simclr_metrics, no_simclr_metrics)

    wandb.finish()