import os
import wandb
import torch
import numpy as np
from tqdm import tqdm
import torch.optim as optim

from model import load_protonet_conv
from utils import extract_sample, read_images


def train(model, optimizer, train_x, train_y, n_way, n_support, n_query, max_epoch, epoch_size, data_dir='./data'):
    """
    Trains the protonet with Weights & Biases logging
    Args:
      model: ProtoNet model
      optimizer: optimizer for training
      train_x (np.array): images of training set
      train_y(np.array): labels of training set
      n_way (int): number of classes in a classification task
      n_support (int): number of labeled examples per class in the support set
      n_query (int): number of labeled examples per class in the query set
      max_epoch (int): max epochs to train on
      epoch_size (int): episodes per epoch
      data_dir (str): directory to save best_model.pth and final_model.pth
    """
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    wandb.init(project="protonet_training", config={
        "n_way": n_way,
        "n_support": n_support,
        "n_query": n_query,
        "max_epoch": max_epoch,
        "epoch_size": epoch_size,
        "learning_rate": optimizer.param_groups[0]['lr']
    })

    unique_classes = np.unique(train_y)
    num_classes = len(unique_classes)

    if n_way > num_classes:
        print(f"Warning: n_way ({n_way}) is larger than the number of available classes ({num_classes})")
        print(f"Reducing n_way to {num_classes}")
        n_way = num_classes

    min_examples = min([np.sum(train_y == cls) for cls in unique_classes])
    if min_examples < (n_support + n_query):
        print(f"Warning: Some classes have fewer than {n_support + n_query} examples")
        print(f"The minimum number of examples for any class is {min_examples}")
        print(f"Reducing n_support + n_query to {min_examples}")
        ratio = n_support / (n_support + n_query)
        n_support = int(min_examples * ratio)
        n_query = min_examples - n_support
        print(f"New values: n_support = {n_support}, n_query = {n_query}")

    scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.5, last_epoch=-1)
    epoch = 0  # epochs done so far
    stop = False  # status to know when to stop

    best_acc = 0.0

    while epoch < max_epoch and not stop:
        running_loss = 0.0
        running_acc = 0.0

        progress_bar = tqdm(range(epoch_size), desc=f"Epoch {epoch + 1}/{max_epoch}")

        for episode in progress_bar:
            sample = extract_sample(n_way, n_support, n_query, train_x, train_y)

            optimizer.zero_grad()
            loss, output = model.set_forward_loss(sample)

            batch_loss = output['loss']
            batch_acc = output['acc']

            running_loss += batch_loss
            running_acc += batch_acc

            global_step = epoch * epoch_size + episode
            wandb.log({
                'Batch/Loss': batch_loss,
                'Batch/Accuracy': batch_acc
            }, step=global_step)

            progress_bar.set_postfix(loss=batch_loss, acc=batch_acc)

            loss.backward()
            optimizer.step()

        epoch_loss = running_loss / epoch_size
        epoch_acc = running_acc / epoch_size

        wandb.log({
            'Epoch/Loss': epoch_loss,
            'Epoch/Accuracy': epoch_acc,
            'Epoch/LearningRate': optimizer.param_groups[0]['lr']
        }, step=epoch)

        print(f'Epoch {epoch+1}/{max_epoch} -- Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        if epoch_acc > best_acc:
            best_acc = epoch_acc
            torch.save(model.state_dict(), os.path.join(data_dir, 'protonet_best_model.pth'))
            wandb.log({'Best Model': f'New best model at epoch {epoch+1} with accuracy {best_acc:.4f}'}, step=epoch)

        epoch += 1
        scheduler.step()

        wandb.log({'Optimizer/LearningRate': scheduler.get_last_lr()[0]}, step=epoch)

    wandb.finish()

    torch.save(model.state_dict(), os.path.join(data_dir, 'protonet_final_model.pth'))

    return model


if __name__ == "__main__":
    model = load_protonet_conv(
        x_dim=(3, 28, 28),
        hid_dim=64,
        z_dim=64,
    )

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    n_way = 60
    n_support = 5
    n_query = 5
    train_x, train_y = read_images('data/images_background')

    max_epoch = 5
    epoch_size = 2000

    train(model, optimizer, train_x, train_y, n_way, n_support, n_query, max_epoch, epoch_size)