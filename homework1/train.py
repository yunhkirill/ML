import wandb
import torch
import torch.nn as nn
from tqdm import tqdm, trange
from torchvision.models import resnet18
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms

from hparams import config

def init_wandb():
    wandb.init(config=config, project="effdl_example", name="baseline")

def compute_accuracy(preds, targets):
    result = (targets == preds).float().mean()
    return result

def get_datasets():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ])
    
    train_dataset = CIFAR10(root='CIFAR10/train',
                          train=True,
                          transform=transform,
                          download=False)
    
    test_dataset = CIFAR10(root='CIFAR10/test',
                         train=False,
                         transform=transform,
                         download=False)
    return train_dataset, test_dataset

def get_model(device):
    model = resnet18(pretrained=False, num_classes=10, zero_init_residual=config["zero_init_residual"])
    model.to(device)
    return model

def train_one_batch(model, images, labels, criterion, optimizer, device):
    images = images.to(device)
    labels = labels.to(device)
    
    outputs = model(images)
    loss = criterion(outputs, labels)
    
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    return loss.item()

def evaluate(model, test_loader, device):
    all_preds = []
    all_labels = []
    
    for test_images, test_labels in test_loader:
        test_images = test_images.to(device)
        test_labels = test_labels.to(device)
        
        with torch.inference_mode():
            outputs = model(test_images)
            preds = torch.argmax(outputs, 1)
            
            all_preds.append(preds)
            all_labels.append(test_labels)
    
    accuracy = compute_accuracy(torch.cat(all_preds), torch.cat(all_labels))
    return accuracy.item()

def main():
    init_wandb()
    train_dataset, test_dataset = get_datasets()
    
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=config["batch_size"],
        shuffle=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=config["batch_size"]
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(device)
    wandb.watch(model)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config["learning_rate"], 
        weight_decay=config["weight_decay"]
    )
    
    for epoch in trange(config["epochs"]):
        for i, (images, labels) in enumerate(tqdm(train_loader)):
            loss = train_one_batch(model, images, labels, criterion, optimizer, device)
            
            if i % config.get("log_interval", 100) == 0:
                accuracy = evaluate(model, test_loader, device)
                metrics = {'test_acc': accuracy, 'train_loss': loss}
                wandb.log(metrics, step=epoch * len(train_dataset) + (i + 1) * config["batch_size"])
    
    torch.save(model.state_dict(), "model.pt")
    
    with open("run_id.txt", "w+") as f:
        print(wandb.run.id, file=f)

if __name__ == '__main__':
    main()