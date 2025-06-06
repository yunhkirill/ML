import os

import torch
import pytest
import tempfile
from unittest.mock import patch

import train


HYPERPARAM_SETS = [
    {"batch_size": 64, "learning_rate": 0.001, "weight_decay": 0.01, "epochs": 1, "zero_init_residual": False},
    {"batch_size": 32, "learning_rate": 0.005, "weight_decay": 0.0, "epochs": 1, "zero_init_residual": False},
    {"batch_size": 128, "learning_rate": 0.01, "weight_decay": 0.001, "epochs": 2, "zero_init_residual": False},
    {"batch_size": 256, "learning_rate": 0.0001, "weight_decay": 0.1, "epochs": 1, "zero_init_residual": False}
]


@pytest.fixture(autouse=True)
def set_random_seed():
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

@pytest.fixture(scope="module")
def real_datasets():
    train_dataset, test_dataset = train.get_datasets()
    return train_dataset, test_dataset

def override_config(new_params):
    original = train.config.copy()
    train.config.update(new_params)
    return original

def test_compute_accuracy():
    preds = torch.tensor([0, 1, 2, 3, 4])
    targets = torch.tensor([0, 1, 2, 3, 3])
    accuracy = train.compute_accuracy(preds, targets)
    assert accuracy == 0.8

def test_train_dataset(real_datasets):
    train_dataset, _ = real_datasets
    assert len(train_dataset) == 50000
    img, label = train_dataset[0]
    assert isinstance(img, torch.Tensor)
    assert img.shape == (3, 32, 32)
    assert isinstance(label, int)
    assert 0 <= label <= 9

@pytest.mark.parametrize("zero_init", [True, False])
def test_get_model(zero_init):
    original = override_config({"zero_init_residual": zero_init})
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = train.get_model(device)
        assert isinstance(model, torch.nn.Module)
        assert model.fc.out_features == 10
    finally:
        train.config.update(original)

@pytest.mark.parametrize("hyperparams", HYPERPARAM_SETS)
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_train_one_batch(hyperparams, device, real_datasets):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    train_dataset, _ = real_datasets
    original = override_config(hyperparams)
    
    try:
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=train.config["batch_size"],
            shuffle=True
        )
        
        device = torch.device(device)
        model = train.get_model(device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=train.config["learning_rate"],
            weight_decay=train.config["weight_decay"]
        )
        
        images, labels = next(iter(train_loader))
        loss = train.train_one_batch(model, images, labels, criterion, optimizer, device)
        
        assert isinstance(loss, float)
        assert not torch.isnan(torch.tensor(loss))
    finally:
        train.config.update(original)

@pytest.mark.parametrize("hyperparams", HYPERPARAM_SETS)
def test_evaluate(hyperparams, real_datasets):
    _, test_dataset = real_datasets
    original = override_config(hyperparams)
    
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = train.get_model(device)
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=train.config["batch_size"]
        )
        
        accuracy = train.evaluate(model, test_loader, device)
        assert 0 <= accuracy <= 1
    finally:
        train.config.update(original)

@pytest.mark.parametrize("hyperparams", HYPERPARAM_SETS[:2])
def test_training(hyperparams):
    original = override_config({**hyperparams, "log_interval": 20})
    
    if not hasattr(test_training, "results"):
        test_training.results = {}
    
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            train.main()
            
            assert os.path.exists("model.pt")
            assert os.path.exists("run_id.txt")
            assert os.path.getsize("model.pt") > 0
            assert os.path.getsize("run_id.txt") > 0
            
            _, test_dataset = train.get_datasets()
            test_loader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=hyperparams["batch_size"]
            )
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = train.get_model(device)
            model.load_state_dict(torch.load("model.pt"))
            
            accuracy = train.evaluate(model, test_loader, device)
            assert 0 <= accuracy <= 1
            
            param_key = tuple(sorted(hyperparams.items()))
            test_training.results[param_key] = accuracy
            
        finally:
            train.config.update(original)
            for f in ["model.pt", "run_id.txt"]:
                if os.path.exists(f):
                    os.remove(f)

def test_hyperparams_differences():
    if not hasattr(test_training, "results"):
        pytest.skip("No training results available for comparison")
    
    results = test_training.results
    assert len(results) >= 2
    
    has_significant_diff = False
    param_pairs = []
    
    for i, (params1, acc1) in enumerate(results.items()):
        for j, (params2, acc2) in enumerate(results.items()):
            if i >= j:
                continue
            
            diff = abs(acc1 - acc2)
            param_pairs.append((params1, params2, acc1, acc2, diff))
            
            if diff > 0.02:
                has_significant_diff = True
    
    assert has_significant_diff

def test_wandb_init():
    with patch('train.wandb.init') as mock_init:
        train.init_wandb()
        mock_init.assert_called_once_with(
            config=train.config,
            project="effdl_example",
            name="baseline"
        )

def test_model_saving():
    device = torch.device("cpu")
    model = train.get_model(device)
    
    test_path = "test_model.pt"
    try:
        torch.save(model.state_dict(), test_path)
        assert os.path.exists(test_path)
    finally:
        if os.path.exists(test_path):
            os.remove(test_path)