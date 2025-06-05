import os
import math
import warnings

import wandb
import torch
from tqdm.auto import tqdm
from torchtext.data import BucketIterator

from model import EncoderDecoder, NoamOpt
from utils import LabelSmoothing, DEVICE, DATA_DIR, prepare_data, convert_batch


tqdm.get_lock().locks = []
warnings.filterwarnings("ignore", category=UserWarning, message="The given NumPy array is not writable")


def do_epoch(model, criterion, data_iter, optimizer=None, name=None):
    epoch_loss = 0
    is_train = optimizer is not None
    name = name or ''
    model.train(is_train)

    batches_count = len(data_iter)

    with torch.autograd.set_grad_enabled(is_train):
        with tqdm(total=batches_count) as progress_bar:
            for _, batch in enumerate(data_iter):
                source_inputs, target_inputs, source_mask, target_mask = convert_batch(batch)
                logits = model.forward(source_inputs, target_inputs[:, :-1], source_mask, target_mask[:, :-1, :-1])

                logits = logits.contiguous().view(-1, logits.shape[-1])
                target = target_inputs[:, 1:].contiguous().view(-1)
                loss = criterion(logits, target)

                epoch_loss += loss.item()

                if optimizer:
                    optimizer.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                loss_for_ppx = min(loss.item(), 700)
                progress_bar.update()
                progress_bar.set_description('{:>5s} Loss = {:.5f}, PPX = {:.2f}'.format(
                    name, loss.item(), math.exp(loss_for_ppx)))

                if is_train:
                    wandb.log({
                        "batch_loss": loss.item(),
                        "batch_ppx": math.exp(loss_for_ppx)
                    })

            avg_loss = epoch_loss / batches_count
            avg_loss_for_ppx = min(avg_loss, 700)
            progress_bar.set_description('{:>5s} Loss = {:.5f}, PPX = {:.2f}'.format(
                name, avg_loss, math.exp(avg_loss_for_ppx)))
            progress_bar.refresh()

            wandb.log({
                f"{name}_epoch_loss": avg_loss,
                f"{name}_epoch_ppx": math.exp(avg_loss_for_ppx)
            })

    return epoch_loss / batches_count


def fit(model, criterion, optimizer, train_iter, epochs_count=1):
    for epoch in range(epochs_count):
        name_prefix = '[{} / {}] '.format(epoch + 1, epochs_count)
        train_loss = do_epoch(model, criterion, train_iter, optimizer, name_prefix + 'Train:')
        print(train_loss)
        wandb.log({"epoch": epoch + 1})


if __name__ == "__main__":
    wandb.init(project="transformer-training", config={
        "epochs": 10,
        "batch_size_train": 16,
        "batch_size_test": 32,
        "smoothing": 0.1,
        "optimizer": "NoamOpt",
        "device": str(DEVICE)
    })

    train_dataset, test_dataset, word_field = prepare_data()

    train_iter, test_iter = BucketIterator.splits(
        datasets=(train_dataset, test_dataset),
        batch_sizes=(16, 32),
        shuffle=True,
        device=DEVICE,
        sort=False
    )

    model = EncoderDecoder(source_vocab_size=len(word_field.vocab), target_vocab_size=len(word_field.vocab)).to(DEVICE)

    pad_idx = word_field.vocab.stoi['<pad>']

    criterion = LabelSmoothing(len(word_field.vocab), pad_idx, 0.1).to(DEVICE)

    optimizer = NoamOpt(model)

    fit(model, criterion, optimizer, train_iter, epochs_count=10)

    os.makedirs(DATA_DIR, exist_ok=True)
    model_path = os.path.join(DATA_DIR, 'pre_transformer.pth')
    vocab_path = os.path.join(DATA_DIR, 'pre_vocabulary.pt')
    torch.save(model.state_dict(), model_path)
    torch.save(word_field.vocab, vocab_path)

    artifact = wandb.Artifact('transformer-model', type='model')
    artifact.add_file(model_path)
    artifact.add_file(vocab_path)
    wandb.log_artifact(artifact)

    wandb.finish()