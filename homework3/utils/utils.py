import os

import torch
import numpy as np


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_DIR = "data"
CACHE_FILE = os.path.join(DATA_DIR, "processed_data.pkl")
BOS_TOKEN = '<s>'
EOS_TOKEN = '</s>'
np.random.seed(42)


def subsequent_mask(size):
    mask = torch.ones(size, size, device=DEVICE).triu_()
    return mask.unsqueeze(0) == 0
    

def make_mask(source_inputs, target_inputs, pad_idx):
    source_mask = (source_inputs != pad_idx).unsqueeze(-2)
    target_mask = (target_inputs != pad_idx).unsqueeze(-2)
    target_mask = target_mask & subsequent_mask(target_inputs.size(-1)).type_as(target_mask)
    return source_mask, target_mask


def convert_batch(batch, pad_idx=1):
    source_inputs, target_inputs = batch.source.transpose(0, 1), batch.target.transpose(0, 1)
    source_mask, target_mask = make_mask(source_inputs, target_inputs, pad_idx)
    return source_inputs, target_inputs, source_mask, target_mask


def text_to_indices(tokens: list, vocab: dict, max_len: int = 256, bos_token: str = BOS_TOKEN, 
                    eos_token: str = EOS_TOKEN, unk_token: str = '<unk>') -> torch.Tensor:
    bos_id = vocab.stoi.get(bos_token, 1)
    eos_id = vocab.stoi.get(eos_token, 2)
    unk_id = vocab.stoi.get(unk_token, 0)

    tokens = tokens[:max_len - 2]
    indices = [bos_id] + [vocab.stoi.get(token, unk_id) for token in tokens] + [eos_id]
    
    if len(indices) < max_len:
        indices += [vocab.stoi.get('<pad>', 0)] * (max_len - len(indices))
    elif len(indices) > max_len:
        indices = indices[:max_len]
    
    return torch.tensor(indices, dtype=torch.long).unsqueeze(0)


def indices_to_text(indices, vocab, skip_special_tokens=True, special_tokens=None):
    if special_tokens is None:
        special_tokens = {'<pad>', BOS_TOKEN, EOS_TOKEN, '<unk>'}

    tokens = []
    for idx in indices:
        token = vocab.itos[idx.item()]
        if skip_special_tokens and token in special_tokens:
            continue
        tokens.append(token)
    
    return ' '.join(tokens)


def save_to_file(filename: str, content: str, mode: str = 'a') -> None:
    with open(filename, mode, encoding='utf-8') as f:
        f.write(content)


def ensure_dir_exists(directory: str) -> None:
    if not os.path.exists(directory):
        os.makedirs(directory)
