import os

import pickle
import pandas as pd
from tqdm.auto import tqdm
from torchtext.data import Field, Example, Dataset

from .utils import DEVICE, CACHE_FILE, DATA_DIR, EOS_TOKEN, BOS_TOKEN


def load_raw_data(filepath=os.path.join(DATA_DIR, 'news.csv')):
    return pd.read_csv(filepath, delimiter=',')


def save_processed_data(data, filepath=CACHE_FILE):
    os.makedirs(DATA_DIR, exist_ok=True)
    
    cache_data = {
        'train_examples': data['train_dataset'].examples,
        'test_examples': data['test_dataset'].examples,
        'fields': data['fields'],
        'vocab': data['word_field'].vocab
    }

    with open(filepath, 'wb') as f:
        pickle.dump(cache_data, f)


def load_processed_data(filepath=CACHE_FILE):
    with open(filepath, 'rb') as f:
        cache_data = pickle.load(f)
    
    train_dataset = Dataset(cache_data['train_examples'], cache_data['fields'])
    test_dataset = Dataset(cache_data['test_examples'], cache_data['fields'])
    word_field = cache_data['fields'][0][1]
    word_field.vocab = cache_data['vocab']

    return train_dataset, test_dataset, word_field


def process_data(data):
    word_field = Field(tokenize='moses', init_token=BOS_TOKEN, eos_token=EOS_TOKEN, lower=True)
    fields = [('source', word_field), ('target', word_field)]

    examples = []
    for _, row in tqdm(data.iterrows(), total=len(data)):
        examples.append(Example.fromlist(
            [word_field.preprocess(row.text), word_field.preprocess(row.title)],
            fields
        ))

    full_dataset = Dataset(examples, fields)
    train_dataset, test_dataset = full_dataset.split(split_ratio=0.85)
    word_field.build_vocab(train_dataset, min_freq=7)

    return {
        'train_dataset': train_dataset,
        'test_dataset': test_dataset,
        'fields': fields,
        'word_field': word_field
    }


def prepare_data(force_reprocess=False):
    if not force_reprocess and os.path.exists(CACHE_FILE):
        try:
            return load_processed_data()
        except Exception as e:
            print(f"Ошибка загрузки данных: {e}")

    raw_data = load_raw_data()
    processed_data = process_data(raw_data)
    save_processed_data(processed_data)
    
    return processed_data['train_dataset'], processed_data['test_dataset'], processed_data['word_field']