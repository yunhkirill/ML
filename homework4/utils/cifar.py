import os

import pickle
import numpy as np


def load_cifar10_batch(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict[b'data'], dict[b'labels']


def load_cifar10(cifar10_dir='data/cifar-10-batches-py'):
    x_train = []
    y_train = []
    for i in range(1, 6):
        batch_file = os.path.join(cifar10_dir, f'data_batch_{i}')
        data, labels = load_cifar10_batch(batch_file)
        x_train.append(data)
        y_train.extend(labels)

    x_train = np.vstack(x_train).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    y_train = np.array(y_train)

    test_file = os.path.join(cifar10_dir, 'test_batch')
    x_test, y_test = load_cifar10_batch(test_file)
    x_test = x_test.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    y_test = np.array(y_test)

    return x_train, y_train, x_test, y_test