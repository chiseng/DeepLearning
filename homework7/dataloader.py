# define dataloader
import os
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import utils


from torch.nn.utils.rnn import pad_sequence

# from torch.autograd import Variable
import math


class nameLanguageDataset(Dataset):
    """ Name Language Dataset """

    def __init__(self, language_name_dict, dataset_type, train_ratio, transform=None):
        self.keys = language_name_dict.keys()
        self.dataset_type = dataset_type
        self.train_ratio = train_ratio
        self.transform = transform
        self.utils = utils.utils()

        self.lines = self.utils.category_lines["st"]
        random.shuffle(self.lines)

        self.split_index = int(len(self.lines) * train_ratio)
        self.test_split_index = int(len(self.lines) * (1 + train_ratio) / 2)

        self.train, self.val, self.test = (
            self.lines[: self.split_index],
            self.lines[self.split_index : self.test_split_index],
            self.lines[self.test_split_index :],
        )

    def __len__(self):
        if self.dataset_type == "train":
            return len(self.train)
        elif self.dataset_type == "test":
            return len(self.test)
        elif self.dataset_type == "val":
            return len(self.val)
        else:
            return None

    def __getitem__(self, idx):
        if self.dataset_type == "train":
            line = self.train[idx]
            sample = (line, self.utils.inputTensor(line), self.utils.targetTensor(line))
            return sample

        elif self.dataset_type == "val":
            line = self.val[idx]
            sample = (
                line,
                self.utils.inputTensor(line),
                self.utils.targetTensor(line),
            )

            return sample

        elif self.dataset_type == "test":
            line = self.test[idx]

            sample = (
                line,
                self.utils.inputTensor(line),
                self.utils.targetTensor(line),
            )

            return sample


# create iterator to return batch size of tensors
class iteratefromDict:
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.initial_b = batch_size
        self.count = 0

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.dataset) // self.batch_size + math.ceil(
            len(self.dataset) % self.batch_size
        )

    def __next__(self):
        self.batch_size = self.initial_b
        if self.count == len(self.dataset):
            self.count = 0
            raise StopIteration()

        else:
            if len(self.dataset) - self.count < self.batch_size:
                self.batch_size = len(self.dataset) - self.count
            input = [self.dataset[self.count + i][1] for i in range(0, self.batch_size)]
            target = [
                self.dataset[self.count + i][2] for i in range(0, self.batch_size)
            ]

            self.count += self.batch_size

            mini_batch_input = pad_sequence(input, padding_value=0).squeeze(2)
            mini_batch_target = pad_sequence(target, padding_value=0)

            return mini_batch_input, mini_batch_target
