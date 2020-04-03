# define dataloader
import os
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sequence_gen import utils


from torch.nn.utils.rnn import pad_sequence
# from torch.autograd import Variable
import math


class nameLanguageDataset(Dataset):
    """ Name Language Dataset """

    def __init__(self, language_name_dict, dataset_type, train_ratio, transform=None):
        self.keys = language_name_dict.keys()
        self.names = []
        self.labels = []
        self.dataset_type = dataset_type
        self.train_ratio = train_ratio
        self.transform = transform
        self.utils = utils()

        for key in self.keys:
            for label in language_name_dict[key]:
                self.names.append(label)
                self.labels.append(key)

        z = list(zip(self.names, self.labels))
        random.shuffle(z)
        self.names[:], self.labels[:] = zip(*z)

        self.split_index = int(len(self.names) * train_ratio)
        self.test_split_index = int(len(self.names) * (1 + train_ratio) / 2)
        self.names_train, self.names_val, self.names_test = self.names[:self.split_index], self.names[
                                                                                           self.split_index:self.test_split_index], self.names[
                                                                                                                                    self.test_split_index:]
        self.labels_train, self.labels_val, self.labels_test = self.labels[:self.split_index], self.labels[
                                                                                               self.split_index:self.test_split_index], self.labels[
                                                                                                                                        self.test_split_index:]

    def __len__(self):
        if self.dataset_type == "train":
            return len(self.names_train)
        elif self.dataset_type == "test":
            return len(self.names_test)
        elif self.dataset_type == "val":
            return len(self.names_val)
        else:
            return None

    def __getitem__(self, idx):
        if self.dataset_type == "train":
            name = self.names_train[idx]
            label = self.labels_train[idx]
            sample = (label, name, self.utils.category_to_tensor(label), self.utils.lineToTensor(name))

            return sample

        elif self.dataset_type == "val":
            name = self.names_val[idx]
            label = self.labels_val[idx]
            sample = (label, name, self.utils.category_to_tensor(label), self.utils.lineToTensor(name))

            return sample

        elif self.dataset_type == "test":
            name = self.names_test[idx]
            label = self.labels_test[idx]

            sample = (label, name, self.utils.category_to_tensor(label), self.utils.lineToTensor(name))

            return sample




# create iterator to return batch size of tensors
class iteratefromDict():
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.count = 0

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.dataset) // self.batch_size + math.ceil(len(self.dataset) % self.batch_size)

    def __next__(self):
        if self.count == len(self.dataset):
            self.count = 0
            raise StopIteration()

        else:
            if len(self.dataset) - self.count < self.batch_size:
                self.batch_size = len(self.dataset) - self.count
            samples = [self.dataset[self.count + i][3] for i in range(0, self.batch_size)]
            labels = [self.dataset[self.count + i][2] for i in range(0, self.batch_size)]

            self.count += self.batch_size

            mini_batch_sample = pad_sequence(samples, padding_value=0).squeeze(2)
            mini_batch_label = torch.Tensor(labels).long()

            return mini_batch_sample, mini_batch_label