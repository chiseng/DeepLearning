import csv
import torch.nn as nn
from io import open
import glob
import os
import torch
import numpy as np
import unicodedata
import dataloader
import tqdm
import string

import utils



class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)


class LSTM_batchy(nn.Module):
    def __init__(self, input_size, output_size, temperature=1, hidden_size=200):
        super(LSTM_batchy, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, 3)
        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=-1)
        self.temperature = temperature

    def forward(self, input, hidden_states):
        h0, c0 = hidden_states
        output, (hn, cn) = self.lstm(input, (h0, c0))
        output = self.out(output)
        output = self.softmax(torch.div(output, self.temperature))
        return output, (hn, cn)

    def initHidden(self, batch_size):
        # return h0, c0
        self.batch_size = batch_size
        return (
            torch.zeros(3 * 1, self.batch_size, self.hidden_size),
            torch.zeros(3 * 1, self.batch_size, self.hidden_size),
        )


# print(category_lines)


def train_lstm(
    criterion, model, device, optimizer, target_tensor, line_tensor, batch_size
):
    hidden, cell = model.initHidden(batch_size)
    model.train()
    model.zero_grad()
    hidden, cell = hidden.to(device), cell.to(device)
    output, (hidden, cell) = model(line_tensor, (hidden, cell))
    # print("out:", output.shape, "out squeeze: ", output.squeeze(1).shape, "category:", target_tensor.shape, "hidden:", hidden.shape, "cell state:", cell.shape)
    loss = criterion(output.squeeze(1), target_tensor)
    loss.backward()
    optimizer.step()

    return output, loss.item()


def train_phase(trainloader, device, model, criterion, optimizer):
    train_loss = []
    for iter, data in enumerate(tqdm.tqdm(trainloader)):
        line_tensor, target_tensor = data[0].to(device), data[1].to(device)
        output, loss = train_lstm(
            criterion,
            model,
            device,
            optimizer,
            target_tensor,
            line_tensor,
            data[0].shape[1],
        )
        train_loss.append(loss)
    return np.mean(train_loss)


def eval_model(
    model, device, criterion, batch_size, target_tensor :torch.Tensor, line_tensor
):
    model.eval()
    hidden, cell = model.initHidden(batch_size)
    hidden = hidden.to(device)
    cell = cell.to(device)
    with torch.no_grad():
        output, (hidden, cell) = model(line_tensor, (hidden, cell))
    # print(output.shape)
    # print(target_tensor.shape)
    loss = criterion(output.squeeze(-1), target_tensor)

    return output, loss.item()


def eval_phase(dataloader, device, model, criterion):
    best_weights = None
    best_loss = 1000
    for iter, data in enumerate(dataloader):
        line_tensor, target_tensor = data[0].to(device), data[1].to(device)
        output, loss = eval_model(
            model,
            device,
            criterion,
            data[0].shape[1],
            target_tensor,
            line_tensor,
        )
    return loss


def test_phase(util_class, dataloader, device, model, criterion):
    best_weights = None
    best_loss = 1000
    correct = 0
    test_loss = []
    total = 0
    for iter, data in enumerate(dataloader):
        line_tensor, target_tensor = data[0].to(device), data[1].to(device)
        total += data[0].shape[1]
        output, loss = eval_model(
            model,
            device,
            criterion,
            data[0].shape[1],
            target_tensor,
            line_tensor,
        )
        category = util_class.batch_categoryFromOutput(output)
        batch_correct = (torch.tensor(category) == target_tensor.cpu()).sum()
        correct += batch_correct
        test_loss.append(loss)
    acc = correct / float(total)
    return np.mean(test_loss), acc.data


def train_predict(util_class, model, device, dataset, batch_size, epochs):
    model.to(device)
    criterion = nn.NLLLoss()
    epochs_train_loss = []
    epochs_test_loss = []
    epochs_test_acc = []

    optimizer = torch.optim.SGD(model.parameters(), lr=0.005)
    name_lang_dataset_train = dataloader.nameLanguageDataset(dataset, "train", 0.8)
    name_lang_dataset_test = dataloader.nameLanguageDataset(dataset, "test", 0.8)
    trainloader = dataloader.iteratefromDict(name_lang_dataset_train, batch_size)
    testloader = dataloader.iteratefromDict(name_lang_dataset_test, batch_size)

    print("Train set: ", len(name_lang_dataset_train))
    print("Test set: ", len(name_lang_dataset_test))
    for step in range(epochs):
        print("\n")
        print("Epoch %i" % step)
        print("=" * 10)
        train_loss = []
        val_loss = []
        current_loss = 0
        correct = 0
        total = 0

        # train loop

        # loss_per_epoch = train_phase(
        #     trainloader, device, model, criterion, optimizer
        # )
        # epochs_train_loss.append(loss_per_epoch)
        # print("Current training loss: %f at epoch %i" % (loss_per_epoch, step))

        # validation phase
        test_loss, test_acc = test_phase(
            util_class, testloader, device, model, criterion
        )
        print(
            "Test loss: %f , test accuracy %f at epoch: %i"
            % (test_loss, test_acc, step)
        )

        epochs_test_loss.append(test_loss)
        epochs_test_acc.append(test_acc)

    return epochs_test_acc


def run():
    util_class = utils.utils()
    epochs = 8
    batch_size = 1
    lstm_batch = LSTM_batchy(util_class.n_letters, util_class.n_categories)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gepochs_test_acc = train_predict(
        util_class, lstm_batch, device, util_class.category_lines, batch_size, epochs
    )


if __name__ == '__main__':
    run()
