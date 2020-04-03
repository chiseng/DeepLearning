import csv
import torch.nn as nn
from io import open
import glob
import os
import torch
import numpy as np
import unicodedata
import dataloader
from tqdm import tqdm
import string


class utils:
    def __init__(self):
        self.path = "star_trek_transcripts_all_episodes_f.csv"
        self.all_letters = string.ascii_letters + "0123456789 .,:!?'[]()/+-="
        self.n_letters = len(self.all_letters)
        self.category_lines = {}
        self.all_categories = ['st']
        self.category_lines["st"] = []
        with open(self.path, newline="") as csvfile:
            reader = csv.reader(csvfile, delimiter=",", quotechar='"')
            for row in reader:
                # print(row)
                for el in row:
                    v = el.strip().replace(";", "").replace('"', "")
                    self.category_lines["st"].append(v)
        self.n_categories = len(self.all_categories)

    # Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427
    def unicodeToAscii(self, s):
        return "".join(
            c
            for c in unicodedata.normalize("NFD", s)
            if unicodedata.category(c) != "Mn" and c in self.all_letters
        )

    # Build the category_lines dictionary, a list of names per language

    # Read a file and split into lines
    def readLines(self, filename):
        lines = open(filename, encoding="utf-8").read().strip().split("\n")
        return [self.unicodeToAscii(line) for line in lines]

    # Find letter index from all_letters, e.g. "a" = 0
    def letterToIndex(self, letter):
        return self.all_letters.find(letter)

    # Just for demonstration, turn a letter into a <1 x n_letters> Tensor
    def letterToTensor(self, letter):
        tensor = torch.zeros(1, self.n_letters)
        tensor[0][self.letterToIndex(letter)] = 1
        return tensor

    # Turn a line into a <line_length x 1 x n_letters>,
    # or an array of one-hot letter vectors
    def lineToTensor(self, line):
        tensor = torch.zeros(len(line), 1, self.n_letters)
        for li, letter in enumerate(line):
            tensor[li][0][self.letterToIndex(letter)] = 1
        return tensor

    def category_to_tensor(self, category):
        return torch.tensor([self.all_categories.index(category)], dtype=torch.long)

    def batch_categoryFromOutput(self, output):
        # print(output)
        output_tensor = torch.unbind(output, dim=0)
        ret_val = []
        for outp in output_tensor:
            top_n, top_i = outp.topk(1, 0)
            category_i = top_i[0].item()
            ret_val.append(category_i)
        return ret_val


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
    criterion, model, device, optimizer, category_tensor, line_tensor, batch_size
):
    hidden, cell = model.initHidden(batch_size)
    model.train()
    model.zero_grad()
    hidden, cell = hidden.to(device), cell.to(device)
    output, (hidden, cell) = model(line_tensor, (hidden, cell))
    output = output[-1]
    print(output)
    # print("out:", output.shape, "out squeeze: ", output.squeeze(1).shape, "category:", category_tensor.shape, "hidden:", hidden.shape, "cell state:", cell.shape)
    loss = criterion(output.squeeze(1), category_tensor)
    loss.backward()

    # Add parameters' gradients to their values, multiplied by learning rate
    # for p in model.parameters():
    #     p.data.add_(-learning_rate, p.grad.data)
    optimizer.step()

    return output, loss.item()


def train_phase(trainloader, device, model, criterion, optimizer):
    train_loss = []
    for iter, data in enumerate(trainloader):
        line_tensor, category_tensor = data[0].to(device), data[1].to(device)
        output, loss = train_lstm(
            criterion,
            model,
            device,
            optimizer,
            category_tensor,
            line_tensor,
            data[0].shape[1],
        )
        train_loss.append(loss)
    return np.mean(train_loss)


def eval_model(
    model, model_type, device, criterion, batch_size, category_tensor, line_tensor
):
    model.eval()
    if model_type == "gru":
        hidden = model.initHidden(batch_size)
        hidden = hidden.to(device)
        with torch.no_grad():
            output, hidden = model(line_tensor, hidden)

    else:
        hidden, cell = model.initHidden(batch_size)
        hidden = hidden.to(device)
        cell = cell.to(device)
        with torch.no_grad():
            output, (hidden, cell) = model(line_tensor, (hidden, cell))

    output = output[-1]
    loss = criterion(output.squeeze(1), category_tensor)

    return output, loss.item()


def eval_phase(dataloader, device, model, model_type, criterion):
    best_weights = None
    best_loss = 1000
    for iter, data in enumerate(dataloader):
        line_tensor, category_tensor = data[0].to(device), data[1].to(device)
        output, loss = eval_model(
            model,
            model_type,
            device,
            criterion,
            data[0].shape[1],
            category_tensor,
            line_tensor,
        )
    return loss


def test_phase(util_class, dataloader, device, model, model_type, criterion):
    best_weights = None
    best_loss = 1000
    correct = 0
    test_loss = []
    total = 0
    for iter, data in enumerate(dataloader):
        line_tensor, category_tensor = data[0].to(device), data[1].to(device)
        total += data[0].shape[1]
        output, loss = eval_model(
            model,
            model_type,
            device,
            criterion,
            data[0].shape[1],
            category_tensor,
            line_tensor,
        )
        category = util_class.batch_categoryFromOutput(output)
        batch_correct = (torch.tensor(category) == category_tensor.cpu()).sum()
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

        loss_per_epoch = train_phase(
            trainloader, device, model, criterion, optimizer
        )
        epochs_train_loss.append(loss_per_epoch)
        print("Current training loss: %f at epoch %i" % (loss_per_epoch, step))

        # validation phase

        loss = eval_phase(testloader, device, model, criterion)
        print("Current val loss: %f at epoch: %i" % (loss, step))

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
    util_class = utils()
    epochs = 8
    batch_size = 10
    lstm_batch = LSTM_batchy(util_class.n_letters, util_class.n_categories)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gepochs_test_acc = train_predict(
        util_class, lstm_batch, device, util_class.category_lines, 10, epochs
    )


if __name__ == '__main__':
    run()
