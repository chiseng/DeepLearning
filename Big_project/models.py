from typing import Tuple

import torch

from utils import HyperParams


class ConvClassifier(torch.nn.Module):
    def __init__(self, num_in: int, hp: HyperParams):
        super().__init__()
        layers = [self.get_block(num_in, hp.num_hidden)]
        for _ in range(hp.num_layers - 1):
            layers.append(self.get_block(hp.num_hidden, hp.num_hidden))
        self.body = torch.nn.Sequential(*layers)
        self.classifier = torch.nn.Linear(hp.num_hidden, hp.num_ways)

    def forward(self, x):
        x = self.body(x)
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        return x

    @classmethod
    def get_block(cls, channels_in, channels_out, size_kernel=3):
        pad = cls.get_same_conv_pad(size_kernel)
        return torch.nn.Sequential(
            torch.nn.Conv2d(channels_in, channels_out, size_kernel, padding=pad),
            torch.nn.BatchNorm2d(channels_out),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
        )

    @staticmethod
    def get_same_conv_pad(k):
        """
        Assume stride and dilation are default
        s_out = s_in + 2p - k + 1
        Assume s_out == s_in
        p = (k - 1) / 2
        """
        assert (k - 1) % 2 == 0
        return (k - 1) // 2


class LinearLayers(torch.nn.Sequential):
    def __init__(self, num_in: int, hp: HyperParams):
        layers = [self.get_block(num_in, hp.num_hidden)]
        for _ in range(hp.num_layers - 1):
            layers.append(self.get_block(hp.num_hidden, hp.num_hidden))
        super().__init__(*layers)

    @staticmethod
    def get_block(num_in, num_out, do_relu=True) -> torch.nn.Sequential:
        layers = [torch.nn.Linear(num_in, num_out)]
        if do_relu:
            layers.append(torch.nn.ReLU())
        return torch.nn.Sequential(*layers)


class LinearClassifier(torch.nn.Module):
    def __init__(self, num_in: int, hp: HyperParams):
        super().__init__()
        self.body = LinearLayers(num_in, hp)
        self.head = torch.nn.Linear(hp.num_hidden, hp.num_ways)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.body(x)
        x = self.head(x)
        return x

class LSTMClassifier(torch.nn.Module):
    def __init__(self, num_in: int, hp:HyperParams):
        super(LSTMClassifier, self).__init__()
        self.inp = num_in
        self.hidden = hp.num_hidden
        self.lstm = torch.nn.LSTM(self.inp, self.hidden, 3, batch_first=True)
        self.out = torch.nn.Linear(self.hidden, hp.bs_inner)

    def forward(self, x): #, hidden_states):
        # h0,c0 = hidden_states
        # output, (hc,hn) = self.lstm(x, (h0,c0))
        output, (hn,cn) = self.lstm(x)
        # print(output[:,-1:].size())
        output = output[:,-1,:]
        output = self.out(output)
        return output

    # def initHidden(self, batch_size):
    #     # return h0, c0
    #     # print(self.num_layers*1, self.batch_size, self.hidden_size)
    #     self.batch_size = batch_size
    #     return (torch.zeros(3, self.batch_size, self.hidden, dtype=torch.float64),
    #             torch.zeros(3, self.batch_size, self.hidden, dtype=torch.float64))


class LinearClassifierWithOOS(LinearClassifier):
    def __init__(self, num_in: int, hp: HyperParams):
        super().__init__(num_in, hp)
        self.head_oos = torch.nn.Linear(hp.num_hidden, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_body = self.body(x)
        x = self.head(x_body)
        x_oos = self.head_oos(x_body)
        return x, x_oos


class LinearEmbedder(LinearClassifier):
    def __init__(self, num_in: int, hp: HyperParams):
        super().__init__(num_in, hp)
        self.head = torch.nn.Linear(hp.num_hidden, hp.num_hidden)
