import copy
from typing import Dict

import torch
import torch.utils.data
import torchmeta
import tqdm

import datasets
import models
import utils


class ReptileSGD:
    """
    Apply Reptile gradient update to weights, average gradient across meta batches
    The Reptile gradient is quite simple, please refer to "get_gradients"
    The model weights are using SGD in "step"
    """

    def __init__(self, net: torch.nn.Module, lr_schedule, num_accum):
        self.net = net
        self.num_accum = num_accum
        self.lr_schedule: utils.LinearDecayLR = lr_schedule

        self.grads = utils.MathDict()
        self.weights_before = utils.MathDict()
        self.counter = 0
        self.zero_grad()

    def zero_grad(self):
        self.grads = utils.MathDict()
        self.weights_before = utils.MathDict(self.net.state_dict())
        self.counter = 0

    def get_gradients(self):
        weights_after = utils.MathDict(self.net.state_dict())
        return self.weights_before - weights_after

    def store_grad(self):
        g = self.get_gradients()
        assert self.counter < self.num_accum
        if self.counter == 0:
            self.grads = g
        else:
            self.grads = self.grads + g
        self.counter += 1

    def step(self, i_step):
        assert self.counter == self.num_accum
        grads_avg = self.grads / self.num_accum
        lr = self.lr_schedule.get_lr(i_step)
        weights_new = self.weights_before - (grads_avg * lr)
        self.net.load_state_dict(weights_new.state)


class LoopParams:
    """Hyperparameters for meta learning loops"""

    def __init__(self, lr, steps, bs):
        self.bs = bs
        self.steps = steps
        self.lr = lr


class ReptileSystem:
    """
    The Reptile meta-learning algorithm was invented by OpenAI
    https://openai.com/blog/reptile/

    System to take in meta-learning data and any kind of model
    and run Reptile meta-learning training loop
    The objective of meta-learning is not to master any single task
    but instead to obtain a system that can quickly adapt to a new task
    using a small number of training steps and data, like a human

    Reptile pseudo-code:
    Initialize initial weights, w
    For iterations:
        Randomly sample a task T
        Perform k steps of SGD on T, now having weights, w_after
        Update w = w - learn_rate * (w - w_after)
    """

    def __init__(
        self,
        loaders: Dict[str, torchmeta.utils.data.BatchMetaDataLoader],
        net: torch.nn.Module,
        inner: LoopParams,
        outer: LoopParams,
        val: LoopParams,
    ):
        super().__init__()
        self.loaders = loaders
        self.inner = inner
        self.outer = outer
        self.val = val

        self.device = utils.get_device()
        self.rng = utils.set_random_state()
        self.net = net.to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.opt_inner = torch.optim.SGD(self.net.parameters(), lr=self.inner.lr)
        lr_schedule = utils.LinearDecayLR(self.outer.lr, self.outer.steps)
        self.opt_outer = ReptileSGD(self.net, lr_schedule, num_accum=self.outer.bs)

        self.batch_val = next(iter(self.loaders[datasets.Splits.val]))

    def run_batch(self, batch, do_train=True):
        if do_train:
            self.net.train()
            context_grad = torch.enable_grad
        else:
            self.net.eval()
            context_grad = torch.no_grad

        with context_grad():
            inputs, targets = batch
            if do_train:
                self.opt_inner.zero_grad()
            outputs = self.net(inputs)
            loss = self.criterion(outputs, targets)
            acc = utils.acc_score(outputs, targets)
            if do_train:
                loss.backward()
                self.opt_inner.step()
        return dict(loss=loss.item(), acc=acc.item())

    def loop_inner(self, task, bs, steps):
        x_train, y_train, x_test, y_test = task
        ds = torch.utils.data.TensorDataset(x_train, y_train)
        loader = torch.utils.data.DataLoader(ds, bs, shuffle=True)

        counter = 0
        while counter < steps:
            for batch in loader:
                self.run_batch(batch, do_train=True)
                counter += 1

        metrics = self.run_batch((x_test, y_test), do_train=False)
        return metrics

    def load_meta_tasks(self, batch):
        meta_x_train, meta_y_train = batch["train"]
        meta_x_test, meta_y_test = batch["test"]
        tensors = [meta_x_train, meta_y_train, meta_x_test, meta_y_test]
        tensors = [t.to(self.device) for t in tensors]
        return [[t[i] for t in tensors] for i in range(self.outer.bs)]

    def loop_outer(self):
        steps = self.outer.steps
        interval = steps // 100
        loader = self.loaders[datasets.Splits.train]
        tracker = utils.MetricsTracker(prefix=datasets.Splits.train)

        with tqdm.tqdm(loader, total=self.outer.steps) as pbar:
            for i, batch in enumerate(pbar):
                if i > steps:
                    break

                self.opt_outer.zero_grad()
                for task in datasets.MetaBatch(batch, self.device).get_tasks():
                    metrics = self.loop_inner(task, self.inner.bs, self.inner.steps)
                    tracker.store(metrics)
                    self.opt_outer.store_grad()
                self.opt_outer.step(i)

                if i % interval == 0:
                    metrics = tracker.average()
                    tracker.reset()
                    metrics.update(self.loop_val())
                    pbar.set_postfix(metrics)

    def loop_val(self):
        tracker = utils.MetricsTracker(prefix=datasets.Splits.val)
        weights_before = copy.deepcopy(self.net.state_dict())

        for task in datasets.MetaBatch(self.batch_val, self.device).get_tasks():
            self.net.load_state_dict(weights_before)
            metrics = self.loop_inner(task, self.val.bs, self.val.steps)
            tracker.store(metrics)
        self.net.load_state_dict(weights_before)
        return tracker.average()

    def run_train(self):
        self.loop_outer()


def run_omniglot(root):
    loaders = {}
    for s in [datasets.Splits.train, datasets.Splits.val]:
        params_data = datasets.DataParams(
            root, data_split=s, bs=5, num_ways=5, num_shots=5, num_shots_test=15
        )
        loaders[s] = datasets.OmniglotMetaLoader(params_data)

    params_inner = LoopParams(lr=1e-3, steps=5, bs=10)
    params_outer = LoopParams(lr=1.0, steps=1000, bs=params_data.bs)
    params_val = LoopParams(lr=1e-3, steps=50, bs=5)
    net = models.ConvClassifier(size_in=1, size_out=params_data.num_ways)
    system = ReptileSystem(loaders, net, params_inner, params_outer, params_val)
    system.run_train()


def run_intent(root):
    loaders = {}
    for s in [datasets.Splits.train, datasets.Splits.val]:
        params_data = datasets.DataParams(
            root, data_split=s, bs=5, num_ways=5, num_shots=5, num_shots_test=15
        )
        loaders[s] = datasets.IntentMetaLoader(params_data)

    params_inner = LoopParams(lr=1e-3, steps=50, bs=5)
    params_outer = LoopParams(lr=1.0, steps=500, bs=params_data.bs)
    params_val = LoopParams(lr=1e-3, steps=50, bs=5)
    net = models.LinearClassifier(
        size_in=loaders[s].embedder.size_embed, size_out=params_data.num_ways
    )
    system = ReptileSystem(loaders, net, params_inner, params_outer, params_val)
    system.run_train()


def main(root="temp"):
    run_intent(root)


if __name__ == "__main__":
    main()
