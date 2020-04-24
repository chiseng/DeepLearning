import torch

from datasets import IntentEmbedBertMetaLoader, Splits, MetaBatch, MetaLoader
from reptile import ReptileSystem
from utils import MetricsTracker, HyperParams
import sklearn.ensemble

class BaselineSystem:
    def __init__(self, hparams: HyperParams, loader: MetaLoader, model):
        self.hparams = hparams
        self.batch_val = next(iter(loader))
        self.model = model

    def loop_val(self) -> dict:
        tracker = MetricsTracker(prefix=Splits.val)
        for task in MetaBatch(self.batch_val, torch.device("cpu")).get_tasks():
            x_train, y_train, x_test, y_test = task
            model = self.model
            model.fit(x_train.numpy(), y_train.numpy())

            logits_numpy = model.predict_proba(x_test.numpy())

            logits = torch.from_numpy(logits_numpy)
            acc = ReptileSystem.get_accuracy(logits, y_test)
            tracker.store(dict(acc=acc))
        return tracker.get_average()

    def run_train(self):
        print(self.loop_val())


def run_intent(root: str, model):
    hparams = HyperParams(root=root)
    loader = IntentEmbedBertMetaLoader(hparams, Splits.val)
    system = BaselineSystem(hparams, loader, model)
    system.run_train()


def main(root="temp"):
    rf = sklearn.ensemble.RandomForestClassifier()
    run_intent(root, rf)


if __name__ == "__main__":
    main()