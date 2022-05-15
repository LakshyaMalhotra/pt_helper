import pytest
import torch
import torch.nn as nn
import numpy as np

from pthelper import trainer


class DummyDataset(torch.utils.data.Dataset):
    def __init__(self):
        super(DummyDataset, self).__init__()
        self.images = torch.rand(10, 28 * 28)
        self.labels = torch.rand(
            10,
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, ix):
        return self.images[ix], self.labels[ix]


class ToyModel:
    def __init__(self):
        self.bs = 2
        self.num_classes = 1
        self.model = nn.Sequential(
            nn.Linear(28 * 28, 16), nn.Linear(16, self.num_classes)
        )
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        self.dataloader = torch.utils.data.DataLoader(
            DummyDataset(), batch_size=self.bs
        )
        self.device = torch.device("cpu")


@pytest.fixture
def get_model_attrs():
    return ToyModel()


class TestPTHelper:
    def test_train(self, get_model_attrs, capsys):
        train_obj = get_model_attrs

        pt_trainer = trainer.PTHelper(
            train_obj.model,
            train_obj.device,
            train_obj.criterion,
            train_obj.optimizer,
            num_classes=train_obj.num_classes,
        )
        losses = pt_trainer.train(train_obj.dataloader, epoch=1, print_every=1)
        captured_sysout = capsys.readouterr()
        assert ("Epoch: [2][5 / 5]" in captured_sysout.out) and (
            isinstance(losses, float) and ~np.isnan(losses)
        )

    def test_evaluate(self, get_model_attrs, capsys):
        eval_obj = get_model_attrs
        pt_evaluator = trainer.PTHelper(
            eval_obj.model,
            eval_obj.device,
            eval_obj.criterion,
            eval_obj.optimizer,
            num_classes=eval_obj.num_classes,
        )
        losses = pt_evaluator.evaluate(eval_obj.dataloader, print_every=1)
        captured_sysout = capsys.readouterr()
        assert ("Evaluating: [5 / 5]" in captured_sysout.out) and (
            isinstance(losses, tuple) and ~np.isnan(losses[0])
        )
