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
        self.train_dataloader = torch.utils.data.DataLoader(
            DummyDataset(), batch_size=self.bs
        )
        self.device = torch.device("cpu")


@pytest.fixture
def get_model_attrs():
    return ToyModel()


def test_train(get_model_attrs, capsys):
    train_obj = get_model_attrs

    pt_trainer = trainer.PTHelper(
        train_obj.model,
        train_obj.device,
        train_obj.criterion,
        train_obj.optimizer,
        num_classes=train_obj.num_classes,
    )
    losses = pt_trainer.train(
        train_obj.train_dataloader, epoch=1, print_every=1
    )
    captured_sysout = capsys.readouterr()
    assert ("Epoch: [2][5 / 5]" in captured_sysout.out) and (
        isinstance(losses, float) and ~np.isnan(losses)
    )


# class TestPTHelper:
#     def test_train(self):
#         bs = 2
#         num_classes = 1
#         model = nn.Sequential(
#             nn.Linear(28 * 28, 16), nn.Linear(16, num_classes)
#         )
#         criterion = (
#             nn.BCEWithLogitsLoss()
#             if num_classes == 1
#             else nn.CrossEntropyLoss()
#         )
#         optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
#         device = torch.device("cpu")
#         train_dataloader = torch.utils.data.DataLoader(
#             DummyDataset(), batch_size=bs
#         )
#         pt_trainer = trainer.PTHelper(
#             model,
#             device,
#             criterion,
#             optimizer,
#             num_classes=num_classes,
#         )
#         losses = pt_trainer.train(train_dataloader, epoch=1, print_every=2)
#         assert isinstance(losses, float) and ~np.isnan(losses)
