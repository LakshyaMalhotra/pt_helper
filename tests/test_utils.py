import logging

import torch.nn as nn

from pthelper import utils


def test_logger(caplog):
    utils.get_logger().info("Logger is working fine!")
    assert caplog.record_tuples == [
        (
            "root",
            logging.INFO,
            "Logger is working fine!",
        )
    ]


def test_model_details(capsys):
    input_size = (2, 28 * 28)
    model = nn.Sequential(nn.Linear(28 * 28, 512), nn.Linear(512, 10))
    utils.model_details(model, input_size)
    captured = capsys.readouterr()
    assert ("Batched output size: torch.Size([2, 10])\n") in captured.out
