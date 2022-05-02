import os
import logging
import time
import math
from pathlib import Path

import torch
from torchinfo import summary


class MetricMonitor:
    """Calculates and stroes the average value of the metrics/loss."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all the parameters to zero."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        """Update the value of the metrics and calculate their average value
        over the entire dataset

        Args:
        -----
            val (float): Computed metric (per batch).
            n (int, optional): Batch size. Defaults to 1.
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def time_since(since: int, percent: float) -> str:
    """Helper function to time the training and evaluation process.

    Args:
    -----
        since (int): Start time.
        percent (float): Percent to the task done.

    Returns:
    --------
        str: Print elapsed/remaining time to console.
    """

    def as_minutes_seconds(s: float) -> str:
        m = math.floor(s / 60)
        s -= m * 60
        m, s = int(m), int(s)
        return f"{m:2d}m {s:2d}s"

    now = time.time()
    elapsed = now - since
    total_estimated = elapsed / percent
    remaining = total_estimated - elapsed
    return f"{as_minutes_seconds(elapsed)} (remain {as_minutes_seconds(remaining)}"


def get_model_params(model: torch.nn.Module) -> int:
    """Helper function to determine the total number of the trainable parameters
    in the PyTorch model.

    Args:
    -----
        model (torch.nn.Module): Instance of the PyTorch model being used.

    Returns:
    --------
        int: Number of the trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_details(
    model: torch.nn.Module,
    x: torch.Tensor,
    input_size: tuple,
    device: torch.device,
):
    """Print Keras like model details on the console.

    Args:
    -----
        model (torch.nn.Module): Instance of the PyTorch model being used.
        x (torch.Tensor): Dummy input.
        input_size (tuple): Size of the input.
    """
    print("\t\t\t\tMODEL SUMMARY")
    summary(model, input_size=input_size, device=device)
    print(f"Batched input size: {x.shape}")
    print(f"Batched output size: {model(x).shape}")
    print(f"Model contains {get_model_params(model)} trainable parameters.")
