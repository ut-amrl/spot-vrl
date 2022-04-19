import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger


class MarginRankingLoss(nn.Module):
    """
    Loss function based on torch.nn.MarginRankingLoss.

    Creates a criterion that measures the loss given mini-batch inputs x1, x2
    and a mini-batch label y (containing -1, 0, or 1).

    If y=1, then it is assumed the first input should be ranked higher (have a
    larger value) than the second input), and vice-versa for y=-1.

    If y=0, then it is assumed the inputs are of the same rank (have similar
    values).

    The loss function for y=1,-1 is identical to torch.nn.MarginRankingLoss:
        loss(x1, x2, y) = max(0, -y * (x1 - x2) + margin)

    The loss function for y=0 enforces that the distance between the inputs is
    within the margin:
        loss(x1, x2, y) = max(0, abs(x1 - x2) - margin)

    See also:
    https://pytorch.org/docs/stable/generated/torch.nn.MarginRankingLoss.html
    """

    def __init__(self, margin: float, reduction: str = "mean") -> None:
        super().__init__()
        self.margin = margin

        reduction = reduction.lower()
        if reduction == "mean":
            self.reduction = torch.mean
        elif reduction == "sum":
            self.reduction = torch.sum
        elif reduction == "none":
            self.reduction = lambda x: x  # type: ignore
        else:
            raise ValueError(f"Unknown reduction '{reduction}'")

    def forward(
        self, x1: torch.Tensor, x2: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """Computes the loss.

        Args:
            x1: 1D minibatch Tensor
            x2: 1D minibatch Tensor
            y:  1D minibatch Tensor, with values -1, 0, or 1

        Implementation:
            Let z = (x1 - x2)

            Standard MarginRankingLoss (y=1, y=-1) computes:
            [1]     -y * z + margin

            However, for same-class pairs (y=0), we want to compute
            [2]     abs(z) - margin

            To compute [2] using the form of [1], for y=0 let
                y' = -1
                z' = abs(z) - 2*margin

            Thus,
                -y' * z' + margin
                =   (abs(z) - 2*margin) + margin
                =   abs(z) - margin
        """
        with logger.catch(
            message="Expected inputs of the same shape", onerror=sys.exit
        ):
            assert x1.shape == x2.shape
            assert x1.shape == y.shape

        z = x1 - x2
        z = torch.where(y == 0.0, torch.abs(z) - 2 * self.margin, z)
        y = torch.where(y == 0.0, -1.0, y)

        return self.reduction(F.relu(-y * z + self.margin))
