from typing import Any, List, Tuple

import numpy as np
import torch
from torch import nn

from spot_vrl.imu_learning.datasets import Triplet


class EmbeddingNet(nn.Module):
    def __init__(self, input_shape: Tuple[int, ...], embedding_dim: int):
        super().__init__()

        input_dim = np.prod(input_shape)

        sizes = [input_dim, 256, 256, embedding_dim]
        layers: List[Any] = []

        for i in range(len(sizes) - 1):
            if i > 0:
                layers.append(nn.PReLU())
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))

        self._fc = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Tensor of shape
                (batch size, imu categories, num observations)

        Returns:
            torch.Tensor: Tensor of shape (batch size, embedding dim)
        """
        x = x.view(x.shape[0], -1)
        return self._fc.forward(x)  # type: ignore


class TripletNet(nn.Module):
    def __init__(self, embedding_net: EmbeddingNet):
        super().__init__()
        self._embedding_net = embedding_net

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        return self._embedding_net.forward(x)

    def forward(self, t: Triplet) -> Triplet:
        e_anchor = self.get_embedding(t[0])
        e_pos = self.get_embedding(t[1])
        e_neg = self.get_embedding(t[2])

        return e_anchor, e_pos, e_neg
