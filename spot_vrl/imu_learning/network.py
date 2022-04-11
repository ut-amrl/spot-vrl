from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from spot_vrl.imu_learning.datasets import Triplet


class BaseEmbeddingNet(nn.Module, ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...

    @abstractmethod
    def arch(self) -> str:
        ...


class MlpEmbeddingNet(BaseEmbeddingNet):
    def __init__(self, input_shape: Tuple[int, ...], embedding_dim: int):
        super().__init__()

        input_dim = np.prod(input_shape)
        self.sizes = [input_dim, 64, 32, embedding_dim]
        # self.sizes = [input_dim, 1024, 256, embedding_dim]
        # self.sizes = [input_dim, 1024, 1024, 256, embedding_dim]

        layers = torch.nn.ModuleList()

        # layers.append(nn.Dropout(0.2))
        for i in range(len(self.sizes) - 1):
            if i > 0:
                # layers.append(nn.Dropout(0.2))
                layers.append(nn.PReLU())
            layers.append(nn.Linear(self.sizes[i], self.sizes[i + 1]))

        self._fc = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Tensor of shape
                (batch size, num observations, imu_categories)

        Returns:
            torch.Tensor: Tensor of shape (batch size, embedding dim)
        """
        x = x.view(x.shape[0], -1)
        out = self._fc(x)
        return F.normalize(out, p=2, dim=1)

    def arch(self) -> str:
        return "mlp" + "-".join(str(x) for x in self.sizes)


class LstmEmbeddingNet(BaseEmbeddingNet):
    def __init__(self, input_shape: Tuple[int, ...], embedding_dim: int):
        super().__init__()

        window_size = input_shape[0]
        input_vec_size = input_shape[1]

        self.num_blocks = 1
        self.num_layers = 8

        rnn_layers = torch.nn.ModuleList()

        # TODO: specify correct input and hidden sizes for num_blocks > 1
        for _ in range(self.num_blocks):
            rnn_layers.append(
                nn.LSTM(
                    input_size=input_vec_size,
                    hidden_size=embedding_dim,
                    num_layers=self.num_layers,
                    batch_first=True,
                )  # type: ignore
            )

        self._rnn = torch.nn.Sequential(*rnn_layers)
        self._fc = nn.Linear(window_size * embedding_dim, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Tensor of shape
                (batch size, num observations, imu_categories)

        Returns:
            torch.Tensor: Tensor of shape (batch size, embedding dim)
        """
        output: torch.Tensor
        output, _ = self._rnn(x)
        output = output.reshape(output.shape[0], -1)
        output = self._fc(output)
        return F.normalize(output, p=2, dim=1)

    def arch(self) -> str:
        return f"lstm-nblk{self.num_blocks}-nlyr{self.num_layers}"


class TripletNet(nn.Module):
    def __init__(self, embedding_net: BaseEmbeddingNet):
        super().__init__()
        self._embedding_net = embedding_net

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        return self._embedding_net.forward(x)

    def forward(self, t: Triplet) -> Triplet:
        e_anchor = self.get_embedding(t[0])
        e_pos = self.get_embedding(t[1])
        e_neg = self.get_embedding(t[2])

        return e_anchor, e_pos, e_neg


class CostNet(nn.Module):
    def __init__(self, embedding_dim: int = 48) -> None:
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(embedding_dim, 32),
            nn.PReLU(),
            nn.Linear(32, 32),
            nn.PReLU(),
            nn.Linear(32, 16),
            nn.PReLU(),
            nn.Linear(16, 1),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)  # type: ignore


class FullPairCostNet(nn.Module):
    def __init__(self, embedding_net: BaseEmbeddingNet, cost_net: CostNet) -> None:
        super().__init__()

        self.embedding_net = embedding_net
        self.cost_net = cost_net

    def forward(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        emb_x = self.embedding_net(x)
        emb_y = self.embedding_net(y)
        return (self.cost_net(emb_x), self.cost_net(emb_y))
