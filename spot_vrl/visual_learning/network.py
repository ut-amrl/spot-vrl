from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn

from spot_vrl.visual_learning.datasets import Triplet


class EmbeddingNet(nn.Module):
    class ConvBlock(nn.Module):
        def __init__(
            self, in_channels: int, out_channels: int, kernel_size: int
        ) -> None:
            super().__init__()
            self.block = torch.nn.Sequential(
                # shrinks the image size by a factor of 2
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=2,
                    padding=kernel_size // 2,
                ),
                nn.BatchNorm2d(out_channels),  # type: ignore
                nn.PReLU(),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.block(x)  # type: ignore

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()

        # Reduce 1x60x60 image into a 128x1x1 image
        self.convnet = nn.Sequential(
            # 1x60x60
            self.ConvBlock(1, 16, 7),
            # 16x30x30
            self.ConvBlock(16, 64, 5),
            # 64x15x15
            self.ConvBlock(64, 128, 5),
            # 128 x 7 x 7
        )

        self.fc = nn.Linear(128, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # input is BxHxW, convnet wants BxCxHxW
        # TODO: do input scaling outside of network?
        x = x[:, None, :, :].float() / 255
        output: torch.Tensor = self.convnet(x)
        output = output.mean(dim=(2, 3))
        output = self.fc(output)
        return F.normalize(output, p=2, dim=1)


class TripletNet(nn.Module):
    def __init__(self, embedding_net: EmbeddingNet) -> None:
        super().__init__()
        self.embedding_net = embedding_net

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        # exists for type coercion
        return self.embedding_net(x)  # type: ignore

    def forward(self, t: Triplet) -> Triplet:
        e_anchor = self.get_embedding(t[0])
        e_pos = self.get_embedding(t[1])
        e_neg = self.get_embedding(t[2])

        return e_anchor, e_pos, e_neg


class CostNet(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.PReLU(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.PReLU(),
            nn.Linear(embedding_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.fc(x)

        # Enforce non-negative costs. The exponential function is used instead
        # of ReLU since ReLU will produce a 0 gradient if the network output is
        # all non-positive.
        return torch.exp(out)


class FullPairCostNet(nn.Module):
    def __init__(self, triplet_net: TripletNet, cost_net: CostNet) -> None:
        super().__init__()

        self.triplet_net = triplet_net
        self.cost_net = cost_net

    def get_cost(self, patch: torch.Tensor) -> torch.Tensor:
        embed = self.triplet_net.get_embedding(patch)
        return self.cost_net(embed)  # type: ignore

    def forward(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        emb_x = self.triplet_net.get_embedding(x)
        emb_y = self.triplet_net.get_embedding(y)
        return (self.cost_net(emb_x), self.cost_net(emb_y))
