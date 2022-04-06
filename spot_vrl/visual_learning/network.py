import torch
from torch import nn

from spot_vrl.visual_learning.datasets import Triplet


class EmbeddingNet(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()

        # Reduce 1x60x60 image into a 128x1x1 image
        self.convnet = nn.Sequential(
            # 1x60x60
            nn.Conv2d(1, 16, 7, padding="same"),
            nn.PReLU(),
            nn.MaxPool2d(2, stride=2),
            # 16x30x30
            nn.Conv2d(16, 64, 5, padding="same"),
            nn.PReLU(),
            nn.MaxPool2d(2, stride=2),
            # 64x15x15
            nn.Conv2d(64, 128, 5, padding="same"),
            nn.PReLU(),
            nn.MaxPool2d(2, stride=2),
            # 128 x 7 x 7
            nn.AvgPool2d(7),
            # 128 x 1 x 1
        )

        self.fc = nn.Linear(128, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # input is BxHxW, convnet wants BxCxHxW
        x = x[:, None, :, :].float() / 255
        output: torch.Tensor = self.convnet(x)
        output = output.squeeze()
        output = self.fc(output)
        return output  # type: ignore


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
