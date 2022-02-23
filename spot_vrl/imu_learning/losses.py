import torch
import torch.nn as nn
import torch.nn.functional as F

from spot_vrl.imu_learning.datasets import Triplet


class TripletLoss(nn.Module):
    """
    Triplet loss

    Takes embeddings of an anchor sample, a positive sample and a negative
    sample
    """

    def __init__(self, margin: float):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(
        self,
        t: Triplet,
        size_average: bool = True,
    ) -> torch.Tensor:
        distance_positive = torch.cdist(t[0], t[1])
        distance_negative = torch.cdist(t[0], t[2])
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()
