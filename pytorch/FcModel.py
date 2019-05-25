"""
May 19. 2019
Fully-connected ANN for a baseline model of forecasting
"""
from typing import Set

import torch
import torch.nn.functional as F


class Net(torch.nn.Module):
    def __init__(
        self,
        num_fea: int,
        num_tar: int,
        neurons: Set[int]
    ) -> None:
        super().__init__()
        self.fc1 = torch.nn.Linear(num_fea, neurons[0])
        self.fc2 = torch.nn.Linear(neurons[0], neurons[1])
        self.fc3 = torch.nn.Linear(neurons[1], num_tar)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

if __name__ == "__main__":
    pass
