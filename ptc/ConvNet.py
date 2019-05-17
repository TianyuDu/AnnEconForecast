import torch
import torchvision


def ConvNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(
            in_channels=3,
            out_channels=6,
            kernel_size=5)

        self.pool = torch.nn.MaxPool2d(
            kernel_size=2,
            stride=2)

        self.conv2 = torch.nn.Conv(
            in_channels=6,
            out_channels=16,
            kernel_size=5)

        self.fc1 = torch.nn.Linear(16*5*5, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 10)

    def forward(self, inputs):
        
