from torch import nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.conv1 = nn.Conv2d(hidden, hidden, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden)
        self.conv2 = nn.Conv2d(hidden, hidden, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(hidden)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.relu(x)
        return x
