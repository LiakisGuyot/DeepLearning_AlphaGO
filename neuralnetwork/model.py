import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary


class Connect4Model(nn.Module):
    def __init__(self, device):
        """
        Model initialization.
        :param device:
        """
        super().__init__()
        self.device = device

        # Conv
        # in_channels : 3 (-1,1 or 0)
        # out_channels : size of the nn, best -> 128
        # kernel_size : 3
        # padding : 1 with zeros
        self.initial_conv = nn.Conv2d(in_channels=3, out_channels=128, kernel_size=3, stride=1, padding=1, bias=True)
        self.initial_bn = nn.BatchNorm2d(num_features=128)

        # Res block 1
        # Same than conv, but with in_channels equals to num features & bias False
        self.res1_conv1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.res1_bn1 = nn.BatchNorm2d(num_features=128)
        self.res1_conv2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.res1_bn2 = nn.BatchNorm2d(num_features=128)

        # Res block 2 (no need of too many res block)
        # Same than res block 1
        self.res2_conv1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.res2_bn1 = nn.BatchNorm2d(num_features=128)
        self.res2_conv2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.res2_bn2 = nn.BatchNorm2d(num_features=128)

        # value head
        # in_channels : 128
        # out_channels : 3 (-1, 1 or 0)
        # kernel_size : 1 (appears to be the best)
        # no padding
        # with bias for each position
        self.value_conv = nn.Conv2d(in_channels=128, out_channels=3, kernel_size=1, stride=1, bias=True)
        self.value_bn = nn.BatchNorm2d(num_features=3)
        # in_features : 3*6*7 => 3*(rows)*(columns)
        # out_features : 32 => number of neurones
        self.value_func = nn.Linear(in_features=3 * 6 * 7, out_features=32)
        # Final value head : 32 -> 1 out_feature
        self.value_head = nn.Linear(in_features=32, out_features=1)

        # policy head
        # in_channels : 128
        # out_channels : 32 (number of neurones)
        # kernel_size : 1 (appears to be the best)
        # no padding
        # with Bias for each position
        self.policy_conv = nn.Conv2d(in_channels=128, out_channels=32, kernel_size=1, stride=1, bias=True)
        self.policy_bn = nn.BatchNorm2d(num_features=32)
        # in_features : 32*6*7 => 32*(rows)*(columns)
        # out_features : 7 => 7 moves [0,1,2,3,4,5,6]
        self.policy_head = nn.Linear(in_features=32 * 6 * 7, out_features=7)
        # Convert into probabilities
        self.policy_lsm = nn.LogSoftmax(dim=1)

        # Using device (GPU or CPU)
        self.to(device)

    def forward(self, x):
        """
        Model layers connections (x shape of 3 * 6 * 7).
        :param x:
        :return:
        """
        # Adding dimension for batch size
        x = x.view(-1, 3, 6, 7)
        x = self.initial_bn(self.initial_conv(x))
        x = F.relu(x)  # Activation function ReLu

        # Res Block 1
        res = x
        x = self.res1_bn1(self.res1_conv1(x))
        x = F.relu(x)
        x = self.res1_bn2(self.res1_conv2(x))
        x = F.relu(x)
        x += res
        x = F.relu(x)

        # Res Block 2
        res = x
        x = self.res2_bn1(self.res2_conv1(x))
        x = F.relu(x)
        x = self.res2_bn2(self.res2_conv2(x))
        x = F.relu(x)
        x += res
        x = F.relu(x)

        # value head
        v = self.value_bn(self.value_conv(x))
        v = F.relu(v)
        v = v.view(-1, 3 * 6 * 7)
        v = self.value_func(v)
        v = F.relu(v)
        v = F.tanh(v)

        # policy head
        p = self.policy_bn(self.policy_conv(x))
        p = F.relu(p)
        p = p.view(-1, 32 * 6 * 7)
        p = self.policy_head(p)
        p = F.relu(p)
        p = self.policy_lsm(p).exp()

        return v, p


def get_model_arch():
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = Connect4Model(dev)
    arch_summary = summary(model, input_size=(16, 3, 6, 7), verbose=0)
    print(arch_summary)

