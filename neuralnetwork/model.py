import torch.cuda
from torch import nn

from game.game import Game
from neuralnetwork.residualblock import ResBlock


class Model(nn.Module):
    def __init__(self, game: Game, res_blocks: int, hidden: int):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.start_block = nn.Sequential(
            nn.Conv2d(3, hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden),
            nn.ReLU()
        )

        self.back_bone = nn.ModuleList(
            [ResBlock(hidden) for i in range(res_blocks)]
        )

        self.policy_head = nn.Sequential(
            nn.Conv2d(hidden, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * game.row_count * game.column_count, game.action_size)
        )

        self.value_head = nn.Sequential(
            nn.Conv2d(hidden, 3, kernel_size=3, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3 * game.row_count * game.column_count, 1),
            nn.Tanh()
        )

        self.to(self.device)

    def forward(self, x):
        """
        Step into the model.
        Represents the connections between layers.
        :param x:
        :return:
        """
        x = self.start_block(x)
        for res_block in self.back_bone:
            x = res_block(x)
        policy = self.policy_head(x)
        value = self.value_head(x)
        return policy, value
