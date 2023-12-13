import os

import numpy as np
import torch
from matplotlib import pyplot as plt

from game.connect4.connect4 import Connect4
from game.connect4.utils import pieces
from mcts.mcts import MCTS
from neuralnetwork.alphazero import AlphaZero
from neuralnetwork.model import Model
from notif.notificationbot import NotificationBot

from dotenv import load_dotenv

load_dotenv()

MODEL_FILENAME_TO_USE = "run/Connect4/model_latest.pth"
bot = NotificationBot(os.getenv("USER_KEY"), os.getenv("API_TOKEN"))


def play_against_ai():
    """
    Start a session Human vs AI.
    :return:
    """
    game = Connect4()
    model = Model(game, 9, 128)
    loaded_model = torch.load(MODEL_FILENAME_TO_USE, map_location=model.device)
    model.load_state_dict(loaded_model["model_state_dict"])

    player = 1
    state = game.get_init_state()
    while True:
        game.render(state)

        if player == 1:
            valid_moves = game.get_legal_moves(state)
            print("valid moves :", [i for i in range(game.action_size) if valid_moves[i] == 1])
            action = int(input(f"Player {pieces[player]} play an action :"))

            if valid_moves[action] == 0:
                print(f"Action {action} unvalid")
                continue
        else:
            encoded_state = game.get_encoded_state(state)
            tensor_state = torch.tensor(encoded_state, device=model.device).unsqueeze(0)
            policy, value = model(tensor_state)
            policy = torch.softmax(policy, 1).squeeze(0).detach().cpu().numpy()

            valid_moves = game.get_legal_moves(state)
            policy *= valid_moves
            policy /= np.sum(policy)

            action = np.argmax(policy)

        state = game.apply_action(state, player, action)

        value, is_terminal = game.get_value_and_ended(state, action)
        if is_terminal:
            game.render(state)
            if value == 1:
                print(f"Player {pieces[player]} won.")
            else:
                print("Game ended : draw")
            break

        player = game.get_opponent(player)


def train_ai():
    """
    Train AI via AlphaZero with settings (in settings.py file).
    :return:
    """
    game = Connect4()
    model = Model(game, 9, 128)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    az = AlphaZero(model, optimizer, game)
    az.learn(bot)


def show_loss_plot():
    """
    After training, show losses plot via loading trained model.
    :return:
    """
    game = Connect4()
    model = Model(game, 9, 128)
    losses_tot = []
    loaded_model = torch.load(MODEL_FILENAME_TO_USE, map_location=model.device)
    for epoch, losses in loaded_model["losses"].items():
        losses_tot += losses

    plt.plot(range(1, len(losses_tot) + 1), losses_tot, label='Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Loss evolution for each epoch')
    plt.legend()

    plt.show()


if __name__ == '__main__':
    print("Welcome aboard, can I help you ?\n")
    print("Enter 1 to train the AI with settings set...........")
    print("Enter 2 to play against the AI......................")
    print("Enter 3 to show losses plot (only after training°...")
    choice = -1
    while choice == -1:
        try:
            choice = int(input())
        finally:
            pass
    if choice == 1:
        train_ai()
    if choice == 2:
        play_against_ai()
    if choice == 3:
        show_loss_plot()
