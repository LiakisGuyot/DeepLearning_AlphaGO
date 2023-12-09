import numpy as np
import torch

from connect4.game import render, apply_action, is_win, is_draw
from mcts.mcts import MCTS
from mcts.node import Node
from neuralnetwork.model import Connect4Model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

settings = {
}


test_board = np.array([
    [0, -1, -1, -1, 1, 0, -1],
    [0, 1, -1, 1, 1, 0, 1],
    [-1, 1, -1, 1, 1, 0, -1],
    [1, -1, 1, -1, -1, 0, -1],
    [-1, -1, 1, -1, 1, 1, -1],
    [-1, 1, 1, -1, 1, -1, 1]
])

player_turn = 1


if __name__ == '__main__':
    mcts = MCTS({'num_simulations': 100})
    i = 0
    while True:
        print("\nIteration", (i//2)+1, "joueur :", player_turn)
        r = mcts.run(Connect4Model(device), test_board, player_turn)

        max_a, _ = r.select_child()
        print(max_a)
        test_board = apply_action(test_board, player_turn, max_a)
        render(test_board)
        if is_win(test_board, player_turn):
            print("WIN", player_turn)
            break
        if is_draw(test_board):
            print("AH")
            break
        player_turn = -player_turn
        i+=1
