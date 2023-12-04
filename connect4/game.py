import random

import numpy as np

from connect4.state import State


class Game:
    def __init__(self):
        self.state = State(np.zeros(42, dtype=int), 1)

    def player_vs_player(self):
        is_ended = False
        while not is_ended:
            self.state.print()
            print(f"Player : {'X' if self.state.player_turn == 1 else 'O'} to play")
            print(self.state.get_legal_actions())
            action = input("Action : ")
            action_done, is_ended = self.state.apply_action(int(action))

    def player_vs_random(self):
        is_ended = False
        while not is_ended:
            self.state.print()
            print(f"Player : {'X' if self.state.player_turn == 1 else 'O'} to play")
            print(self.state.get_legal_actions())
            action = input("Action : ")
            is_ended = self.state.apply_action(int(action))
            if not is_ended:
                action = random.choice(self.state.get_legal_actions())
                is_ended = self.state.apply_action(action)
