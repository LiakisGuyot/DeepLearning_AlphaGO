import numpy as np
from game.connect4.utils import *
from game.game import Game


class Connect4(Game):
    def __init__(self):
        super().__init__(row_count=6, column_count=7, action_size=7)

    def __repr__(self):
        return "Connect4"

    def get_init_state(self):
        return np.zeros((self.row_count, self.column_count))

    def apply_action(self, state: np.ndarray, player: int, action: int):
        row_i = np.max(np.where(state[:, action] == 0))
        state[row_i, action] = player
        return state

    def get_legal_moves(self, state: np.ndarray):
        return (state[0] == 0).astype(np.uint8)

    def __is_win(self, state: np.ndarray, action: int):
        """
        Does the last action made the player win the game ?
        :param state:
        :param action:
        :return:
        """
        if action is None:
            return False
        row = np.min(np.where(state[:, action] != 0))
        player = state[row][action]
        _state = state.flatten()
        for a, b, c, d in win_states:
            if _state[a] + _state[b] + _state[c] + _state[d] == 4 * player:
                return True
        return False

    def get_value_and_ended(self, state: np.ndarray, action: int):
        if self.__is_win(state, action):
            return 1, True
        if np.sum(self.get_legal_moves(state)) == 0:
            return 0, True
        return 0, False

    def get_opponent(self, player: int):
        return -player

    def get_opponent_value(self, value: float):
        return -value

    def switch_state_perspective(self, state: np.ndarray, player: int):
        return state * player

    def get_encoded_state(self, state: np.ndarray):
        encoded_state = np.stack((
            state == -1,
            state == 0,
            state == 1
        )).astype(np.float32)

        # Multiple states
        if len(state.shape) == 3:
            encoded_state = np.swapaxes(encoded_state, 0, 1)

        return encoded_state

    def render(self, state: np.ndarray):
        print()
        for i in range(len(state)):
            line = "| "
            for j in range(len(state[i])):
                line += pieces[state[i, j]] + " "
            print(line + "|")
