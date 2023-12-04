import numpy as np

from connect4.utils import win_states, pieces


class State:
    def __init__(self, board: np.array, player_turn: int):
        self.board = board
        self.player_turn = player_turn

    def get_id(self):
        player_pos = np.zeros(len(self.board), dtype=int)
        player_pos[self.board == 1] = 1
        opp_pos = np.zeros(len(self.board), dtype=int)
        opp_pos[self.board == -1] = 1
        pos = np.append(player_pos, opp_pos)

        return ''.join(map(str, pos))

    def get_legal_actions(self):
        legal_actions = []
        for i in range(len(self.board)):
            if self.__is_action_legal(i):
                legal_actions.append(i)

        return legal_actions

    def __is_action_legal(self, action: int):
        if action >= len(self.board) - 7:
            if self.board[action] == 0:
                return True
        else:
            if self.board[action] == 0 and self.board[action + 7] != 0:
                return True
        return False

    def apply_action(self, action: int):
        next_state = None
        if self.__is_action_legal(action):
            next_state = State(np.copy(self.board), -self.player_turn)
            next_state.board[action] = self.player_turn
            return next_state, self.is_ended()
        return next_state, False

    def is_ended(self):
        if np.count_nonzero(self.board) == 42:
            return True
        for a, b, c, d in win_states:
            if self.board[a] + self.board[b] + self.board[c] + self.board[d] == 4 * -self.player_turn:
                return True
        return False

    def get_value(self, player: int):
        if np.count_nonzero(self.board) == 42:
            return 0.5  # Draw
        for a, b, c, d in win_states:
            if self.board[a] + self.board[b] + self.board[c] + self.board[d] == 4 * player:  # Win
                return 1
        return 0  # Lose

    def print(self):
        print("-------------- STATE --------------")
        for r in range(6):
            print([pieces[str(x)] for x in self.board[7 * r: (7 * r + 7)]])
        print("-----------------------------------")
