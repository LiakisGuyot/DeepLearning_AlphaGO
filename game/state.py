import numpy as np
from logging import Logger
from utils import winner_states, pieces


class State:
    def __init__(self, board: np.array, player_turn: int):
        self.board = board
        self.player_turn = player_turn
        self.binary_size = self._convert_to_binary(player_turn)
        self.id = self._convert_to_id()
        self.legal_actions = self._get_legal_actions()
        self.is_ended = self._check_for_end_state()
        self.value = self._get_value()
        self.score = self._get_score()

    def _convert_to_binary(self, player_base: int):
        """
        Convert board tab to a merged one :
        - first half : all player piece positions
        - second half : all opponent player piece positions
        :param player_base: int
        :return positions: Numpy Array
        """
        player_pos = np.zeros(len(self.board), dtype=np.int32)
        player_pos[self.board == player_base] = 1
        opp_player_pos = np.zeros(len(self.board), dtype=np.int32)
        opp_player_pos[self.board == -player_base] = 1

        return np.append(player_pos, opp_player_pos)

    def _convert_to_id(self):
        """
        Generate board id thanks to binary (player_base = 1)
        :return id: string
        """
        binary = self._convert_to_binary(1)
        return ''.join(map(str, binary))

    def __is_legal_action(self, action: int):
        """
        Private function : returns if the action is legal
        :param action: int
        :return legal: bool
        """
        if action >= len(self.board) - 7:
            return self.board[action] == 0
        else:
            return self.board[action] == 0 and self.board[action + 7] != 0

    def _get_legal_actions(self):
        """
        Get all legal actions from a state
        :return legal_actions: int[]
        """
        return [i for i in range(len(self.board)) if self.__is_legal_action(i)]

    def _check_for_end_state(self):
        """
        Check if game is in an end state after a step
        :return ended: bool
        """
        if np.count_nonzero(self.board) == 42:
            return True

        for a, b, c, d in winner_states:
            if self.board[a] + self.board[b] + self.board[c] + self.board[d] == 4 * -self.player_turn:
                return True
        return False

    def _get_value(self):
        """
        Get the state's value for the curr player
        :return:
        """
        for a, b, c, d in winner_states:
            if self.board[a] + self.board[b] + self.board[c] + self.board[d] == 4 * -self.player_turn:
                return -1, -1, 1  # lose
        return 0, 0, 0

    def _get_score(self):
        """
        Get the state's score
        :return score: int[]
        """
        return self.value[0], self.value[1]

    def apply_action(self, action: int):
        """
        Apply an action
        :param action: int
        :return next_state, value, done: mix
        """
        next_board = np.array(self.board)
        next_board[action] = self.player_turn
        next_state = State(next_board, -self.player_turn)

        value = next_state.value[0] if next_state.is_ended else 0
        done = 1 if next_state.is_ended else 0
        return next_state, value, done

    def print(self, logger: Logger):
        """
        Print the board state
        :param logger:
        :return:
        """
        for r in range(6):
            logger.info([pieces[str(x)] for x in self.board[7 * r: (7 * r + 7)]])
        logger.info('--------------')
