from abc import ABC, abstractmethod

import numpy as np


class Game(ABC):
    def __init__(self, row_count, column_count, action_size):
        self.row_count = row_count
        self.column_count = column_count
        self.action_size = action_size

    @abstractmethod
    def get_init_state(self):
        """
        Get initial state.
        :return:
        """
        pass

    @abstractmethod
    def apply_action(self, state: np.ndarray, player: int, action: int):
        """
        Apply an action in the state.
        :param state:
        :param player:
        :param action:
        :return:
        """
        pass

    @abstractmethod
    def get_legal_moves(self, state: np.ndarray):
        """
        Get all legal moves.
        :param state:
        :return:
        """
        pass

    @abstractmethod
    def get_value_and_ended(self, state: np.ndarray, action: int):
        """
        Get state value depending on the last action, and if game is ended.
        :param state:
        :param action:
        :return:
        """
        pass

    @abstractmethod
    def get_opponent(self, player: int):
        """
        Get opponent player.
        :param player:
        :return:
        """
        pass

    @abstractmethod
    def get_opponent_value(self, value: float):
        """
        Get opponent value.
        :param value:
        :return:
        """
        pass

    @abstractmethod
    def switch_state_perspective(self, state: np.ndarray, player: int):
        """
        Switch state perspective from one player to his opponent.
        :param state:
        :param player:
        :return:
        """
        pass

    @abstractmethod
    def get_encoded_state(self, state: np.ndarray):
        """
        Get encoded state :
        - channel 0 : state with 1 for -1
        - channel 1 : state with 1 for 0
        - channel 2 : state with 1 for 1
        Total channels : 3
        Can be multiple states to be encoded...
        :param state:
        :return:
        """
        pass

    @abstractmethod
    def render(self, state: np.ndarray):
        """
        Render the state.
        :param state:
        :return:
        """
        pass
