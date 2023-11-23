import numpy as np
from game.state import State


def invert_array(array: np.array):
    """
    Invert array for identities
    :param array: Numpy array
    :return inverted_array: Numpy array
    """
    if len(array) != 42:
        return None
    return np.array([
        array[6], array[5], array[4], array[3], array[2], array[1], array[0],
        array[13], array[12], array[11], array[10], array[9], array[8], array[7],
        array[20], array[19], array[18], array[17], array[16], array[15], array[14],
        array[27], array[26], array[25], array[24], array[23], array[22], array[21],
        array[34], array[33], array[32], array[31], array[30], array[29], array[28],
        array[41], array[40], array[39], array[38], array[37], array[36], array[35],
    ])


def identities(state: State, action_val: np.array):
    """
    Invert the board (same for front/back board)
    :param state: game.state.State
    :param action_val: Numpy array
    :return state, action_val, inverted_state, inverted_action_val: mix
    """
    return state, action_val, State(invert_array(state.board), state.player_turn), invert_array(action_val)


class Game:
    def __init__(self):
        self.player = 1
        self.state = State(np.zeros(42, dtype=np.int32), self.player)
        self.action_space = np.zeros(42, dtype=np.int32)
        self.grid_shape = (6, 7)
        self.input_shape = (2, 6, 7)
        self.name = "p4"
        self.state_size = len(self.state.binary_size)
        self.action_size = len(self.action_space)

    def reset(self):
        """
        Reset game state
        :return state: game.state.State
        """
        self.player = 1
        self.state = State(np.zeros(42, dtype=np.int32), self.player)
        return self.state

    def step(self, action: int):
        """
        Step function: new state, new value, is done ? Info...
        :param action: int
        :return next_state, value, done, info: mix
        """
        next_state, value, done = self.state.apply_action(action)
        self.state = next_state
        self.player = -self.player
        info = None
        return next_state, value, done, info
