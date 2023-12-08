import numpy as np
from connect4.utils import *


def init_board():
    return np.zeros((6, 7), dtype=int)


def apply_action(board: np.ndarray, player: int, action: int):
    """
    Apply an action in the board.
    :param board:
    :param player:
    :param action:
    :return:
    """
    _board = board.copy()
    row_i = sum(board[:, action] == 0) - 1
    _board[row_i, action] = player
    return _board


def get_legal_moves(board: np.ndarray):
    """
    Get all legal moves.
    :param board:
    :return:
    """
    legal_moves = np.zeros(7, dtype=int)
    legal_moves[board[0, :] == 0] = 1
    return legal_moves


def is_draw(board: np.ndarray):
    """
    Is the board full ?
    :param board:
    :return:
    """
    return np.count_nonzero(board) == 42


def is_win(board: np.ndarray, player: int):
    """
    Does the player win the game ?
    :param board:
    :param player:
    :return:
    """
    _board = board.flatten()
    for a, b, c, d in win_states:
        if _board[a] + _board[b] + _board[c] + _board[d] == 4 * player:
            return True
    return False


def render(board: np.ndarray):
    """
    Render the board.
    :param board:
    :return:
    """
    print()
    for i in range(len(board)):
        line = "| "
        for j in range(len(board[i])):
            line += pieces[board[i, j]] + " "
        print(line + "|")
