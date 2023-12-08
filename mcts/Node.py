import math

import numpy as np
from connect4.game import apply_action


class Node:
    def __init__(self, prior: int, player_turn: int, state: np.ndarray):
        self.prior = prior
        self.player_turn = player_turn
        self.state = state
        self.children = {}
        self.value = 0
        self.visits = 0

    def expand(self, action_p: list[float | int]):
        """
        Expand a node (creation of children).
        :param action_p:
        :return:
        """
        for action, p in enumerate(action_p):
            if p > 0:
                next_state = apply_action(self.state, self.player_turn, action)
                self.children[action] = Node(p, -self.player_turn, next_state)

    def select_child(self):
        """
        Select best child (random here).
        :return:
        """
        max_action, max_child = max(self.children.items(), key=lambda item: ucb(self, item[1]))
        return max_action, max_child


def ucb(parent: Node, child: Node):
    """
    UCB score function.
    :param parent:
    :param child:
    :return:
    """
    p_score = child.prior * math.sqrt(parent.visits) / (child.visits + 1)
    if child.visits == 0:
        return p_score
    v_score = child.value / child.visits
    return v_score + p_score
