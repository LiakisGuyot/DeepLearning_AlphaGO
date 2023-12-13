import math

import numpy as np

from game.game import Game
from settings import C


class Node:
    def __init__(self, game: Game, state: np.ndarray, parent: 'Node' = None, action: int = None,
                 prior: float = 0, visits: int = 0):
        self.game = game
        self.state = state
        self.parent = parent
        self.action = action

        self.children = []

        self.prior = prior
        self.visits = visits
        self.value = 0

    def is_expanded(self):
        """
        Is the node expanded ?
        :return:
        """
        return len(self.children) > 0

    def select(self):
        """
        Get the best child depending on UCB score (Upper Confident Bounds).
        :return:
        """
        best_child = None
        best_ucb_score = -np.inf
        for child in self.children:
            ucb = self.ucb_score(child)
            if ucb > best_ucb_score:
                best_child = child
                best_ucb_score = ucb
        return best_child

    def expand(self, policy: np.ndarray):
        """
        Expand parent node into children nodes for each legal moves (based on policy).
        :param policy:
        :return:
        """
        for action, prob in enumerate(policy):
            if prob > 0:
                child_state = self.state.copy()
                child_state = self.game.apply_action(state=child_state, player=1, action=action)
                child_state = self.game.switch_state_perspective(state=child_state, player=-1)

                child = Node(self.game, child_state, self, action, prob)
                self.children.append(child)

    def back_propagate(self, value):
        """
        Back propagate value through parents.
        :param value:
        :return:
        """
        self.value += value
        self.visits += 1

        value = self.game.get_opponent_value(value)
        if self.parent is not None:
            self.parent.back_propagate(value)

    def ucb_score(self, child: 'Node'):
        """
        Get UCB score of parent-child.
        :param child:
        :return:
        """
        if child.visits == 0:
            q_val = 0
        else:
            q_val = 1 - ((child.value / child.visits) + 1) / 2
        return q_val + child.prior * C * (math.sqrt(self.visits) / (child.visits + 1))
