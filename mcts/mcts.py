import math
from collections import defaultdict

from mcts.node import Node


class MCTS:
    def __init__(self, exploration_weight=1):
        self.q_val = defaultdict(int)
        self.n_val = defaultdict(int)
        self.children = dict()
        self.exploration_weight = exploration_weight

    def choose(self, node: Node):
        if node.is_terminal():
            raise RuntimeError(f"choose called on terminal node {node}")

        if node not in self.children:
            return node.find_random_child()

        def score(n):
            if self.n_val[n] == 0:
                return float("-inf")
            return self.q_val[n] / self.n_val[n]

        return max(self.children[node], key=score)

    def do_rollout(self, node: Node):
        path = self._select(node)
        leaf = path[-1]
        self._expand(leaf)
        reward = self._simulate(leaf)
        self._backpropagate(path, reward)

    def _select(self, node: Node):
        path = []
        while True:
            path.append(node)
            if node not in self.children or not self.children[node]:
                return path
            unexplored = self.children[node] - self.children.keys()
            if unexplored:
                n = unexplored.pop()
                path.append(n)
                return path
            node = self._uct_select(node)

    def _expand(self, node: Node):
        if node in self.children:
            return
        self.children[node] = node.find_children()

    def _simulate(self, node: Node):
        invert_reward = False
        while True:
            if node.is_terminal():
                reward = node.reward()
                return 1 - reward if invert_reward else reward
            node = node.find_random_child()
            invert_reward = not invert_reward

    def _backpropagate(self, path, reward):
        for node in reversed(path):
            self.n_val[node] += 1
            self.q_val[node] += reward
            reward = 1 - reward

    def _uct_select(self, node: Node):
        assert all(n in self.children for n in self.children[node])

        log_n_vertex = math.log(self.n_val[node])

        def uct(n):
            return self.q_val[n] / self.n_val[n] + self.exploration_weight * math.sqrt(
                log_n_vertex / self.n_val[n]
            )

        return max(self.children[node], key=uct)
