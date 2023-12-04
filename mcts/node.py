import random

from connect4.state import State


class Node:
    def __init__(self, state: State):
        self.state = state
        self.player_turn = state.player_turn

    def find_children(self):
        if self.state.is_ended():
            return set()
        children = set()
        for action in self.state.get_legal_actions():
            next_state, _ = self.state.apply_action(action)
            children.add(Node(next_state))
        return children

    def find_random_child(self):
        if self.state.is_ended():
            return None
        action = random.choice(self.state.get_legal_actions())
        return Node(self.state.apply_action(action)[0])

    def is_terminal(self):
        return self.state.is_ended()

    def reward(self):
        return self.state.get_value(self.player_turn)  # last player played final move
