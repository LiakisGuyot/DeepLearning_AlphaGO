from game.state import State


class Node:
    def __init__(self, state: State):
        self.state = state
        self.player_turn = state.player_turn
        self.id = state.id
        self.edges = []

    def is_leaf(self):
        """
        Is the node a leaf ?
        :return is_leaf: boolean
        """
        if len(self.edges) == 0:
            return True
        return False
    