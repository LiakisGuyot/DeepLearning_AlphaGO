from node import Node


class Edge:
    def __init__(self, in_node: Node, out_node: Node, prior: float, action: int):
        self.id = in_node.id + '-' + out_node.id
        self.in_node = in_node
        self.out_node = out_node
        self.player_turn = in_node.player_turn
        self.action = action
        self.stats = {
            'N': 0,
            'W': 0,
            'Q': 0,
            'P': prior
        }
