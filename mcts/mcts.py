import numpy as np

import config
from node import Node
from logger.logger import logger_mcts as lg


def back_propagate(leaf: Node, value: float, path):
    """
    Back propagate edges. Updating stats for each one of them.
    :param leaf: Node
    :param value: flaot
    :param path: Edge[]
    :return:
    """
    lg.info('[ DOING BACK PROPAGATION ]')
    curr_player = leaf.player_turn
    for edge in path:
        player_turn = edge.player_turn
        value_updated = -value if player_turn == curr_player else value

        edge.stats['N'] += 1
        edge.stats['W'] += value_updated
        edge.stats['Q'] = edge.stats['W'] / edge.stats['N']

        lg.info('Updating edge with value %f for player %d... N = %d, W = %f, Q = %f',
                value_updated,
                player_turn,
                edge.stats['N'],
                edge.stats['W'],
                edge.stats['Q'])

        edge.out_node.state.print(lg)


class MCTS:
    def __init__(self, root: Node, cpuct):
        self.root = root
        self.tree = {}
        self.cpuct = cpuct
        self.add_node(root)

    def __len__(self):
        return len(self.tree)

    def move_to_leaf(self):
        """
        Moving from root node to leaf. Maximisation of Q+U and selection of the best edge.
        :return curr_node, value, done, path: mix
        """
        lg.info('[ MOVING TO LEAF ]')
        path = []
        curr_node = self.root
        done, value = 0, 0
        while not curr_node.is_leaf():
            lg.info('PLAYER TURN...%d', curr_node.player_turn)
            max_qu = -99999

            epsilon = config.EPSILON if curr_node == self.root else 0
            nu = np.random.dirichlet([config.ALPHA] * len(curr_node.edges)) if curr_node == self.root \
                else [0] * len(curr_node.edges)

            n = 0
            for _, edge in curr_node.edges:
                n += edge.stats['N']

            simulation_action = -1
            simulation_edge = None
            for i, (action, edge) in enumerate(curr_node.edges):
                # U value
                adj_p = ((1 - epsilon) * edge.stats['P'] + epsilon * nu[i])
                u = self.cpuct * adj_p * np.sqrt(n) / (1 + edge.stats['N'])
                # Q value
                q = edge.stats['Q']

                lg.info('action: %d (%d)... N = %d, P = %d, nu = %f, adjP = %f, W = %f, Q = %f, U = %f, QU = %f',
                        action, action % 7, edge.stats['N'], np.round(edge.stats['P'], 6), np.round(nu[i], 6),
                        adj_p, np.round(edge.stats['W'], 6), np.round(q, 6), np.round(u, 6), np.round(q + u, 6))

                if q + u > max_qu:
                    max_qu = q + u
                    simulation_action = action
                    simulation_edge = edge

            lg.info('action with best QU...%d', simulation_action)
            next_state, value, done = curr_node.state.apply_action(simulation_action)
            curr_node = simulation_edge.out_node
            path.append(simulation_edge)

        lg.info('DONE...%d', done)

        return curr_node, value, done, path

    def add_node(self, node: Node):
        """
        Add a node to the tree.
        :param node: Node
        :return:
        """
        self.tree[node.id] = node
