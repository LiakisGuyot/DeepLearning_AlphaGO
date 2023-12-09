from connect4.game import *
from mcts.node import Node


def dummy_model_predict(board: np.ndarray):
    value_head = 0.5
    policy_head = [0.5, 0, 0, 0, 0, 0.5, 0]
    return value_head, policy_head


class MCTS:
    def __init__(self, settings):
        self.settings = settings

    def run(self, model, state, player_turn):
        root = Node(0, player_turn, state)
        render(root.state)

        # value, action_p = model.predict(state)
        value, action_p = dummy_model_predict(state)
        legal_actions = get_legal_moves(state)
        action_p = action_p * legal_actions
        action_p /= np.sum(action_p)  # To be checked
        root.expand(action_p)

        for _ in range(self.settings['num_simulations']):
            node = root
            path = [node]

            # Select node
            while node.is_expanded():
                action, node = node.select_child()
                path.append(node)

            value = None
            if is_draw(node.state):
                value = 0
            if is_win(node.state, player_turn):
                value = 1
            if is_win(node.state, -player_turn):
                value = -1

            if value is None:
                # value, action_p = model.predict(node.state)
                value, action_p = dummy_model_predict(node.state)
                legal_actions = get_legal_moves(node.state)
                action_p = action_p * legal_actions  # avoid illegal moves
                action_p /= np.sum(action_p)  # To be checked
                node.expand(action_p)

            self.rollout(path, value, -player_turn)

        return root

    def rollout(self, path, value, player_turn):
        for n in path:
            n.value += value if n.player_turn == player_turn else -value
            n.visits += 1
