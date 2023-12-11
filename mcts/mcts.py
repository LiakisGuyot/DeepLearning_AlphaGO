import numpy as np
import torch

from game.game import Game
from mcts.node import Node
from neuralnetwork.model import Model
from settings import DIRICHLET_EPSILON, DIRICHLET_ALPHA, SEARCHES


class MCTS:
    def __init__(self, game: Game, model: Model):
        self.game = game
        self.model = model

    @torch.no_grad()
    def search(self, states: np.ndarray, sp_games: list):
        """
        Search best action by predicting policy, value and expanding tree.
        Update root values...
        Search parallelized (multiple games running at same time).
        4 main steps :
        1. Selection
        2. Expansion
        3. Policy & value prediction
        4. Back propagation
        :param states:
        :param sp_games:
        :return:
        """
        policy, _ = self.model(
            torch.tensor(self.game.get_encoded_state(states), device=self.model.device)
        )
        policy = torch.softmax(policy, dim=1).cpu().numpy()
        policy = ((1 - DIRICHLET_EPSILON) * policy + DIRICHLET_EPSILON
                  * np.random.dirichlet([DIRICHLET_ALPHA] * self.game.action_size, size=policy.shape[0]))

        for i, spg in enumerate(sp_games):
            spg_policy = policy[i]
            legal_moves = self.game.get_legal_moves(states[i])
            spg_policy *= legal_moves
            spg_policy /= np.sum(policy)

            spg.root = Node(self.game, states[i], visits=1)
            spg.root.expand(spg_policy)

        for search in range(SEARCHES):
            for spg in sp_games:
                spg.node = None
                node = spg.root

                while node.is_expanded():
                    node = node.select()

                value, is_ended = self.game.get_value_and_ended(node.state, node.action)
                value = self.game.get_opponent_value(value)

                if is_ended:
                    node.back_propagate(value)
                else:
                    spg.node = node

            expandable_sp_games = [map_i for map_i in range(len(sp_games)) if sp_games[map_i].node is not None]
            value = None
            if len(expandable_sp_games) > 0:
                states = np.stack(
                    [sp_games[map_i].node.state for map_i in expandable_sp_games]
                )
                policy, value = self.model(
                    torch.tensor(self.game.get_encoded_state(states), device=self.model.device)
                )
                policy = torch.softmax(policy, dim=1).cpu().numpy()
                value = value.cpu().numpy()

            for i, map_i in enumerate(expandable_sp_games):
                node = sp_games[map_i].node
                spg_policy, spg_value = policy[i], value[i]
                legal_moves = self.game.get_legal_moves(node.state)
                spg_policy *= legal_moves
                spg_policy /= np.sum(spg_policy)

                node.expand(spg_policy)
                node.back_propagate(spg_value)
