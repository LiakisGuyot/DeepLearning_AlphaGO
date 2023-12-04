import random

from connect4.game import Game
from connect4.state import State
from mcts.mcts import MCTS
from mcts.node import Node

if __name__ == '__main__':
    tree = MCTS()
    game = Game()
    game.state.print()
    while True:
        # action = input(f"Choose action in {game.state.get_legal_actions()} :")
        action = random.choice(game.state.get_legal_actions())
        next_state, is_ended = game.state.apply_action(int(action))
        game.state = next_state
        game.state.print()
        if game.state.is_ended():
            break

        for _ in range(1000):
            tree.do_rollout(Node(game.state))
        best = tree.choose(Node(game.state))
        game.state = best.state
        best.state.print()
        if best.is_terminal():
            break
