from game.game import Game


class SelfPlayGame:
    def __init__(self, game: Game):
        self.state = game.get_init_state()
        self.memory = []
        self.root = None
        self.node = None
