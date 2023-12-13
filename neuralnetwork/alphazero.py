import os.path
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Optimizer
from tqdm import trange

from game.game import Game
from game.spg import SelfPlayGame
from mcts.mcts import MCTS
from neuralnetwork.model import Model
from notif.notificationbot import NotificationBot
from settings import SELF_PLAY_ITERATIONS, ITERATIONS, PARALLEL_GAMES, EPOCHS, DIRICHLET_EPSILON, DIRICHLET_ALPHA, \
    SEARCHES, C, TEMPERATURE, BATCH_SIZE


def save(file_name: str, model: Model, optimizer: Optimizer, losses: dict):
    """
    Save the model with settings, optimizer and losses.
    :param file_name:
    :param model:
    :param optimizer:
    :param losses:
    :return:
    """
    model_file = {
        "settings": {
            "DIRICHLET_EPSILON": DIRICHLET_EPSILON,
            "DIRICHLET_ALPHA": DIRICHLET_ALPHA,
            "SEARCHES": SEARCHES,
            "C": C,
            "TEMPERATURE": TEMPERATURE,
            "BATCH_SIZE": BATCH_SIZE,
            "ITERATIONS": ITERATIONS,
            "SELF_PLAY_ITERATIONS": SELF_PLAY_ITERATIONS,
            "PARALLEL_GAMES": PARALLEL_GAMES,
            "EPOCHS": EPOCHS
        },
        "model_state_dict": model.state_dict(),
        "optim_state_dict": optimizer.state_dict(),
        "losses": losses
    }
    torch.save(model_file, file_name)


class AlphaZero:
    def __init__(self, model: Model, optimizer: Optimizer, game: Game):
        self.model = model
        self.optimizer = optimizer
        self.game = game

        self.mcts = MCTS(game, model)
        self.losses = {}

    def self_play(self):
        """
        Self-play the AI.
        Get data for training...
        :return:
        """
        return_memory = []
        player = 1
        sp_games = [SelfPlayGame(self.game) for spg in range(PARALLEL_GAMES)]

        while len(sp_games) > 0:
            states = np.stack([spg.state for spg in sp_games])

            neutral_states = self.game.switch_state_perspective(states, player)
            self.mcts.search(neutral_states, sp_games)

            for i in range(len(sp_games))[::-1]:
                spg = sp_games[i]

                action_probs = np.zeros(self.game.action_size)
                for child in spg.root.children:
                    action_probs[child.action] = child.visits
                action_probs /= np.sum(action_probs)

                spg.memory.append(
                    (spg.root.state, action_probs, player)
                )

                temperature_action_probs = action_probs ** (1 / TEMPERATURE)
                temperature_action_probs /= np.sum(temperature_action_probs)
                action = np.random.choice(self.game.action_size, p=temperature_action_probs)

                spg.state = self.game.apply_action(state=spg.state, player=player, action=action)
                value, is_ended = self.game.get_value_and_ended(spg.state, action)
                if is_ended:
                    for hist_neutral_state, hist_action_probs, hist_player in spg.memory:
                        hist_outcome = value if hist_player == player else self.game.get_opponent_value(value)
                        return_memory.append((
                            self.game.get_encoded_state(hist_neutral_state),
                            hist_action_probs,
                            hist_outcome
                        ))
                    del sp_games[i]

            player = self.game.get_opponent(player)

        return return_memory

    def train(self, memory: list, idx: int):
        """
        Train with datas obtained with self-playing
        :param memory:
        :param idx:
        :return:
        """
        random.shuffle(memory)
        _losses = []
        for batch_i in range(0, len(memory), BATCH_SIZE):
            sample = memory[batch_i:min(len(memory)-1, batch_i+BATCH_SIZE)]
            state, policy_targets, value_targets = zip(*sample)
            state, policy_targets, value_targets = (np.array(state),
                                                    np.array(policy_targets),
                                                    np.array(value_targets).reshape(-1, 1))

            state = torch.tensor(state, dtype=torch.float32, device=self.model.device)
            policy_targets = torch.tensor(policy_targets, dtype=torch.float32, device=self.model.device)
            value_targets = torch.tensor(value_targets, dtype=torch.float32, device=self.model.device)

            out_policy, out_value = self.model(state)

            policy_loss = F.cross_entropy(out_policy, policy_targets)
            value_loss = F.mse_loss(out_value, value_targets)
            loss = policy_loss + value_loss
            _losses.append(loss.item())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        self.losses[idx] = _losses

    def learn(self, bot: NotificationBot = None):
        """
        Learn process.
        Bot inclusion (PushOver).
        :param bot:
        :return:
        """
        start = datetime.now()
        if bot is not None:
            bot.send_notification("Starting learning", f"Learning with ITERATIONS = {ITERATIONS} "
                                                       f"& SP ITERATIONS = {SELF_PLAY_ITERATIONS} "
                                                       f"& PARALLEL_GAMES = {PARALLEL_GAMES} "
                                                       f"& EPOCHS = {EPOCHS}...")
        for iteration in range(ITERATIONS):
            memory = []
            self.losses = {}

            self.model.eval()
            for self_play_iteration in trange(SELF_PLAY_ITERATIONS // PARALLEL_GAMES):
                memory += self.self_play()
                if bot is not None:
                    bot.send_notification("Self-play", "SelfPlay iteration "
                                                       f"{self_play_iteration+1}/{SELF_PLAY_ITERATIONS//PARALLEL_GAMES} done...")

            self.model.train()
            for epoch in trange(EPOCHS):
                self.train(memory, epoch)
                
            self.create_directories()
            save(f"run/{self.game}/model_{iteration}.pth", self.model, self.optimizer, self.losses)
            if bot is not None:
                bot.send_notification("Main iteration", f"Main iteration {iteration+1}/{ITERATIONS} done...")
        end = datetime.now()
        if bot is not None:
            bot.send_notification("Learning done", f"Learning just finished, took {end-start}")

    def create_directories(self):
        """
        Create directories to save models.
        :return:
        """
        if not os.path.exists("run"):
            os.mkdir("run")
        if not os.path.exists(f"run/{self.game}"):
            os.mkdir(f"run/{self.game}")
