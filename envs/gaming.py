import random
from typing import List, Tuple, Any
import numpy as np


class Game:
    def get_possible_actions(self, state, player) -> List[Any]:
        pass

    def get_result_state(self, state, action, player) -> Tuple[Any, int, bool]:
        return np.array(0), 0, False

    def get_player_count(self) -> int:
        return 1

    def get_initial_state(self):
        return np.array(0)

    def clone(self, state):
        return state


class PlayerPolicy:
    def __call__(self, state):
        pass

    def get_player(self) -> int:
        return 1


class RandomStrategy(PlayerPolicy):
    def __init__(self, game: Game, player: int):
        self.game = game
        self.player = player

    def __call__(self, state):
        actions = self.game.get_possible_actions(state, self.player)
        return actions[random.randint(0, len(actions) - 1)]

    def get_player(self) -> int:
        return self.player


def play_game(game: Game, strategies: List[PlayerPolicy], max_turns=1000):
    state = game.get_initial_state()

    action_log = []
    tatal_reward = dict((strategy.get_player(), 0) for strategy in strategies)

    turn = 0
    for turn in range(max_turns):
        for strategy in strategies:
            player = strategy.get_player()
            action = strategy(state)
            action_log.append((player, action))
            state, reward, done = game.get_result_state(state, action, player)

            tatal_reward[player] += reward

            if done:
                return state, tatal_reward, turn, action_log

    return state, tatal_reward, turn, action_log  # draw by turns limit
