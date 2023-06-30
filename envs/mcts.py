import math
from typing import List

import numpy as np

import envs.gaming as gaming


class UcbNode:
    def __init__(self, action, parent):
        self.parent: UcbNode = parent
        self.action = action
        self.children: List[UcbNode] = list()
        self.n_sim = 0
        self.reward = 0
        self.c = math.sqrt(2)
        self.sf = np.finfo(np.float).tiny

    def add_child(self, action):
        child = UcbNode(action, self)
        self.children.append(child)

    def get_score(self):
        # Upper-confidence bound (UCT)
        base = (self.reward + 1) / (self.n_sim + 1)
        exp = math.sqrt(math.log(self.parent.n_sim + 1, math.e) / (self.n_sim + 1))
        return base + self.c * exp

    def select_child(self):
        ucts = np.array([e.get_score() for e in self.children])
        pos = np.flatnonzero(ucts == max(ucts))
        selected_child = np.random.choice(pos, 1)[0]
        return self.children[selected_child]

    def update_stats(self, reward):
        self.n_sim += 1
        self.reward += reward

        if self.parent is not None:
            self.parent.update_stats(reward)


class MCTS(gaming.PlayerPolicy):
    def __init__(self, game: gaming.Game, n_plays: int, player: int, max_depth=500):
        self.n_plays = n_plays
        self.max_depth = max_depth
        self.game = game
        self.player = player

    def __call__(self, root_state):
        root = UcbNode(None, None)

        for i in range(self.n_plays):
            current_node = root
            current_state = self.game.clone(root_state)
            for j in range(self.max_depth):
                ssrd = self.perform_action(current_node, current_state, self.player)
                current_node, current_state, reward, done = ssrd

                current_node.update_stats(reward)
                if done:
                    break

                for adversary in range(1, self.game.get_player_count()):
                    if adversary != self.player:
                        ssrd = self.perform_action(
                            current_node, current_state, adversary
                        )
                        current_node, current_state, reward, done = ssrd

                        reward = reward * -1
                        current_node.update_stats(reward)
                        if done:
                            break
                if done:
                    break

        best_action = root.select_child().action

        return best_action

    def perform_action(self, node, state, player):
        if len(node.children) == 0:
            actions = self.game.get_possible_actions(state, player)
            for action in actions:
                node.add_child(action)

        if len(node.children) != 0:
            node = node.select_child()
            state, reward, done = self.game.get_result_state(state, node.action, player)
            return node, state, reward, done
        else:
            return node, state, 0, False

    def get_player(self) -> int:
        return self.player
