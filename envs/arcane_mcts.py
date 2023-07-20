import math
from typing import List

import numpy as np

import envs.gaming as gaming

from sklearn.tree import DecisionTreeRegressor


class UcbNode:
    def __init__(self, state, action, parent):
        self.parent: UcbNode = parent
        self.state = state
        self.action = action
        self.children: List[UcbNode] = list()
        self.n_sim = 0
        self.reward = 0
        self.c = math.sqrt(2)
        self.sf = np.finfo(np.float).tiny

    def add_child(self, state, action):
        child = UcbNode(state, action, self)
        self.children.append(child)

    def get_score(self, regressor):
        """Upper-confidence bound (UCT) + heuristic"""

        try:
            X = np.array([self.state]).reshape(1, -1)
            heuristic = regressor.predict(X)
        except:
            heuristic = 0

        base = (self.reward + 1) / (self.n_sim + 1)
        exp = math.sqrt(math.log(self.parent.n_sim + 1, math.e) / (self.n_sim + 1))
        return heuristic + base + self.c * exp

    def select_child(self, regressor):
        ucts = np.array([e.get_score(regressor) for e in self.children])
        pos = np.flatnonzero(ucts == max(ucts))
        selected_child = np.random.choice(pos, 1)[0]
        return self.children[selected_child]

    def update_stats(self, ds, reward):
        self.n_sim += 1
        self.reward += reward

        ds.append((self.state, reward))

        if self.parent is not None:
            self.parent.update_stats(ds, reward)


class MCTS(gaming.PlayerPolicy):
    def __init__(self, game: gaming.Game, n_plays: int, player: int, max_depth=500):
        self.n_plays = n_plays
        self.max_depth = max_depth
        self.game = game
        self.player = player
        self.regressor = DecisionTreeRegressor()
        self.ds = []

    def __call__(self, root_state):
        root = UcbNode(root_state, None, None)

        for _ in range(self.n_plays):
            current_node = root
            current_state = self.game.clone(root_state)
            for _ in range(self.max_depth):
                ssrd = self.perform_action(current_node, current_state, self.player)
                current_node, current_state, reward, done = ssrd

                current_node.update_stats(self.ds, reward)
                self.update_regressor()

                if done:
                    break

                for adversary in range(1, self.game.get_player_count() + 1):
                    if adversary != self.player:
                        ssrd = self.perform_action(
                            current_node, current_state, adversary
                        )
                        current_node, current_state, reward, done = ssrd

                        reward = reward * -1
                        current_node.update_stats(self.ds, reward)
                        self.update_regressor()
                        if done:
                            break
                if done:
                    break

        best_action = root.select_child(self.regressor).action

        return best_action

    def perform_action(self, node, state, player):
        if len(node.children) == 0:
            actions = self.game.get_possible_actions(state, player)
            for action in actions:
                node.add_child(state, action)

        if len(node.children) != 0:
            node = node.select_child(self.regressor)
            state, reward, done = self.game.get_result_state(state, node.action, player)
            node.state = state
            return node, state, reward, done
        else:
            return node, state, 0, False

    def get_player(self) -> int:
        return self.player

    def update_regressor(self):
        # limit the size of the dataset to the last 1000
        self.ds = self.ds[-1000:]

        # update the regressor
        X = np.array([item[0] for item in self.ds]).reshape(len(self.ds), -1)
        y = np.array([item[1] for item in self.ds])
        self.regressor.fit(X, y)
