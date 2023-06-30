import numpy as np

import envs.gaming as gaming
from envs.mcts import MCTS


class GuessNumber(gaming.Game):
    def __init__(self, max_number=10, n_players=1):
        self.max_number = max_number
        self.n_players = n_players
        self.target = int(np.random.random() * max_number) + 1

    def get_possible_actions(self, state, player):
        return list(range(1, self.max_number + 1))

    def get_result_state(self, state, action, player):
        state = state.copy()

        if action == self.target:
            state[player] = 0
        elif action < self.target:
            state[player] = -1
        else:
            state[player] = 1

        winner = self.get_winner(state)
        return state, winner == player, winner != -1

    def get_player_count(self):
        return self.n_players

    def get_initial_state(self):
        return dict((p, None) for p in range(1, self.n_players + 1))

    def get_winner(self, state) -> int:
        for player, state in state.items():
            if state == 0:
                return player
        return -1


def test_mcts_play():
    ttt = GuessNumber(n_players=2)
    s1 = MCTS(ttt, n_plays=50, max_depth=4, player=1)
    s2 = gaming.RandomStrategy(ttt, player=2)

    state, rewards, turn, log = gaming.play_game(ttt, [s1, s2], max_turns=50)
    print()
    print(
        f"the winner is the player {[p for p, r in rewards.items() if r == 1]}, turn: {turn}"
    )
    print(state)
    print(log)


def test_mcts_play_1player():
    ttt = GuessNumber(n_players=1)
    s1 = MCTS(ttt, n_plays=50, max_depth=20, player=1)

    state, rewards, turn, log = gaming.play_game(ttt, [s1], max_turns=50)
    print()
    print(
        f"the winner is the player {[p for p, r in rewards.items() if r == 1]}, turn: {turn}"
    )
    print(state)
    print(log)
