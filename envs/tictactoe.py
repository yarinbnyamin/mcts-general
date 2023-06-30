import numpy as np

import envs.mcts as mcts
import envs.gaming as gaming


class TicTacToeGame(gaming.Game):
    def __init__(self, size_x: int, size_y: int, len_to_win: int, n_players: int):
        self.size_x = size_x
        self.size_y = size_y
        self.len_to_win = len_to_win
        self.n_players = n_players

    def get_possible_actions(self, state, player):
        return [tuple(e) for e in np.argwhere(state == 0)]

    def get_result_state(self, state, action, player):
        new_state = state.copy()
        new_state[action] = player

        winner = self.get_winner(new_state)
        return new_state, winner == player, winner != -1

    def get_player_count(self):
        return self.n_players

    def get_initial_state(self):
        return np.zeros((self.size_x, self.size_y), dtype=np.byte)

    def get_winner(self, state) -> int:
        size_max = max(state.shape)
        lines = []
        lines.extend(state[::-1, :].diagonal(i) for i in range(-size_max, size_max))
        lines.extend(state.diagonal(i) for i in range(-size_max, size_max))
        lines.extend(state[i] for i in range(len(state)))
        lines.extend(state.T[i] for i in range(len(state)))

        for line in lines:
            if len(line) < self.len_to_win:
                continue

            for i, j in zip(range(len(line)), range(self.len_to_win, len(line) + 1)):
                for player in range(1, self.n_players + 1):
                    if (
                        sum(line[i:j] == np.full(self.len_to_win, player))
                        >= self.len_to_win
                    ):
                        return player

        if np.sum(state == 0) == 0:
            return 0
        else:
            return -1


def test_play():
    ttt = TicTacToeGame(size_x=4, size_y=4, len_to_win=3, n_players=2)
    s1 = mcts.MCTS(ttt, n_plays=50, max_depth=500, player=1)
    s2 = gaming.RandomStrategy(ttt, player=2)

    state, rewards, turn, log = gaming.play_game(ttt, [s1, s2], max_turns=50)
    print()
    print(
        f"the winner is the player {[p for p, r in rewards.items() if r == 1]}, turn: {turn}"
    )
    print(state)
    print(log)

    state, rewards, turn, log = gaming.play_game(ttt, [s1, s2], max_turns=50)
    print()
    print(
        f"the winner is the player {[p for p, r in rewards.items() if r == 1]}, turn: {turn}"
    )
    print(state)
    print(log)


def test_possible_actions():
    ttt = TicTacToeGame(size_x=4, size_y=4, len_to_win=3, n_players=2)
    board = np.ones((4, 4), dtype=np.byte)
    board[0, 0] = 0
    board[1, 1] = 0
    actions = ttt.get_possible_actions(board, 1)
    print(actions)


def test_in_progress():
    ttt = TicTacToeGame(size_x=3, size_y=3, len_to_win=3, n_players=2)
    board = np.zeros((3, 3), dtype=np.byte)
    winner = ttt.get_winner(board)
    assert winner == -1


def test_draw():
    ttt = TicTacToeGame(size_x=3, size_y=3, len_to_win=3, n_players=2)
    board = np.ones((3, 3), dtype=np.byte)
    board[1, 1] = 2
    board[2, 2] = 2
    board[0, 1] = 2
    board[0, 2] = 2
    board[1, 0] = 2
    print()
    print(board)
    winner = ttt.get_winner(board)
    assert winner == 0


def test_win():
    ttt = TicTacToeGame(size_x=4, size_y=4, len_to_win=3, n_players=2)
    board = np.ones((4, 4), dtype=np.byte)
    for i in range(0, 4, 2):
        for j in range(0, 4, 2):
            board[i, j] = 2
    print()
    print(board)
    winner = ttt.get_winner(board)
    assert winner == 1


def test_win_diag():
    ttt = TicTacToeGame(size_x=4, size_y=4, len_to_win=3, n_players=2)
    board = np.zeros((4, 4), dtype=np.byte)
    board[0, 0] = 1
    board[1, 1] = 1
    board[2, 2] = 1
    board[0, 1] = 2
    board[0, 2] = 2
    winner = ttt.get_winner(board)
    assert winner == 1
