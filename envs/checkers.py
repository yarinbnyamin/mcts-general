import numpy as np

import envs.gaming as gaming
import envs.mcts as mcts


class CheckersGame(gaming.Game):
    def __init__(self, size_x, size_y):
        self.size_x = size_x
        self.size_y = size_y

    def get_possible_actions(self, state, player):
        size_x, size_y = state.shape
        max_shift = min(state.shape)
        figure_pos = np.argwhere(state // 10 == player)
        moves = []
        base_dir = 1 if player == 1 else -1

        for pos_x, pos_y in figure_pos:
            if state[(pos_x, pos_y)] % 10 == 0:  # normal figures
                for direction in [-1, 1]:
                    action = [(pos_x, pos_y)]
                    for shift in range(1, max_shift):
                        move_x, move_y = (
                            pos_x + shift * base_dir,
                            pos_y + direction * shift,
                        )
                        if (
                            move_x >= size_x
                            or move_y >= size_y
                            or move_x < 0
                            or move_y < 0
                        ):
                            break
                        elif state[move_x, move_y] // 10 == player:
                            break
                        elif state[move_x, move_y] != 0:
                            action.append((move_x, move_y))
                        else:
                            action.append((move_x, move_y))
                            moves.append(action)
                            break
            else:  # kings
                for dir_x in [-1, 1]:
                    for dir_y in [-1, 1]:
                        action = [(pos_x, pos_y)]
                        for shift in range(1, max_shift):
                            move_x, move_y = pos_x + dir_x * shift, pos_y + dir_y
                            if (
                                move_x >= size_x
                                or move_y >= size_y
                                or move_x < 0
                                or move_y < 0
                            ):
                                break
                            elif state[move_x, move_y] // 10 == player:
                                break
                            elif state[move_x, move_y] != 0:
                                action.append((move_x, move_y))
                            else:
                                action.append((move_x, move_y))
                                moves.append(action.copy())

        return moves

    def get_result_state(self, state, action, player):
        new_state = state.copy()

        for pos_x, pos_y in action[1:-1]:
            new_state[pos_x, pos_y] = 0

        new_state[action[-1]] = state[action[0]]
        new_state[action[0]] = 0

        if player == 1:
            if action[-1][0] == (state.shape[0] - 1):
                new_state[action[-1]] = 11
        else:
            if action[-1][0] == 0:
                new_state[action[-1]] = 21

        winner = self.get_winner(new_state)
        return new_state, winner == player, winner != -1

    def get_player_count(self):
        return 2

    def get_initial_state(self):
        initial_state = np.zeros((self.size_x, self.size_y), dtype=np.byte)

        for i in range(3):
            for j in range(0, self.size_x, 2):
                initial_state[i, j + i % 2] = 10

        for i in range(self.size_y - 3, self.size_y):
            for j in range(0, self.size_x, 2):
                initial_state[i, j + i % 2] = 20

        return initial_state

    def get_winner(self, state) -> int:
        if np.sum(state // 10 == 1) == 0:
            return 2
        elif np.sum(state // 10 == 2) == 0:
            return 1
        else:
            return -1


def test_play():
    ttt = CheckersGame(8, 8)
    s1 = mcts.MCTS(ttt, n_plays=20, max_depth=500, player=1)
    s2 = gaming.RandomStrategy(ttt, player=2)

    state, rewards, turn, log = gaming.play_game(ttt, [s1, s2], max_turns=100)
    print()
    print(
        f"the winner is the player {[p for p, r in rewards.items() if r == 1]}, turn: {turn}"
    )
    print(state)
    print(log)


def test_initial_state():
    ttt = CheckersGame(8, 8)
    board = ttt.get_initial_state()
    print()
    print(board)


def test_simple_action():
    ttt = CheckersGame(8, 8)
    board = np.zeros((8, 8), dtype=np.byte)
    board[0, 2] = 10
    actions = ttt.get_possible_actions(board, 1)
    print()
    print(board)
    print(actions)
    res = ttt.get_result_state(board, actions[0], player=1)
    print(res)


def test_simple_action_king():
    ttt = CheckersGame(8, 8)
    board = np.zeros((8, 8), dtype=np.byte)
    board[0, 2] = 11
    actions = ttt.get_possible_actions(board, 1)
    print()
    print(board)
    print(actions)
    res = ttt.get_result_state(board, actions[0], player=1)
    print(res)


def test_simple_edge_action():
    ttt = CheckersGame(8, 8)
    board = np.zeros((8, 8), dtype=np.byte)
    board[0, 0] = 10
    actions = ttt.get_possible_actions(board, 1)
    print()
    print(board)
    print(actions)
    res = ttt.get_result_state(board, actions[0], player=1)
    print(res)


def test_capture_action():
    ttt = CheckersGame(8, 8)
    board = np.zeros((8, 8), dtype=np.byte)
    board[0, 0] = 10
    board[1, 1] = 20
    actions = ttt.get_possible_actions(board, 1)
    print()
    print(board)
    print(actions)
    res = ttt.get_result_state(board, actions[0], player=1)
    print(res)


def test_king_action():
    ttt = CheckersGame(8, 8)
    board = np.zeros((8, 8), dtype=np.byte)
    board[6, 2] = 10
    actions = ttt.get_possible_actions(board, 1)
    print()
    print(board)
    print(actions)
    res = ttt.get_result_state(board, actions[0], player=1)
    print(res)


def test_king_action_p2():
    ttt = CheckersGame(8, 8)
    board = np.zeros((8, 8), dtype=np.byte)
    board[1, 2] = 20
    actions = ttt.get_possible_actions(board, 2)
    print()
    print(board)
    print(actions)
    res = ttt.get_result_state(board, actions[0], player=2)
    print(res)


def test_win():
    ttt = CheckersGame(8, 8)
    board = np.zeros((8, 8), dtype=np.byte)
    board[0, 0] = 10
    res = ttt.get_winner(board)
    assert res == 1
    board[1, 1] = 20
    res = ttt.get_winner(board)
    assert res == -1
