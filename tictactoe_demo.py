from envs.tictactoe import TicTacToeGame
from envs.mcts import MCTS
import envs.gaming as gaming

import time as t

game = TicTacToeGame(size=3, n_players=2)
agent1 = MCTS(game, n_plays=50, max_depth=500, player=1)
agent2 = gaming.RandomStrategy(game, player=2)

games = 50  # games from each "side"

count_p1 = 0
count_p2 = 0

t1 = t.time()

play_order = [agent1, agent2]
for _ in range(games * 2):
    state, rewards, turn, log = gaming.play_game(game, play_order, max_turns=50)

    winner = [p for p, r in rewards.items() if r == 1]
    if winner == [1]:
        count_p1 += 1
    elif winner == [2]:
        count_p2 += 1

    print()
    print(f"the winner is the player {winner}, turn: {turn+1}")
    print(state)
    print(log)

    # Flip the order for equal play
    play_order = play_order[::-1]

t2 = t.time()

print()
print(f"win rate of player1 is: {round(count_p1/(count_p1+count_p2), 3)}")
print(f"total play time: {round(t2-t1, 3)}")
