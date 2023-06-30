from envs.tictactoe import TicTacToeGame
from envs.mcts import MCTS
import envs.gaming as gaming

game = TicTacToeGame(size_x=4, size_y=4, len_to_win=4, n_players=2)
agent1 = MCTS(game, n_plays=50, max_depth=500, player=1)
agent2 = gaming.RandomStrategy(game, player=2)

state, rewards, turn, log = gaming.play_game(game, [agent1, agent2], max_turns=50)
print()
print(
    f"the winner is the player {[p for p, r in rewards.items() if r == 1]}, turn: {turn}"
)
print(state)
print(log)

state, rewards, turn, log = gaming.play_game(game, [agent1, agent2], max_turns=50)
print()
print(
    f"the winner is the player {[p for p, r in rewards.items() if r == 1]}, turn: {turn}"
)
print(state)
print(log)
