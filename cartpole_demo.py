import envs.gaming as gaming
from envs.gym_game import GymNStepsGame
from envs.mcts import MCTS

game = GymNStepsGame("CartPole-v1", render=True)
agent1 = MCTS(game, n_plays=2, max_depth=50, player=1)

state, rewards, steps, log = gaming.play_game(game, [agent1], max_turns=500)
print()
print(f"steps: {steps}")
print(rewards)
print([a for p, a in log])
