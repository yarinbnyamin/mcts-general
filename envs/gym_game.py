import warnings
import gym

import envs.gaming as gaming
from envs.mcts import MCTS


class GymGame(gaming.Game):
    def __init__(self, env_name: str, render=False):
        self.env_name = env_name
        self.render = render

    def get_initial_state(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            env = gym.make(self.env_name)

        if self.render:
            from gym.wrappers import RecordVideo

            env = RecordVideo(env, video_folder="videos/")

        env.seed(1)
        state = env.reset()
        return env, state, list()

    def get_possible_actions(self, state, player):
        env, _, _ = state
        return list(range(env.action_space.n))

    def get_player_count(self):
        return 1

    def get_result_state(self, state, action, player):
        env, _, actions = state
        inner_state, reward, done, _ = env.step(action)
        actions = actions.copy()
        actions.append(action)
        return (env, inner_state, actions), reward, done

    def clone(self, state):
        _, _, actions = state
        actions = actions.copy()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            env = gym.make(self.env_name)
        env.seed(1)
        env.reset()

        for a in actions:
            state, reward, done, _ = env.step(a)
            if done:
                raise Exception(
                    "cloning environment which is considered done for current action list"
                )
        return env, state, actions


class GymNStepsGame(GymGame):
    def __init__(self, env_name: str, render=False):
        super().__init__(env_name, render)

    def get_result_state(self, state, action, player):
        state, reward, done = super().get_result_state(state, action, player)
        reward = (
            -10 if done else 0
        )  # aggressive penalty for loss to reduce exploration level
        return state, reward, done


def test_play():
    ttt = GymNStepsGame("CartPole-v1")
    s1 = MCTS(ttt, n_plays=50, max_depth=30, player=1)

    ttt_play = GymNStepsGame("CartPole-v1")
    state, rewards, steps, log = gaming.play_game(ttt_play, [s1], max_turns=50)
    print()
    print(f"steps: {steps}")
    print(rewards)
    print([a for p, a in log])
