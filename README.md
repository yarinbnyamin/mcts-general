This project is forked from [dllllb/imaginarium](https://github.com/dllllb/imaginarium) and focus on Monte-Carlo Tree Search for board games and for the deterministic OpenAI Gym environments.

<p align="center">
  <img src="cartpole-mcts-demo.gif" width="300" />
</p>

To improve the MCTS algorithm (envs/mcts.py) change the function get_score (in line 23) to calculate differently the importance of each node. See envs/arcane_mcts.py for reference - adding a Regression Decision Tree as a heuristic.
