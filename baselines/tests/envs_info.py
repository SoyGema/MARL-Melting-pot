import argparse
from gymnasium.spaces import discrete
from meltingpot import substrate
from meltingpot.configs.substrates import prisoners_dilemma_in_the_matrix__arena
from meltingpot.configs import substrates
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from baselines.train import make_envs


class MeltingPotEnvInformation():
  """Prints out some infomation about the MeltingPotEnv"""

  def __init__(self , substrate_name):
      env_config = substrate.get_config(substrate_name)
      roles = env_config.default_player_roles
      self._num_players = len(roles)
      self._env = make_envs.env_creator({
          'substrate': substrate_name,
          'roles': roles,
          'scaled': 8,
    })

  def action_space(self):
      """Test the action space."""

      actions_count = len(prisoners_dilemma_in_the_matrix__arena.ACTION_SET)
      env_action_space = self._env.action_space['player_1']
      print(f"Expected Discrete Action Space Size and number of players: {discrete.Discrete(actions_count)}")
      print(f"Action Space: {prisoners_dilemma_in_the_matrix__arena.ACTION_SET}")

  def observation_space(self):
    """Prints the observation space."""

    env_obs_space = self._env.observation_space['player_1']
    print(f"Observation Space for player_1: {env_obs_space}") 


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Print information about a MeltingPotEnv")
    parser.add_argument('substrate_name', type=str, help="Name of the substrate for which to print information")
    args = parser.parse_args()

    env_info = MeltingPotEnvInformation(args.substrate_name)

  # For printing environment information
    env_info.action_space()
    env_info.observation_space()