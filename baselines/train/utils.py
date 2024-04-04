"""Shared utils for third-party library examples."""

from typing import Any, Mapping

import dm_env
import numpy as np
import cv2
import tree
from gymnasium import spaces
from gymnasium.spaces import Discrete
import logging
import sys

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)])


PLAYER_STR_FORMAT = 'player_{index}'
_IGNORE_KEYS = ['WORLD.RGB', 'INTERACTION_INVENTORIES', 'NUM_OTHERS_WHO_CLEANED_THIS_STEP']


def downsample_observation(array: np.ndarray, scaled) -> np.ndarray:
    """Downsample image component of the observation.
    Args:
      array: RGB array of the observation provided by substrate
      scaled: Scale factor by which to downsaple the observation
    
    Returns:
      ndarray: downsampled observation  
    """
 
    original_shape = array.shape
    
    frame = cv2.resize(
            array, (array.shape[0]//scaled-2, array.shape[1]//scaled-2), interpolation=cv2.INTER_AREA)
    new_shape = frame.shape
    #logging.debug(f"downsample_observation: Original shape {original_shape}, New shape {new_shape}")
    # Called during training therefore commented (88,88,3)
    return frame

def timestep_to_observations(timestep: dm_env.TimeStep) -> Mapping[str, Any]:
  """Extract observation from timestep structure returned from substrate."""

  gym_observations = {}
  for index, observation in enumerate(timestep.observation):
    gym_observations[PLAYER_STR_FORMAT.format(index=index)] = {
        key: value
        for key, value in observation.items()
        if key not in _IGNORE_KEYS
    }
  #logging.debug(f"timestep_to_observations: Extracted keys {list(gym_observations.keys())}")
  # Here is given the number of players.
  return gym_observations


def remove_unrequired_observations_from_space(
    observation: spaces.Dict) -> spaces.Dict:
  """Remove observations that are not supposed to be used by policies."""

  original_keys = list(observation.keys())
  filtered_space = spaces.Dict({
        key: observation[key] for key in observation if key not in _IGNORE_KEYS
    })
  filtered_keys = list(filtered_space.keys())
  #logging.debug(f"remove_unrequired_observations_from_space: Original keys {original_keys}, Filtered keys {filtered_keys}")
  ## things that for whatever reason we are not using in the policy, in this case 'COLLECTIVE REWARD','INTERACTION_INVENTORIES',ETC
  return filtered_space



def spec_to_space(spec: tree.Structure[dm_env.specs.Array]) -> spaces.Space:
  """Converts a dm_env nested structure of specs to a Gym Space.

  BoundedArray is converted to Box Gym spaces. DiscreteArray is converted to
  Discrete Gym spaces. Using Tuple and Dict spaces recursively as needed.

  Args:
    spec: The nested structure of specs

  Returns:
    The Gym space corresponding to the given spec.
  """
  if isinstance(spec, dm_env.specs.DiscreteArray):
    return Discrete(spec.num_values)
  elif isinstance(spec, dm_env.specs.BoundedArray):
    return spaces.Box(spec.minimum, spec.maximum, spec.shape, spec.dtype)
  elif isinstance(spec, dm_env.specs.Array):
    if np.issubdtype(spec.dtype, np.floating):
      return spaces.Box(-np.inf, np.inf, spec.shape, spec.dtype)
    elif np.issubdtype(spec.dtype, np.integer):
      info = np.iinfo(spec.dtype)
      return spaces.Box(info.min, info.max, spec.shape, spec.dtype)
    else:
      raise NotImplementedError(f'Unsupported dtype {spec.dtype}')
  elif isinstance(spec, (list, tuple)):
    return spaces.Tuple([spec_to_space(s) for s in spec])
  elif isinstance(spec, dict):
    return spaces.Dict({key: spec_to_space(s) for key, s in spec.items()})
  else:
    raise ValueError('Unexpected spec of type {}: {}'.format(type(spec), spec))
