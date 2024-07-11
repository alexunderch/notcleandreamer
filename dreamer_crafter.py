from typing import Any, Callable, Tuple

import crafter
import gymnasium as gym
import hydra
import jax
import jax.numpy as jnp
import numpy as np
import supersuit as ss
import wandb
from gymnasium import make, wrappers
from omegaconf import DictConfig, OmegaConf

from noncleandreamer.ac import create_ac_dreamer as policy_fn
from noncleandreamer.custom_types import BaseDataType, Transition
from noncleandreamer.dreamer import DreamerAgent, create_item_buffer
from noncleandreamer.networks import (
    create_conv_rssm_model,
    create_discrete_lin_ppo_policy,
)
