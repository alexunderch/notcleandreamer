from typing import Dict, List, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
import optax
from flax.core.frozen_dict import FrozenDict
from flax.training.dynamic_scale import DynamicScale
from flax.training.train_state import TrainState as FlaxTrainState

BaseDataType = NamedTuple
base_jnp_type = jnp.float16


class RSSMState(BaseDataType):
    logits: jax.Array = None
    stochastic_state: jax.Array = None
    deterministic_state: jax.Array = None

    def get_state(self) -> jax.Array:
        # NOTE: unbatched!
        h = self.deterministic_state
        z = jnp.reshape(self.stochastic_state, (-1,))
        return jnp.concatenate([h, z], -1)


class TrainState(BaseDataType):
    params: FrozenDict
    opt_state: optax.OptState
    target_params: Optional[FrozenDict] = None


class DreamerTrainState(FlaxTrainState):
    aux: RSSMState
    dynamic_scale: DynamicScale


class Observation(BaseDataType):
    agent_observation: jax.Array
    global_observation: jax.Array
    rng: jax.Array = None
    action_mask: Optional[jax.Array] = None


class CriticOutput(BaseDataType):
    memory: Tuple[jax.Array, jax.Array] | jax.Array
    value: jax.Array


class ActorOutput(BaseDataType):
    memory: Tuple[jax.Array, jax.Array] | jax.Array
    logits: jax.Array
    action: jax.Array
    log_prob: jax.Array
    entropy: jax.Array


class Transition(BaseDataType):
    """Transition tuple for trajectories"""

    state: Optional[jax.Array] = None
    observation: jax.Array = None
    termination: jax.Array = None
    action: jax.Array = None
    value: jax.Array = None
    reward: jax.Array = None
    log_prob: jax.Array = None
    info: List[Dict] = None
    action_mask: Optional[jax.Array] = None
    is_first: Optional[jax.Array] = None


class ACLossInfo(BaseDataType):
    """Losses computed in training"""

    total_loss: jax.Array = None
    critic_loss: jax.Array = None
    actor_loss: jax.Array = None
    approx_kl: jax.Array = None
    entropy: jax.Array = None
    actor_grad_norm: jax.Array = None
    critic_grad_norm: jax.Array = None
    grad_norm: jax.Array = None


class RSSMLossInfo(BaseDataType):
    loss: jax.Array = None
    rep_loss: jax.Array = None
    dyn_loss: jax.Array = None
    prior_entropy: jax.Array = None
    posterior_entropy: jax.Array = None
    log_p_o: jax.Array = None
    log_p_a: jax.Array = None
    log_p_r: jax.Array = None
    log_p_d: jax.Array = None
    grad_norm: jax.Array = None
