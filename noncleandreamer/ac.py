import functools
from typing import Callable, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import rlax

from noncleandreamer.custom_types import (
    ActorOutput,
    BaseDataType,
    CriticOutput,
    Observation,
    Transition,
)
from noncleandreamer.distributions import Normalizer


class ACDreamer(nn.Module):

    actor: nn.Module
    critic: nn.Module
    slow_critic: nn.Module

    config: BaseDataType

    def setup(self) -> None:
        self.all_but_last = lambda x: jax.tree_map(lambda y: y[:-1], x)  # noqa: E731
        
        self.counter = self.variable("aux", "counter", jnp.zeros, (), jnp.float32)
        self.normalizer = Normalizer(decay=0.99, max_scale=1.0, q_low=5.0, q_high=95.0)

    @staticmethod
    @functools.partial(jax.jit, static_argnums=(2, 3))
    def _calculate_targets(
        traj_batch: Transition, values: jax.Array, gamma: float, lambda_: float
    ) -> jax.Array:
        targets = rlax.lambda_returns(
            traj_batch.reward.astype(jnp.float32),
            gamma * (1.0 - traj_batch.termination[1:]),
            values,
            lambda_,
        )
        return targets

    @staticmethod
    def get_weight(termination: jax.Array, gamma: float) -> jax.Array:
        weight = jnp.cumprod(gamma * termination, axis=0) / (gamma + 1e-8)
        return weight

    def act(
        self,
        latent: jax.Array,
        actor_memory: jax.Array,
        termination: jax.Array,
        action: jax.Array = None,
    ) -> ActorOutput:
        observation = Observation(latent, None)
        return self.actor(observation, action)

    def value(
        self, latent: jax.Array, critic_memory: jax.Array, termination: jax.Array
    ) -> CriticOutput:
        observation = Observation(latent, None)
        return self.critic(observation)

    def compute_advantages(self, traj_batch: Transition) -> Tuple[jax.Array, jax.Array]:

        values = self.value(traj_batch.observation, None, None).value.mean()
        unnormalised_targets = self._calculate_targets(
            traj_batch, values[1:], self.config.gamma, self.config.lambda_
        )
        unnormalised_values = values[:-1]

        if self.config.normalize_returns:
            offset, inv_scale = self.normalizer(unnormalised_targets)
            targets = (unnormalised_targets - offset) / inv_scale
            values = (unnormalised_values - offset) / inv_scale
        else:
            targets = unnormalised_targets
            values = unnormalised_values

        advantages = targets - values

        return advantages, unnormalised_values, unnormalised_targets

    def _actor_loss_fn(
        self,
        traj_batch: Transition,
        gae: jax.Array,
    ) -> Tuple:
        """Calculate the actor loss."""

        traj_obs = Observation(
            self.all_but_last(traj_batch.observation),
            self.all_but_last(traj_batch.state),
            None,
            self.all_but_last(traj_batch.action_mask),
        )
        actor_output = self.actor(traj_obs, self.all_but_last(traj_batch.action))
        log_prob = actor_output.log_prob

        gae = (gae - gae.mean())/ (gae.std() + 1e-8)


        loss_actor = -log_prob * jax.lax.stop_gradient(gae)
        entropy = actor_output.entropy
        weight = self.get_weight(
            self.all_but_last(1 - traj_batch.termination), self.config.gamma
        )
        total_loss_actor = weight * (loss_actor - self.config.ent_coeff * entropy)

        # jax.debug.print(
        #     "ACTOR {x}\t{y}\t{z}\t{t}",
        #     x=gae.mean(),
        #     y=log_prob.mean(),
        #     z=loss_actor.mean(),
        #     t=jnp.argmax(all_but_last(traj_batch.action), -1),
        # )
        # jax.debug.print("ACTOR {x}", x=jnp.argmax(all_but_last(traj_batch.action), -1))

        return total_loss_actor.mean(), (loss_actor.mean(), entropy.mean())

    def _critic_loss_fn(
        self,
        traj_batch: Transition,
        targets: jax.Array,
    ) -> Tuple:

        traj_obs = Observation(traj_batch.observation, traj_batch.state, None, None)

        values = self.critic(self.all_but_last(traj_obs)).value
        slow_values = self.slow_critic(self.all_but_last(traj_obs)).value.mean()


        values = (values - values.mean())/ (values.std() + 1e-8)
        slow_values = (slow_values - slow_values.mean())/ (slow_values.std() + 1e-8)
        targets = (targets - targets.mean())/ (targets.std() + 1e-8)

        # CALCULATE VALUE LOSS
        value_loss = -values.log_prob(jax.lax.stop_gradient(targets))
        reg = -values.log_prob(jax.lax.stop_gradient(slow_values))

        weight = self.get_weight(
            1 - self.all_but_last(traj_batch.termination), self.config.gamma
        )
        critic_total_loss = weight * (value_loss + self.config.vf_coeff * reg)
        return critic_total_loss.mean(), (value_loss.mean(),)


    def critic_loss(
        self, traj_batch: Transition, traj_values: jax.Array, targets: jax.Array
    ) -> Tuple[jax.Array, Tuple]:

        # Calculate the critic loss.
        critic_loss, critic_metric = self._critic_loss_fn(traj_batch, targets)
        return critic_loss, critic_metric


    def actor_loss(
        self, traj_batch: Transition, advantages: jax.Array
    ) -> Tuple[jax.Array, Tuple]:
        # Calculate the actor loss.
        actor_loss, actor_metric = self._actor_loss_fn(traj_batch, advantages)
        return actor_loss, actor_metric

    def update_policy(self):

        counter = self.counter.value

        init = jnp.astype(counter == 0, jnp.float32)
        update = jnp.astype(
            counter % self.config.target_update_period == 0, jnp.float32
        )
        mix = jnp.clip(
            init * 1.0 + update * (1 - self.config.target_update_tau), 0.0, 1.0
        )

        # Update the slow value head.
        def update(x: jax.Array, y: jax.Array) -> jax.Array:
            return (1 - mix) * x + mix * y

        params = self.critic.variables.get("params")
        slow_params = self.slow_critic.variables.get("params")
        for key in slow_params.keys():
            param = params.get(key)
            slow_param = slow_params.get(key)
            slow_param = jax.tree_map(update, slow_param, param)
            self.slow_critic.put_variable("params", key, slow_param)
        # Update the counter.
        self.counter.value = counter + 1


def create_ac_dreamer(
    actor: nn.Module, critic: nn.Module, slow_critic: nn.Module, config: BaseDataType
) -> Callable:

    def call():
        return ACDreamer(
            actor=actor, critic=critic, slow_critic=slow_critic, config=config
        )

    return call
