"""
    NOTE to myself.
    RSSM equations

    Sequence Model:       h = f( h, z, a )
    Embedder:             e = q( x )
    Encoder:              zprior ~ q ( zprior | h, e )
    Dynamics predictor    zpost ~ p ( zpost | h )
    Reward predictor:     r ~ p( z | h )
    Continue predictor:   c ~ p( z | h )
    Decoder:              x ~ p( x | h, z )

    During training z = zprior
    During prediction z = zpost
"""

from typing import Any, Callable, Optional, Tuple

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.linen import initializers

from noncleandreamer.ac import ACDreamer
from noncleandreamer.custom_types import (
    ACLossInfo,
    RSSMLossInfo,
    RSSMState,
    Transition,
    BaseDataType,
    base_jnp_type
)
from noncleandreamer.distributions import MSE, Discrete, OneHotCategorical


def logits_to_normal(logits: jax.Array):
    mean, logstd = jnp.split(logits, 2, -1)
    return {"loc": mean, "scale_diag": logstd}


class RecurrentCellwResets(nn.Module):
    hidden_dim: int
    reset_on_termination: bool
    kernel_initializer: initializers.Initializer
    recurrent_initializer: initializers.Initializer

    @nn.compact
    def __call__(
        self, carry: jnp.ndarray, inputs: jnp.ndarray, terminations: jnp.ndarray,
    ) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray]:

        xs = inputs

        if self.reset_on_termination:
            carry = jax.tree.map(
                lambda x: jnp.where(
                    terminations[:, jnp.newaxis],
                    jax.tree.map(lambda y: jnp.zeros_like(y), tree=x),
                    x,
                ),
                carry,
            )

        h_state, ys = nn.GRUCell(
            self.hidden_dim,
            kernel_init=self.kernel_initializer,
            recurrent_kernel_init=self.recurrent_initializer,
            dtype=base_jnp_type,
        )(carry, xs)

        return h_state, ys

    @staticmethod
    def initialize_carry(
        batch_size: int, hidden_dim: int, seed: int = 0, **kwargs: Any
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        return nn.GRUCell(hidden_dim, parent=None).initialize_carry(
            jax.random.key(seed),
            (batch_size, hidden_dim),
        )


class Prior(nn.Module):
    latent_repr_fn: Callable
    sequential_proc_fn: Callable
    decoder_fn: Callable

    def setup(self) -> None:
        self.latent_repr = self.latent_repr_fn()
        self.seq_fn = self.sequential_proc_fn()
        self.decoder = self.decoder_fn()

    def __call__(self, carry: Tuple[jax.Array, jax.Array], x: Tuple) -> Tuple:
        """Computes the forward pass for time step t for prior"""

        stochastic_state, deterministic_state = carry
        prev_agent_action, termination = x

        inp = jnp.concatenate(
            [prev_agent_action.reshape(-1), stochastic_state.reshape(-1)], axis=0
        )

        latent_repr = self.latent_repr(inp)

        deterministic_state, _ = self.seq_fn(
            deterministic_state, latent_repr, termination
        )

        seq_repr = self.decoder(deterministic_state)

        return (stochastic_state, deterministic_state), seq_repr


class Posterior(nn.Module):
    latent_repr_fn: Callable
    decoder_fn: Callable

    def setup(self) -> None:
        self.latent_repr = self.latent_repr_fn()
        self.dec_fn = self.decoder_fn()

    def __call__(self, carry: Tuple[jax.Array, jax.Array], x: Tuple) -> Tuple:
        """Computes the forward pass for time step t for posterior"""

        stochastic_state, deterministic_state = carry
        (agent_observation,) = x

        inp = jnp.concatenate(
            [deterministic_state, agent_observation.reshape(-1)], axis=-1
        )

        latent_repr = self.latent_repr(inp)

        seq_repr = self.dec_fn(latent_repr)

        return (stochastic_state, deterministic_state), seq_repr


class RSSM(nn.Module):
    prior_fn: Callable
    posterior_fn: Callable
    seq_init_fn_deter: Callable
    seq_init_fn_stoch: Callable
    discrete_dim: Optional[jax.Array]
    num_categories: Optional[jax.Array]
    unimix_ratio: Optional[jax.Array]

    def setup(self) -> None:
        self.deterministic_state = self.param(
            "deterministic_state", self.seq_init_fn_deter, 1
        )
        self.prior = self.prior_fn()
        self.posterior = self.posterior_fn()

    def _mask(self, value: jax.Array, mask: jax.Array) -> jnp.ndarray:
        # stolen
        return jnp.where((mask == 0), jnp.array(0), value) #value[mask.astype(jnp.bool)]

    # @functools.partial(jax.jit, static_argnums=(1,))
    def initial_state(self, batch_size: int) -> RSSMState:
        deterministic_state = jax.tree.map(
            lambda x: jnp.repeat(x, batch_size, axis=0), self.deterministic_state
        )
        deterministic_state = jax.tree.map(
            lambda x: jnp.tanh(x.astype(base_jnp_type)), deterministic_state
        )
        logits = self.prior.decoder(deterministic_state).astype(
            base_jnp_type
        )
        stochastic_state = self.distritbution(logits).mode().astype(base_jnp_type)

        return RSSMState(
            logits=logits,
            deterministic_state=deterministic_state,
            stochastic_state=stochastic_state,
        )

    def imagine_step(
        self,
        x: RSSMState,
        action: jax.Array,
        termination: jax.Array,
    ) -> RSSMState:

        _, stochastic_state, deterministic_state = x

        rng = self.make_rng("prior")

        (stochastic_state, deterministic_state), seq_repr = self.prior(
            (stochastic_state, deterministic_state),
            (action.astype(base_jnp_type), termination),
        )

        logits = seq_repr.astype(
            base_jnp_type
        )

        pred_dynamics_distribution = self.distritbution(logits)

        stochastic_state = (
            pred_dynamics_distribution.sample(seed=rng).astype(base_jnp_type).squeeze(0)
        )

        return RSSMState(
            logits=logits,
            stochastic_state=stochastic_state,
            deterministic_state=jax.tree.map(
                lambda x: x.astype(base_jnp_type), deterministic_state
            ),
        )

    def __call__(
        self,
        x: RSSMState,
        action: jax.Array,
        termination: jax.Array,
        first: jax.Array,
        encoded_observation: jax.Array,
    ) -> Tuple[RSSMState, RSSMState]:

        _, stochastic_state, deterministic_state = x

        initial_x = self.initial_state(1)
        initial_x = jax.tree.map(lambda x: x[0], initial_x)

        first = first.astype(base_jnp_type)
        action = action.astype(base_jnp_type)

        x, action = jax.tree.map(lambda x: self._mask(x, 1.0 - first), (x, action))
        x = jax.tree.map(lambda _x, y: _x + self._mask(y, first), x, initial_x)

        prior_output = self.imagine_step(x, action, termination)

        (stochastic_state, deterministic_state), posterior_seq_repr = self.posterior(
            (stochastic_state, prior_output.deterministic_state), (encoded_observation,)
        )

        posterior_distribution = self.distritbution(posterior_seq_repr)
        posterior_rng = self.make_rng("posterior")

        stochastic_state = (
            posterior_distribution.sample(seed=posterior_rng)
            .astype(base_jnp_type)
            .squeeze(0)
        )

        posterior_output = RSSMState(
            logits=posterior_seq_repr.astype(base_jnp_type),
            stochastic_state=stochastic_state,
            deterministic_state=deterministic_state,
        )

        return prior_output, posterior_output

    def distritbution(
        self, features: jax.Array
    ) -> distrax.MultivariateNormalDiag | distrax.OneHotCategorical:

        features = jnp.astype(features, jnp.float32)

        if self.num_categories is None or self.num_categories <= 1:
            return distrax.MultivariateNormalDiag(**logits_to_normal(features))

        features = features.reshape(-1, self.discrete_dim, self.num_categories)
        return OneHotCategorical(features, self.unimix_ratio)

    def dyn_loss(
        self, prior_data: RSSMState, posterior_data: RSSMState, free_kl: jax.Array
    ) -> Tuple[jax.Array, jax.Array]:

        # Get the posterior and prior distributions.
        posterior_distribution = self.distritbution(
            jax.lax.stop_gradient(posterior_data.logits)
        )
        prior_distribution = self.distritbution(prior_data.logits)

        # Calculate the dynamic loss.
        dyn_loss = posterior_distribution.kl_divergence(prior_distribution)
        dyn_loss = jnp.maximum(dyn_loss, free_kl)

        # Calculate the posterior entropy.
        posterior_entropy = posterior_distribution.entropy()

        return dyn_loss, posterior_entropy

    def rep_loss(
        self, prior_data: RSSMState, posterior_data: RSSMState, free_kl: jax.Array
    ) -> Tuple[jax.Array, jax.Array]:

        # Get the posterior and prior distributions.
        posterior_distribution = self.distritbution(posterior_data.logits)
        prior_distribution = self.distritbution(
            jax.lax.stop_gradient(prior_data.logits)
        )

        # Calculate the representation loss.
        rep_loss = posterior_distribution.kl_divergence(prior_distribution)
        rep_loss = jnp.maximum(rep_loss, free_kl)

        # Calculate the prior entropy.
        prior_entropy = prior_distribution.entropy()

        return rep_loss, prior_entropy


class WorldModel(nn.Module):
    config: BaseDataType

    rssm_fn: Callable
    representation_fn: Callable
    observation_decoder_fn: Callable
    ternination_decoder_fn: Callable
    reward_decoder_fn: Callable
    continuous_actions: jax.Array
    ind_action_decoder_fn: Callable

    def setup(self) -> None:
        self.rssm: RSSM = self.rssm_fn()
        self.num_categories: int = self.rssm.num_categories
        self.observation_encoder = self.representation_fn()
        self.observation_decoder = self.observation_decoder_fn()
        self.termination_predictor = self.ternination_decoder_fn()
        self.reward_predictor = self.reward_decoder_fn()

        self.ind_action_predictor = self.ind_action_decoder_fn()

    def initial_state(self, batch_size: int) -> RSSMState:
        return (
            self.rssm.initial_state(batch_size),
            jnp.zeros((batch_size, self.config.num_actions)),
        )

    def get_state(self, x: RSSMState) -> jax.Array:
        # NOTE: input is sequential, i.e. (seq_len, ...)
        base_shape = x.stochastic_state.shape[0]
        h = x.deterministic_state.reshape(base_shape, -1)
        z = x.stochastic_state.reshape(base_shape, -1)
        return jnp.concatenate([h, z], -1)

    @staticmethod
    def observation_step(
        cell: RSSM,
        state: Tuple[RSSMState, jax.Array],
        encoded_observation: jax.Array,
        action: jax.Array,
        termination: jax.Array,
        first: jax.Array,
    ) -> RSSMState:
        """Runs an observation step."""
        # Run a RSSM observation step.
        prior_output, posterior_output = cell(
            *state, termination, first, encoded_observation
        )

        # Update the state.
        next_state = (posterior_output, action)

        return next_state, (prior_output, posterior_output)

    def __call__(
        self,
        rssm_state: RSSMState,
        obsevarion: jax.Array,
        action: jax.Array,
        termination: jax.Array,
        first: jax.Array,
    ):
        """
        Compute the next RSSM's state
        """
        encoded_observations = self.observation_encoder(obsevarion)

        # Run an observation step.
        scan = nn.scan(
            self.observation_step,
            variable_broadcast="params",
            split_rngs={"params": False, "prior": True, "posterior": True},
            in_axes=0,
            out_axes=0,
        )
        x, (prior_output, posterior_output) = scan(
            self.rssm, rssm_state, encoded_observations, action, termination, first
        )
        return x, (prior_output, posterior_output)

    def decode_terminals(self, features: jax.Array) -> distrax.Bernoulli:
        return distrax.Bernoulli(self.termination_predictor(features).squeeze(-1))

    def decode_rewards(self, features: jax.Array) -> jax.Array:
        loc = self.reward_predictor(features)
        return Discrete(loc)

    def decode_observations(self, features: jax.Array) -> jax.Array:
        loc = self.observation_decoder(features)
        return MSE(loc)

    def decode_actions(
        self, features: jax.Array
    ) -> distrax.Categorical | distrax.MultivariateNormalDiag:

        action_features = self.ind_action_predictor(features)
        if self.continuous_actions:
            return distrax.MultivariateNormalDiag(**logits_to_normal(action_features))

        return OneHotCategorical(
            logits=action_features, unimix_ratio=self.rssm.unimix_ratio
        )

    def loss(
        self,
        rssm_state: RSSMState,
        observation: jax.Array,
        action: jax.Array,
        reward: jax.Array,
        termination: jax.Array,
        first: jax.Array,
    ) -> RSSMLossInfo:

        rssm_state, (prior, posterior) = self(
            rssm_state, observation, action, termination, first
        )

        cond_state = self.get_state(posterior).astype(
            jnp.float32
        )  # all losses in full precision
        terminal_d = self.decode_terminals(cond_state)
        reward_d = self.decode_rewards(cond_state)
        observation_d = self.decode_observations(cond_state)
        action_d = self.decode_actions(cond_state)

        rep_loss, prior_entropy = self.rssm.rep_loss(
            prior, posterior, jnp.array(self.config.free_kl)
        )
        dyn_loss, posterior_entropy = self.rssm.dyn_loss(
            prior, posterior, jnp.array(self.config.free_kl)
        )

        return (
            RSSMLossInfo(
                prior_entropy=prior_entropy,
                posterior_entropy=posterior_entropy,
                rep_loss=self.config.kl_balance_rep * rep_loss,
                dyn_loss=self.config.kl_balance_dyn * dyn_loss,
                log_p_o=-observation_d.log_prob(observation),
                log_p_a=-action_d.log_prob(action),
                log_p_r=-reward_d.log_prob(reward),
                log_p_d=-terminal_d.log_prob(termination),
            ),
            rssm_state,
            posterior,
        )


class WorldModelAgent(nn.Module):
    world_model_fn: Callable
    policy_fn: Callable
    imagine_horizon: int

    def setup(self) -> None:
        self.world_model: WorldModel = self.world_model_fn()
        self.policy: ACDreamer = self.policy_fn()

    def initial_state_world_model(self, batch_size: int) -> RSSMState:
        return self.world_model.initial_state(batch_size)

    def initial_state_policy(self, batch_size: int) -> RSSMState:
        return self.policy.initial_state(batch_size)

    def world_model_loss(
        self,
        rssm_state: RSSMState,
        observation: jax.Array,
        action: jax.Array,
        reward: jax.Array,
        termination: jax.Array,
        first: jax.Array,
    ) -> Tuple[RSSMLossInfo, jax.Array]:

        if rssm_state is None:
            rssm_state = self.initial_state_world_model(1)
            rssm_state = jax.tree.map(lambda x: x[0], rssm_state)

        observation = observation.astype(jnp.float32)
        losses, rssm_state, posterior = self.world_model.loss(
            rssm_state, observation, action, reward, termination, first
        )

        losses: RSSMLossInfo = jax.tree.map(lambda x: jnp.mean(x), losses)
        total_loss = (
            losses.dyn_loss
            + losses.rep_loss
            + losses.log_p_o
            + losses.log_p_a
            + losses.log_p_r
            + losses.log_p_d
        )
        losses = losses._replace(loss=total_loss)

        return total_loss, (posterior, rssm_state, losses)

    def act(
        self,
        observation: jax.Array,
        termination: jax.Array,
        first: jax.Array,
        rssm_state: RSSMState,
    ) -> Tuple[jax.Array, Tuple]:

        # print(observation.shape, termination.shape, action.shape)

        if rssm_state is None:
            rssm_state = self.initial_state_world_model(1)
            rssm_state = (jax.tree.map(lambda x: x[0], rssm_state),)

        # Encode the observation.
        observation = observation.astype(jnp.float32)
        encoded_observation = self.world_model.observation_encoder(observation)

        # Run the RSSM observation step.
        _, posterior = self.world_model.rssm(
            *rssm_state, termination, first, encoded_observation
        )

        # Sample an action.
        latent = posterior.get_state()
        actor_output = self.policy.act(latent, None, None, None)

        # Update the state.
        state = (posterior, actor_output.action)

        return actor_output, state

    def compute_advantages(
        self,
        traj_batch: Transition,
    ) -> Tuple[jax.Array, jax.Array]:

        return self.policy.compute_advantages(traj_batch)

    def critic_loss(
        self, traj_batch: Transition, traj_values: jax.Array, targets: jax.Array
    ) -> Tuple[jax.Array, Tuple]:

        return self.policy.critic_loss(traj_batch, traj_values, targets)

    def actor_loss(
        self, traj_batch: Transition, advantages: jax.Array
    ) -> Tuple[jax.Array, Tuple]:

        return self.policy.actor_loss(traj_batch, advantages)

    def policy_loss(
        self,
        traj_batch: Transition,
    ) -> Tuple[jax.Array, ACLossInfo]:
        # Calculate the policy loss.
        advantages, targets = self.policy.compute_advantages(traj_batch)
        critic_loss, critic_metric = self.policy.critic_loss(traj_batch, targets)
        actor_loss, actor_metric = self.policy.actor_loss(traj_batch, advantages)
        policy_loss = actor_loss + critic_loss

        loss_info = ACLossInfo(
            total_loss=policy_loss,
            actor_loss=actor_metric[0],
            critic_loss=critic_metric[0],
            entropy=actor_metric[1],
        )

        return policy_loss, jax.tree.map(jnp.mean, loss_info)

    def update_policy(self):
        """Updates the policy."""
        self.policy.update_policy()

    @staticmethod
    def img_step(
        agent: "WorldModelAgent",
        state: Tuple[RSSMState, jax.Array],
    ) -> RSSMState:
        # Run a RSSM imagination step.
        rssm_state, action = state
        prior = agent.world_model.rssm.imagine_step(rssm_state, action, None)

        # Sample an action.
        latent = prior.get_state()
        action = agent.policy.act(latent, None, None, None).action

        # Update the state.
        state = (prior, action)

        return state, (latent, action)

    def imagine(self, initial_state: RSSMState, data: jax.Array) -> Transition:
        termination = data

        initial_latent = initial_state.get_state()
        initial_action = self.policy.act(initial_latent, None, None, None).action
        initial_termination = termination

        # Run an imagination step.
        scan = nn.scan(
            self.img_step,
            variable_broadcast=["params", "aux"],
            split_rngs={"params": False, "prior": True, "action": True},
            in_axes=0,
            out_axes=0,
            length=self.imagine_horizon,
        )
        _, (latent, action) = scan(self, (initial_state, initial_action))

        # Concatenate the input and imagined data.
        reward = self.world_model.decode_rewards(latent).mean()
        termination = self.world_model.decode_terminals(latent).mode().astype(jnp.bool_)

        latent = jnp.concatenate([initial_latent[jnp.newaxis], latent], axis=0)
        action = jnp.concatenate([initial_action[jnp.newaxis], action], axis=0)
        termination = jnp.concatenate(
            [initial_termination[jnp.newaxis], termination], axis=0
        )

        return Transition(
            observation=latent, termination=termination, action=action, reward=reward
        )


class SupervisedWorldModelAgent(nn.Module):
    world_model_fn: Callable
    imagine_horizon: int

    def setup(self) -> None:
        self.world_model: WorldModel = self.world_model_fn()

    def initial_state_world_model(self, batch_size: int) -> RSSMState:
        return self.world_model.initial_state(batch_size)

    def world_model_loss(
        self,
        rssm_state: RSSMState,
        observation: jax.Array,
        action: jax.Array,
        reward: jax.Array,
        termination: jax.Array,
        first: jax.Array,
    ) -> Tuple[RSSMLossInfo, jax.Array]:

        if rssm_state is None:
            rssm_state = self.initial_state_world_model(1)
            rssm_state = jax.tree.map(lambda x: x[0], rssm_state)

        observation = observation.astype(jnp.float32)
        losses, rssm_state, posterior = self.world_model.loss(
            rssm_state, observation, action, reward, termination, first
        )

        losses: RSSMLossInfo = jax.tree.map(lambda x: jnp.mean(x), losses)
        total_loss = (
            losses.dyn_loss
            + losses.rep_loss
            + losses.log_p_o
            + losses.log_p_a
            + losses.log_p_r
            + losses.log_p_d
        )
        losses = losses._replace(loss=total_loss)

        return total_loss, (posterior, rssm_state, losses)

    def act(
        self,
        observation: jax.Array,
        termination: jax.Array,
        first: jax.Array,
        rssm_state: RSSMState,
    ) -> Tuple[jax.Array, jax.Array]:

        if rssm_state is None:
            rssm_state = self.initial_state_world_model(1)
            rssm_state = (jax.tree.map(lambda x: x[0], rssm_state),)

        # Encode the observation.
        observation = observation.astype(jnp.float32)
        encoded_observation = self.world_model.observation_encoder(observation)

        # Run the RSSM observation step.
        _, posterior = self.world_model.rssm(
            *rssm_state, termination, first, encoded_observation
        )

        # Sample an action.
        latent = posterior.get_state()
        decoded_action = self.world_model.decode_actions(latent).mode()

        # Update the state.
        state = (posterior, decoded_action)

        return decoded_action, state

    @staticmethod
    def img_step(
        agent: "SupervisedWorldModelAgent",
        state: Tuple[RSSMState, jax.Array],
    ) -> RSSMState:
        # Run a RSSM imagination step.
        rssm_state, action = state
        prior = agent.world_model.rssm.imagine_step(rssm_state, action, None)

        # Sample an action.
        latent = prior.get_state()
        action = agent.world_model.decode_actions(latent).mode()

        # Update the state.
        state = (prior, action)

        return state, (latent, action)

    def imagine(self, initial_state: RSSMState, data: jax.Array) -> Transition:
        termination = data

        initial_latent = initial_state.get_state()
        initial_action = self.world_model.decode_actions(initial_latent).mode()
        initial_termination = termination

        # Run an imagination step.
        scan = nn.scan(
            self.img_step,
            variable_broadcast=["params", "aux"],
            split_rngs={"params": False, "prior": True, "action": True},
            in_axes=0,
            out_axes=0,
            length=self.imagine_horizon,
        )
        _, (latent, action) = scan(self, (initial_state, initial_action))

        # Concatenate the input and imagined data.
        reward = self.world_model.decode_rewards(latent).mean()
        termination = self.world_model.decode_terminals(latent).mode().astype(jnp.bool_)

        latent = jnp.concatenate([initial_latent[jnp.newaxis], latent], axis=0)
        action = jnp.concatenate([initial_action[jnp.newaxis], action], axis=0)
        termination = jnp.concatenate(
            [initial_termination[jnp.newaxis], termination], axis=0
        )

        return Transition(
            observation=latent, termination=termination, action=action, reward=reward
        )


def seq_proc_fn(seed: int, kwargs: Any) -> Tuple[Callable, Callable]:
    def call() -> RecurrentCellwResets:
        return RecurrentCellwResets(**kwargs)

    def init(rng, batch_size: int):
        return RecurrentCellwResets.initialize_carry(
            **kwargs, batch_size=batch_size, seed=seed
        )

    return call, init


def prior_fn(
    latent_repr_fn: Callable,
    seq_init_fn: Callable,
    seq_proc_fn: Callable,
    decoder_fn: Callable,
):
    def call():
        return Prior(latent_repr_fn, seq_proc_fn, decoder_fn)

    return call, seq_init_fn


def posterior_fn(latent_repr_fn: Callable, decoder_fn: Callable):
    def call():
        return Posterior(latent_repr_fn, decoder_fn)

    return call


def rssm_fn(
    prior_fns: Tuple[Callable, Callable],
    posterior_fn: Callable,
    state_dim: int,
    stochastic_state_state_dim: int,
    num_categories: int,
    unimix_ratio: float,
) -> Callable:

    prior_fn, prior_seq_init_fn = prior_fns

    def init_stochastic_state(
        batch_size: int, discrete_dim: int, num_categories: Optional[int] = None
    ) -> jax.Array:
        if num_categories is None or num_categories <= 1:
            return jnp.zeros((batch_size, state_dim), dtype=jnp.float32)
        return jnp.zeros((batch_size, discrete_dim, num_categories), dtype=jnp.float32)

    def call():
        return RSSM(
            prior_fn,
            posterior_fn,
            prior_seq_init_fn,
            init_stochastic_state,
            stochastic_state_state_dim,
            num_categories,
            unimix_ratio,
        )

    return call


def world_model_fn(
    config: BaseDataType,
    rssm_fn: Callable,
    representation_fn: Callable,
    observation_decoder_fn: Callable,
    ternination_decoder_fn: Callable,
    reward_decoder_fn: Callable,
    continuous_actions: jax.Array,
    ind_action_decoder_fn: Callable,
) -> Callable:
    def call():
        return WorldModel(
            config=config,
            rssm_fn=rssm_fn,
            representation_fn=representation_fn,
            observation_decoder_fn=observation_decoder_fn,
            ternination_decoder_fn=ternination_decoder_fn,
            reward_decoder_fn=reward_decoder_fn,
            continuous_actions=continuous_actions,
            ind_action_decoder_fn=ind_action_decoder_fn,
        )

    return call
