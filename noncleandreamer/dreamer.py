import functools
from typing import Any, Callable, Tuple

import flashbax as fbx
import jax
import jax.numpy as jnp
import optax
from flashbax.buffers.trajectory_buffer import TrajectoryBuffer
from flax.training.dynamic_scale import DynamicScale

from noncleandreamer.custom_types import (
    ACLossInfo,
    ActorOutput,
    BaseDataType,
    DreamerTrainState,
    FrozenDict,
    RSSMLossInfo,
    RSSMState,
    Transition,
)
from noncleandreamer.rssm import SupervisedWorldModelAgent, WorldModelAgent


def create_item_buffer(
    max_length: int,
    min_length: int,
    add_batch_size: int,
    sample_batch_size: int,
    sample_sequence_length: int,
) -> fbx.item_buffer.TrajectoryBuffer:
    return fbx.make_trajectory_buffer(
        max_length_time_axis=int(max_length),
        min_length_time_axis=min_length,
        add_batch_size=add_batch_size,
        sample_batch_size=sample_batch_size,
        sample_sequence_length=sample_sequence_length,
        period=1,
    )


class Every:
    def __init__(self, every):
        self._every = every
        self._last = None

    def __call__(self, step):
        if not self._every:
            return 0
        if self._last is None:
            self._last = step
            return 1
        count = int((step - self._last) / self._every)
        self._last += self._every * count
        return count


class Ratio:

    def __init__(self, ratio):
        assert ratio >= 0, ratio
        self._ratio = ratio
        self._prev = None

    def __call__(self, step):
        step = int(step)

        if self._ratio == 0:
            return 0

        if self._prev is None:
            self._prev = step
            return 1

        repeats = int((step - self._prev) * self._ratio)

        self._prev += repeats / self._ratio

        return repeats


class DreamerAgent:

    def __init__(
        self,
        config: BaseDataType,
        env: Any,
        eval_env: Any,
        replay_buffer: TrajectoryBuffer,
        world_model_fn: Callable,
        policy_fn: Callable,
        rollout_fn: Callable,
        train_ratio: int,
        **kwargs: Any
    ) -> None:

        # here we take only supervised learning part
        self.name = self.__class__.__name__
        # self.num_vec_ens = config.num_envs * config.num_agents
        self.config = config
        self.rollout_fn = rollout_fn

        self.should_train = Ratio(train_ratio)

        self.devices = jax.local_devices()
        env, self.env_specs = env

        self.agent: WorldModelAgent = WorldModelAgent(
            world_model_fn,
            policy_fn,
            imagine_horizon=config.world_model_config.imagination_horizon,
        )

        # Key
        self.key = jax.random.key(config.seed)

        # Model state
        self.model_state = self._init_world_model_state(config)

        # Policy state
        self.policy_state, _ = self._init_policy_state(config)

        # not to overflow gpu ram, allocating replay memeory on cpu
        self.replay_buffer = replay_buffer.replace(
            init=jax.jit(replay_buffer.init, backend="cpu"),
            add=jax.jit(replay_buffer.add, donate_argnums=0, backend="cpu"),
            sample=jax.jit(replay_buffer.sample, backend="cpu"),
            can_sample=jax.jit(replay_buffer.can_sample, backend="cpu"),
        )

        self.num_envs = self.config.num_envs
        self.num_agents = self.config.num_agents
        self.num_vecenvs = self.num_envs * self.num_agents

    def initial_state_world_model(self, batch_size: int) -> Tuple[RSSMState, Tuple]:
        variables = {"params": self.model_state.params, "aux": self.model_state.aux}
        return self.agent.apply(
            variables, batch_size, method=self.agent.initial_state_world_model
        )

    def initial_state_policy(self, batch_size: int) -> Tuple[RSSMState, Tuple]:
        variables = {"params": self.policy_state.params, "aux": self.policy_state.aux}
        return self.agent.apply(
            variables, batch_size, method=self.agent.initial_state_policy
        )

    def _init_world_model_state(self, config: BaseDataType) -> DreamerTrainState:
        param_key, post_key, prior_key, self.key = jax.random.split(self.key, 4)
        rngs = {"params": param_key, "posterior": post_key, "prior": prior_key}
        data = {
            "observation": jnp.zeros(
                (1, *self.env_specs["observation_space"]), jnp.float32
            ),
            "action": jnp.zeros((1, self.env_specs["action_space"]), jnp.float32),
            "reward": jnp.zeros((1,), jnp.float32),
            "termination": jnp.zeros((1,), jnp.uint8),
            "first": jnp.zeros((1,), jnp.float32),  # ??????
        }
        variables = self.agent.init(
            rngs, rssm_state=None, **data, method=self.agent.world_model_loss
        )

        # Define the model state.
        params = variables["params"]

        tx = optax.chain(
            optax.clip_by_global_norm(config.world_model_config.max_grad_norm),
            optax.adam(learning_rate=config.world_model_config.lr, eps=1e-8),
        )
        dynamic_scale = DynamicScale()

        model_state = DreamerTrainState.create(
            apply_fn=functools.partial(
                self.agent.apply, method=self.agent.world_model_loss
            ),
            params=params,
            tx=tx,
            aux={},
            dynamic_scale=dynamic_scale,
        )

        return model_state

    def _init_policy_state(self, config: BaseDataType) -> DreamerTrainState:
        param_key, self.key = jax.random.split(self.key)
        rngs = {"params": param_key}
        traj = Transition(
            **{
                "state": None,
                "observation": jnp.zeros((2, config.latent_dim), jnp.float16),
                "action": jnp.zeros((2, self.env_specs["action_space"]), jnp.float32),
                "reward": jnp.zeros((1,), jnp.float32),
                "termination": jnp.zeros((2,), jnp.uint8),
                "value": jnp.zeros((2,), jnp.float32),
            }
        )
        adv = jnp.zeros((1,), jnp.float32)
        critic_variables = self.agent.init(
            rngs, traj, adv, method=self.agent.critic_loss
        )
        actor_variables = self.agent.init(rngs, traj, adv, method=self.agent.actor_loss)

        tx_actor = tx_critic = optax.chain(
            optax.clip_by_global_norm(config.policy_config.max_grad_norm),
            optax.adam(learning_rate=config.policy_config.lr, eps=1e-5),
        )
        actor_dynamic_scale, critic_dynamic_scale = DynamicScale(), DynamicScale()

        policy_state_actor = DreamerTrainState.create(
            apply_fn=functools.partial(self.agent.apply, method=self.agent.actor_loss),
            params=actor_variables["params"],
            tx=tx_actor,
            aux=None,
            dynamic_scale=actor_dynamic_scale,
        )

        policy_state_critic = DreamerTrainState.create(
            apply_fn=functools.partial(self.agent.apply, method=self.agent.critic_loss),
            params=critic_variables["params"],
            tx=tx_critic,
            aux=critic_variables["aux"],
            dynamic_scale=critic_dynamic_scale,
        )

        return (policy_state_actor, policy_state_critic), traj

    @staticmethod
    @functools.partial(jax.jit, static_argnums=(0,))
    def _act(
        agent: WorldModelAgent,
        variables: FrozenDict,
        rngs: jax.Array,
        observation: jax.Array,
        termination: jax.Array,
        firsts: jax.Array,
        rssm_state: RSSMState = None,
    ) -> ActorOutput:
        num_envs, num_agents = observation.shape[:2]

        @functools.partial(jax.vmap, in_axes=(None, 1, 1, 1, 1, 1), out_axes=1)
        @functools.partial(jax.vmap, in_axes=(None, 0, 0, 0, 0, 0), out_axes=0)
        def apply_fn(
            variables: FrozenDict,
            rng: jax.Array,
            observation: jax.Array,
            termination: jax.Array,
            firsts: jax.Array,
            rssm_state: RSSMState,
        ):
            return agent.apply(
                variables,
                observation,
                termination,
                firsts,
                rssm_state,
                method=agent.act,
                rngs=rng,
            )

        return apply_fn(
            variables,
            jax.tree_map(
                lambda rng: jax.random.split(rng, num_envs * num_agents).reshape(
                    num_envs, num_agents
                ),
                rngs,
            ),
            observation,
            termination,
            firsts,
            rssm_state,
        )

    def act(
        self,
        observation: jax.Array,
        termination: jax.Array,
        firsts: jax.Array,
        rssm_state: RSSMState = None,
    ) -> ActorOutput:
        # Get the agent and key.
        agent = self.agent
        key = self.key

        # Sample an action.
        actor_state, critic_state = self.policy_state
        params = {
            "policy": {**actor_state.params["policy"], **critic_state.params["policy"]},
            **self.model_state.params,
        }

        aux = {**self.model_state.aux, **critic_state.aux}
        variables = {"params": params, "aux": aux}
        post_key, prior_key, action_key, key = jax.random.split(key, 4)
        rngs = {"posterior": post_key, "prior": prior_key, "action": action_key}
        action, state = self._act(
            agent,
            variables,
            rngs,
            observation,
            termination,
            firsts,
            rssm_state=rssm_state,
        )
        # Update the key.
        self.key = key
        # self.policy_state = (actor_state, critic_state)
        return action, state

    @staticmethod
    @functools.partial(jax.jit, static_argnums=(0,))
    def _imagine(
        agent: WorldModelAgent,
        variables: FrozenDict,
        rngs: jax.Array,
        rssm_state: RSSMState,
        termination: jax.Array,
    ) -> Transition:

        num_envs, num_agents = termination.shape[:2]

        @functools.partial(jax.vmap, in_axes=(None, 1, 1, 1), out_axes=2)
        @functools.partial(jax.vmap, in_axes=(None, 0, 0, 0), out_axes=1)
        def apply_fn(
            variables: FrozenDict,
            rng: jax.Array,
            rssm_state: RSSMState,
            termination: jax.Array,
        ):
            return agent.apply(
                variables, rssm_state, termination, method=agent.imagine, rngs=rng
            )

        return apply_fn(
            variables,
            jax.tree_map(
                lambda rng: jax.random.split(rng, num_envs * num_agents).reshape(
                    num_envs, num_agents
                ),
                rngs,
            ),
            rssm_state,
            termination,
        )

    def imagine(self, posterior: RSSMState, termination: jax.Array) -> Transition:
        # Get the agent and key.
        agent = self.agent
        key = self.key

        # Run an imagination.

        actor_state, critic_state = self.policy_state
        params = {
            "policy": {**actor_state.params["policy"], **critic_state.params["policy"]},
            **self.model_state.params,
        }

        aux = {**self.model_state.aux, **critic_state.aux}
        variables = {"params": params, "aux": aux}
        post_key, prior_key, action_key, key = jax.random.split(key, 4)
        rngs = {"posterior": post_key, "prior": prior_key, "action": action_key}
        traj = self._imagine(agent, variables, rngs, posterior, termination)
        # Update the key.
        self.key = key
        # self.policy_state = (actor_state, critic_state)

        return traj

    @staticmethod
    @jax.jit
    def _train_model(
        model_state: DreamerTrainState,
        rngs: jax.Array,
        data: Transition,
        rssm_state: RSSMState,
    ) -> Tuple[DreamerTrainState, Tuple[RSSMState, RSSMState, RSSMLossInfo]]:

        # Update the model parameters.
        def loss_fn(params: FrozenDict):
            variables = {"params": params, "aux": model_state.aux}

            @functools.partial(
                jax.vmap, in_axes=(None, 1, 2, 1), out_axes=(1, (2, 1, 1))
            )
            @functools.partial(
                jax.vmap, in_axes=(None, 0, 1, 0), out_axes=(0, (1, 0, 0))
            )
            def apply_fn(
                variables: FrozenDict,
                rngs: jax.Array,
                data: Transition,
                rssm_state: RSSMState,
            ):

                return model_state.apply_fn(
                    variables,
                    rssm_state,
                    data.observation,
                    data.action,
                    data.reward,
                    data.termination,
                    data.is_first,
                    rngs=rngs,
                )

            total_loss, aux = apply_fn(
                variables,
                rngs,
                data,
                rssm_state,
            )

            return total_loss.mean(), aux

        grad_fn = model_state.dynamic_scale.value_and_grad(loss_fn, has_aux=True)
        dynamic_scale, finite, aux, grads = grad_fn(model_state.params)
        _, (posterior, rssm_state, losses) = aux

        new_model_state = model_state.apply_gradients(grads=grads)

        losses = losses._replace(grad_norm=optax.global_norm(grads))

        # Update the model state.
        opt_state = jax.tree_map(
            functools.partial(jnp.where, finite),
            new_model_state.opt_state,
            model_state.opt_state,
        )
        params = jax.tree_map(
            functools.partial(jnp.where, finite),
            new_model_state.params,
            model_state.params,
        )
        model_state = new_model_state.replace(
            opt_state=opt_state,
            params=params,
            dynamic_scale=dynamic_scale,
        )

        return model_state, (posterior, rssm_state, losses)

    def train_model(
        self,
        data: Transition,
        rssm_state: RSSMState | None = None,
    ) -> Tuple[DreamerTrainState, Tuple]:
        """Trains the model."""
        # Get the model state and key.
        model_state = self.model_state
        key = self.key
        num_envs, num_agents = data.action[0].shape[:2]
        # Train the model.
        post_key, prior_key, key = jax.random.split(key, 3)
        rngs = {"posterior": post_key, "prior": prior_key}
        model_state, (posterior, rssm_state, losses) = self._train_model(
            model_state,
            jax.tree_map(
                lambda rng: jax.random.split(rng, num_envs * num_agents).reshape(
                    num_envs, num_agents
                ),
                rngs,
            ),
            data,
            rssm_state,
        )

        # Update the model state and key.
        self.key = key
        self.model_state = model_state

        return posterior, rssm_state, losses

    @staticmethod
    @functools.partial(jax.jit, static_argnums=(0,))
    def _train_policy(
        agent: WorldModelAgent,
        policy_state: DreamerTrainState,
        rngs: jax.Array,
        traj: Transition,
    ) -> Tuple[DreamerTrainState, ACLossInfo]:
        """Trains the policy (jitted)."""

        num_envs, num_agents = traj.action[0].shape[:2]
        actor_state, critic_state = policy_state

        @functools.partial(jax.vmap, in_axes=(None, 2), out_axes=((2, 2), 1))
        @functools.partial(jax.vmap, in_axes=(None, 1), out_axes=((1, 1), 0))
        def _get_advantages(params: FrozenDict, traj: Transition):
            variables = {"params": params, "aux": critic_state.aux}

            return agent.apply(
                variables,
                traj,
                mutable="aux",
                method=agent.compute_advantages,
            )

        # Update the policy parameters.
        def critic_loss_fn(params: FrozenDict, traj: Transition, targets: jax.Array):
            variables = {"params": params, "aux": critic_state.aux}

            @functools.partial(jax.vmap, in_axes=(None, 2, 2), out_axes=1)
            @functools.partial(jax.vmap, in_axes=(None, 1, 1), out_axes=0)
            def critic_apply_fn(
                variables: FrozenDict, traj: Transition, targets: jax.Array
            ):
                # print(jax.tree_map(lambda x: x.shape, variables["aux"]))
                return critic_state.apply_fn(
                    variables,
                    traj,
                    targets,
                    mutable="aux",
                )

            (critic_loss, critic_metric), aux = critic_apply_fn(
                variables, traj, targets
            )

            return critic_loss.mean(), (critic_metric, aux)

        (advantages, targets), aux = _get_advantages(critic_state.params, traj)
        critic_state = critic_state.replace(aux=jax.tree_map(jnp.mean, aux["aux"]))

        critic_grad_fn = critic_state.dynamic_scale.value_and_grad(
            critic_loss_fn, has_aux=True
        )
        critic_dynamic_scale, critic_finite, critic_aux, critic_grads = critic_grad_fn(
            critic_state.params, traj, targets
        )
        critic_loss, (critic_metric, critic_variables) = critic_aux

        new_critic_state = critic_state.apply_gradients(grads=critic_grads)

        # Update the policy state.
        critic_opt_state = jax.tree_map(
            functools.partial(jnp.where, critic_finite),
            new_critic_state.opt_state,
            critic_state.opt_state,
        )
        critic_params = jax.tree_map(
            functools.partial(jnp.where, critic_finite),
            new_critic_state.params,
            critic_state.params,
        )
        aux = jax.tree_map(jnp.mean, critic_variables["aux"])

        critic_state = new_critic_state.replace(
            opt_state=critic_opt_state,
            params=critic_params,
            aux=aux,
            dynamic_scale=critic_dynamic_scale,
        )

        def actor_loss_fn(
            params: FrozenDict, rng: jax.Array, traj: Transition, advantages: jax.Array
        ):
            variables = {"params": params, "aux": critic_state.aux}

            @functools.partial(jax.vmap, in_axes=(None, 1, 2, 2), out_axes=1)
            @functools.partial(jax.vmap, in_axes=(None, 0, 1, 1), out_axes=0)
            def actor_apply_fn(
                variables: FrozenDict,
                rng: jax.Array,
                traj: Transition,
                advantages: jax.Array,
            ):
                # print(jax.tree_map(lambda x: x.shape, variables["aux"]))
                return actor_state.apply_fn(
                    variables,
                    traj,
                    advantages,
                    rngs=rng,
                )

            actor_loss, actor_metric = actor_apply_fn(
                variables,
                jax.tree_map(
                    lambda rng: jax.random.split(rng, num_envs * num_agents).reshape(
                        num_envs, num_agents
                    ),
                    rngs,
                ),
                traj,
                advantages,
            )
            return actor_loss.mean(), actor_metric

        actor_grad_fn = actor_state.dynamic_scale.value_and_grad(
            actor_loss_fn, has_aux=True
        )
        actor_dynamic_scale, actor_finite, actor_aux, actor_grads = actor_grad_fn(
            actor_state.params, rngs, traj, advantages
        )
        actor_loss, actor_metric = actor_aux
        new_actor_state = actor_state.apply_gradients(grads=actor_grads)

        # Update the policy state.
        actor_opt_state = jax.tree_map(
            functools.partial(jnp.where, actor_finite),
            new_actor_state.opt_state,
            actor_state.opt_state,
        )
        actor_params = jax.tree_map(
            functools.partial(jnp.where, actor_finite),
            new_actor_state.params,
            actor_state.params,
        )

        actor_state = new_actor_state.replace(
            opt_state=actor_opt_state,
            params=actor_params,
            dynamic_scale=actor_dynamic_scale,
        )

        policy_loss = actor_loss + critic_loss

        loss_info = ACLossInfo(
            total_loss=policy_loss,
            actor_loss=actor_metric[0],
            critic_loss=critic_metric[0],
            entropy=actor_metric[1],
            actor_grad_norm=jnp.array(optax.global_norm(actor_grads)),
            critic_grad_norm=jnp.array(optax.global_norm(critic_grads)),
        )
        jax.debug.print("ACTOR GRADS {x}", x=optax.global_norm(actor_grads))

        loss_info = jax.tree_map(jnp.mean, loss_info)

        return (actor_state, critic_state), loss_info

    def train_policy(self, traj: Transition) -> ACLossInfo:
        """Trains the policy."""
        # Get the policy state and key.
        policy_state = self.policy_state
        key = self.key
        agent = self.agent

        # Train the policy.
        action_key, key = jax.random.split(key, 2)
        rngs = {"action": action_key}
        policy_state, policy_metric = self._train_policy(
            agent, policy_state, rngs, traj
        )
        # if jnp.isinf(optax.global_norm(policy_metric.grad_norm)) or jnp.isnan(optax.global_norm(policy_metric.grad_norm)):
        #     return
        # Update the policy state and key.
        self.policy_state = policy_state
        self.key = key

        return policy_metric

    @staticmethod
    @functools.partial(jax.jit, static_argnums=(0,))
    def _update_policy(
        agent: WorldModelAgent, policy_state: DreamerTrainState
    ) -> DreamerTrainState:
        """Updates the policy (jitted)."""
        # Update the policy variables.
        actor_state, critic_state = policy_state
        variables = {"params": critic_state.params, "aux": critic_state.aux}
        _, variables = agent.apply(
            variables,
            mutable=["params", "aux"],
            method=agent.update_policy,
        )

        # Update the policy state.
        params = variables["params"]
        aux = variables["aux"]
        return (actor_state, critic_state.replace(params=params, aux=aux))

    def update_policy(self):
        """Updates the policy."""
        # Get the agent and policy state.
        agent = self.agent
        policy_state = self.policy_state

        # Update the policy.
        policy_state = self._update_policy(agent, policy_state)

        # Update the policy state.
        self.policy_state = policy_state

    def train(
        self,
        buffer_state,
        rng,
        rssm_state: RSSMState,
        step: int,
        model_epochs: int,
        policy_epochs: int,
        policy_update_per_epoch: int,
    ) -> Tuple[RSSMState, Tuple[RSSMLossInfo, ACLossInfo]]:
        """Trains the agent."""
        # Train the agent.
        train_metric = None

        train_steps = self.should_train(step)
        for step in range(train_steps):
            data = None
            if self.replay_buffer.can_sample(buffer_state):

                data = self.replay_buffer.sample(buffer_state, rng).experience
                data = jax.tree_map(lambda x: jnp.swapaxes(x, 0, 1), data)
                data = jax.device_put(data, jax.local_devices()[0])

                posterior, rssm_state, model_metric = self.train_model(data, rssm_state)

                flatten = lambda x: jax.tree_map(
                    lambda y: y.reshape(-1, *y.shape[2:]), x
                )
                traj = self.imagine(flatten(posterior), flatten(data.termination))

                policy_metric = self.train_policy(traj)
                self.update_policy()

                # Define the train metric.
                train_metric = (model_metric, policy_metric)

        return rssm_state, train_metric


class SupervisedDreamerAgent:

    def __init__(
        self,
        config: BaseDataType,
        env: Any,
        eval_env: Any,
        replay_buffer: TrajectoryBuffer,
        world_model_fn: Callable,
        rollout_fn: Callable,
        train_ratio: int,
        **kwargs: Any
    ) -> None:

        # here we take only supervised learning part
        self.name = self.__class__.__name__
        # self.num_vec_ens = config.num_envs * config.num_agents
        self.config = config
        self.rollout_fn = rollout_fn

        self.should_train = Every(every=train_ratio)

        self.devices = jax.local_devices()
        env, self.env_specs = env

        self.agent: WorldModelAgent = SupervisedWorldModelAgent(
            world_model_fn,
            imagine_horizon=config.world_model_config.imagination_horizon,
        )

        # Key
        self.key = jax.random.key(config.seed)

        # Model state
        self.model_state = self._init_world_model_state(config)

        # not to overflow gpu ram, allocating replay memeory on cpu
        self.replay_buffer = replay_buffer.replace(
            init=jax.jit(replay_buffer.init, backend="cpu"),
            add=jax.jit(replay_buffer.add, donate_argnums=0, backend="cpu"),
            sample=jax.jit(replay_buffer.sample, backend="cpu"),
            can_sample=jax.jit(replay_buffer.can_sample, backend="cpu"),
        )

        self.num_envs = self.config.num_envs
        self.num_agents = self.config.num_agents
        self.num_vecenvs = self.num_envs * self.num_agents

    def initial_state_world_model(self, batch_size: int) -> Tuple[RSSMState, Tuple]:
        variables = {"params": self.model_state.params, "aux": self.model_state.aux}
        return self.agent.apply(
            variables, batch_size, method=self.agent.initial_state_world_model
        )

    def _init_world_model_state(self, config: BaseDataType) -> DreamerTrainState:
        param_key, post_key, prior_key, self.key = jax.random.split(self.key, 4)
        rngs = {"params": param_key, "posterior": post_key, "prior": prior_key}
        data = {
            "observation": jnp.zeros(
                (1, *self.env_specs["observation_space"]), jnp.float32
            ),
            "action": jnp.zeros((1, self.env_specs["action_space"]), jnp.float32),
            "reward": jnp.zeros((1,), jnp.float32),
            "termination": jnp.zeros((1,), jnp.uint8),
            "first": jnp.zeros((1,), jnp.float32),  # ??????
        }
        variables = self.agent.init(
            rngs, rssm_state=None, **data, method=self.agent.world_model_loss
        )

        # Define the model state.
        params = variables["params"]

        tx = optax.chain(
            optax.clip_by_global_norm(config.world_model_config.max_grad_norm),
            optax.adam(learning_rate=config.world_model_config.lr, eps=1e-8),
        )
        dynamic_scale = DynamicScale()

        model_state = DreamerTrainState.create(
            apply_fn=functools.partial(
                self.agent.apply, method=self.agent.world_model_loss
            ),
            params=params,
            tx=tx,
            aux={},
            dynamic_scale=dynamic_scale,
        )

        return model_state

    @staticmethod
    @functools.partial(jax.jit, static_argnums=(0,))
    def _act(
        agent: WorldModelAgent,
        variables: FrozenDict,
        rngs: jax.Array,
        observation: jax.Array,
        termination: jax.Array,
        firsts: jax.Array,
        rssm_state: RSSMState = None,
    ) -> ActorOutput:
        num_envs, num_agents = observation.shape[:2]

        @functools.partial(jax.vmap, in_axes=(None, 1, 1, 1, 1, 1), out_axes=1)
        @functools.partial(jax.vmap, in_axes=(None, 0, 0, 0, 0, 0), out_axes=0)
        def apply_fn(
            variables: FrozenDict,
            rng: jax.Array,
            observation: jax.Array,
            termination: jax.Array,
            firsts: jax.Array,
            rssm_state: RSSMState,
        ):
            return agent.apply(
                variables,
                observation,
                termination,
                firsts,
                rssm_state,
                method=agent.act,
                rngs=rng,
            )

        return apply_fn(
            variables,
            jax.tree_map(
                lambda rng: jax.random.split(rng, num_envs * num_agents).reshape(
                    num_envs, num_agents
                ),
                rngs,
            ),
            observation,
            termination,
            firsts,
            rssm_state,
        )

    def act(
        self,
        observation: jax.Array,
        termination: jax.Array,
        firsts: jax.Array,
        rssm_state: RSSMState = None,
    ) -> ActorOutput:
        # Get the agent and key.
        agent = self.agent
        key = self.key

        # Sample an action.
        variables = {"params": self.model_state.params, "aux": self.model_state.aux}
        post_key, prior_key, action_key, key = jax.random.split(key, 4)
        rngs = {"posterior": post_key, "prior": prior_key, "action": action_key}
        action, state = self._act(
            agent,
            variables,
            rngs,
            observation,
            termination,
            firsts,
            rssm_state=rssm_state,
        )
        # Update the key.
        self.key = key
        return action, state

    @staticmethod
    @functools.partial(jax.jit, static_argnums=(0,))
    def _imagine(
        agent: WorldModelAgent,
        variables: FrozenDict,
        rngs: jax.Array,
        rssm_state: RSSMState,
        termination: jax.Array,
    ) -> Transition:

        num_envs, num_agents = termination.shape[:2]

        @functools.partial(jax.vmap, in_axes=(None, 1, 1, 1), out_axes=2)
        @functools.partial(jax.vmap, in_axes=(None, 0, 0, 0), out_axes=1)
        def apply_fn(
            variables: FrozenDict,
            rng: jax.Array,
            rssm_state: RSSMState,
            termination: jax.Array,
        ):
            return agent.apply(
                variables, rssm_state, termination, method=agent.imagine, rngs=rng
            )

        return apply_fn(
            variables,
            jax.tree_map(
                lambda rng: jax.random.split(rng, num_envs * num_agents).reshape(
                    num_envs, num_agents
                ),
                rngs,
            ),
            rssm_state,
            termination,
        )

    def imagine(self, posterior: RSSMState, termination: jax.Array) -> Transition:
        # Get the agent and key.
        agent = self.agent
        key = self.key

        # Run an imagination.
        variables = {"params": self.model_state.params, "aux": self.model_state.aux}
        post_key, prior_key, action_key, key = jax.random.split(key, 4)
        rngs = {"posterior": post_key, "prior": prior_key, "action": action_key}
        traj = self._imagine(agent, variables, rngs, posterior, termination)
        # Update the key.
        self.key = key

        return traj

    @staticmethod
    @jax.jit
    def _train_model(
        model_state: DreamerTrainState,
        rngs: jax.Array,
        data: Transition,
        rssm_state: RSSMState,
    ) -> Tuple[DreamerTrainState, Tuple[RSSMState, RSSMState, RSSMLossInfo]]:

        # Update the model parameters.
        def loss_fn(params: FrozenDict):
            variables = {"params": params, "aux": model_state.aux}

            @functools.partial(
                jax.vmap, in_axes=(None, 1, 2, 1), out_axes=(1, (2, 1, 1))
            )
            @functools.partial(
                jax.vmap, in_axes=(None, 0, 1, 0), out_axes=(0, (1, 0, 0))
            )
            def apply_fn(
                variables: FrozenDict,
                rngs: jax.Array,
                data: Transition,
                rssm_state: RSSMState,
            ):

                return model_state.apply_fn(
                    variables,
                    rssm_state,
                    data.observation,
                    data.action,
                    data.reward,
                    data.termination,
                    data.is_first,
                    rngs=rngs,
                )

            total_loss, aux = apply_fn(
                variables,
                rngs,
                data,
                rssm_state,
            )

            return total_loss.mean(), aux

        grad_fn = model_state.dynamic_scale.value_and_grad(loss_fn, has_aux=True)
        dynamic_scale, finite, aux, grads = grad_fn(model_state.params)
        _, (posterior, rssm_state, losses) = aux

        new_model_state = model_state.apply_gradients(grads=grads)

        # finite_grads =
        _grads = jax.tree_map(
            functools.partial(jnp.where, finite),
            grads,
            jax.tree_map(lambda g: jnp.zeros_like(g), grads),
        )
        losses = losses._replace(grad_norm=optax.global_norm(grads))

        # Update the model state.
        opt_state = jax.tree_map(
            functools.partial(jnp.where, finite),
            new_model_state.opt_state,
            model_state.opt_state,
        )
        params = jax.tree_map(
            functools.partial(jnp.where, finite),
            new_model_state.params,
            model_state.params,
        )
        model_state = new_model_state.replace(
            opt_state=opt_state,
            params=params,
            dynamic_scale=dynamic_scale,
        )

        return model_state, (posterior, rssm_state, losses)

    def train_model(
        self,
        data: Transition,
        rssm_state: RSSMState | None = None,
    ) -> Tuple[DreamerTrainState, Tuple]:
        """Trains the model."""
        # Get the model state and key.
        model_state = self.model_state
        key = self.key
        num_envs, num_agents = data.action[0].shape[:2]
        # Train the model.
        post_key, prior_key, key = jax.random.split(key, 3)
        rngs = {"posterior": post_key, "prior": prior_key}
        model_state, (posterior, rssm_state, losses) = self._train_model(
            model_state,
            jax.tree_map(
                lambda rng: jax.random.split(rng, num_envs * num_agents).reshape(
                    num_envs, num_agents
                ),
                rngs,
            ),
            data,
            rssm_state,
        )

        # Update the model state and key.
        self.key = key
        self.model_state = model_state

        return posterior, rssm_state, losses

    def train(
        self,
        buffer_state,
        rng,
        rssm_state: RSSMState,
        step: int,
        model_epochs: int,
        policy_epochs: int,
        policy_update_per_epoch: int,
    ) -> Tuple[RSSMState, Tuple[RSSMLossInfo, ACLossInfo]]:
        """Trains the agent."""
        # Train the agent.
        train_metric = None

        for step in range(self.should_train(step)):
            data = None
            if self.replay_buffer.can_sample(buffer_state):
                data = self.replay_buffer.sample(buffer_state, rng).experience
                data = jax.tree_map(lambda x: jnp.swapaxes(x, 0, 1), data)
                data = jax.device_put(data, jax.local_devices()[0])
                posterior, rssm_state, model_metric = self.train_model(data, rssm_state)

                flatten = lambda x: jax.tree_map(
                    lambda y: y.reshape(-1, *y.shape[2:]), x
                )
                # Define the train metric.
                train_metric = (model_metric,)

        return rssm_state, train_metric
