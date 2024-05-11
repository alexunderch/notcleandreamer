from typing import Any, Callable, Tuple

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


class AtariEnv(gym.Wrapper):

    def __init__(self, env) -> None:
        super().__init__(env)
        self._done: bool = True
        self._episode_return: float = 0
        self._episode_length: int = 0

    def reset(self, seed=None, options=None):
        self.reset_kwargs = dict(seed=seed, options=options)
        return super().reset(seed=seed, options=options)

    def step(self, action: int):
        if self._done:
            # Reset the environment.
            obs, info = super().reset(**self.reset_kwargs)
            reward = 0.0
            done = False
            first = True

            # Update the statistics.
            self._done = done
            self._episode_return = 0
            self._episode_length = 0

        else:
            # Step the environment.
            obs, reward, terminated, truncated, info = super().step(action)
            done = np.maximum(terminated, truncated)
            first = False

            # Update the statistics.
            self._done = done
            self._episode_return += reward
            self._episode_length += 1

            # Return the episode return and length.
            if done:
                info["episode_return"] = self._episode_return
                info["episode_length"] = self._episode_length

        return obs, reward, done, first, info


def make_gymnasium_env(
    env_name: str,
    max_episode_steps: int,
    num_envs: int,
    frame_stack: int,
    **env_kwargs: Any,
):

    env = make(env_name, render_mode="rgb_array", full_action_space=False)
    env = wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)
    env = wrappers.AutoResetWrapper(env)
    env = AtariEnv(env)
    env = wrappers.AtariPreprocessing(
        env,
        screen_size=64,
        noop_max=0,
        frame_skip=4,
        grayscale_obs=False,
        scale_obs=True,
        terminal_on_life_loss=False,
    )

    if frame_stack > 1:
        env = ss.frame_stack_v1(env, frame_stack)

    env = ss.gym_vec_env_v0(env, num_envs=num_envs, multiprocessing=False)

    env.single_observation_space = env.observation_space
    env.single_action_space = env.action_space
    env.is_vector_env = True

    return env


class WMConfig(BaseDataType):
    discrete_latent_dim: int

    imagination_horizon: int

    seed: int
    num_actions: int

    lr: float
    max_grad_norm: float

    free_kl: float
    kl_balance_rep: float
    kl_balance_dyn: float


class PolicyConfig(BaseDataType):

    target_update_period: int
    target_update_tau: float

    gamma: float
    lambda_: float
    normalize_returns: bool
    vf_coeff: float
    ent_coeff: float

    seed: int
    logger_freq: int

    lr: float
    anneal_lr: bool
    max_grad_norm: float


class ReplayBufferConfig(BaseDataType):
    max_length: int
    min_length: int
    add_batch_size: int
    sample_sequence_length: int
    sample_batch_size: int


class DreamerConfig(BaseDataType):
    policy_config: PolicyConfig
    world_model_config: WMConfig
    replay_buffer_config: ReplayBufferConfig
    policy_network_config: BaseDataType
    wm_network_config: BaseDataType
    num_envs: int
    num_agents: int
    latent_dim: int = 1
    seed: int = 0


def make_atati_step_fn(
    env: Any, agent: DreamerAgent, config: DreamerConfig
) -> Callable:

    def rollout(
        data, rssm_state, rollout_len: int, return_metrics: bool = False
    ) -> Tuple[Transition, bool]:
        observations = []
        terminations = []
        actions = []
        rewards = []
        firsts = []
        metrics = {"episode_return": [], "episode_length": []}
        transform = lambda x: x.reshape(
            config.num_envs, config.num_agents, *x.shape[1:]
        )  # noqa: E731

        observation, reward, done, first, info, action = data

        for _ in range(rollout_len):
            observation = jnp.array(observation)
            observations.append(transform(observation))
            terminations.append(transform(jnp.array(done)))
            rewards.append(transform(jnp.array(reward)))
            actions.append(action)
            firsts.append(jnp.array(first).reshape(config.num_envs, config.num_agents))

            act_out, rssm_state = agent.act(
                transform(jnp.array(observation)),
                done.reshape(config.num_envs, config.num_agents),
                jnp.array(first).reshape(config.num_envs, config.num_agents),
                rssm_state,
            )
            action = act_out.action
            observation, reward, done, first, info = env.step(
                jnp.argmax(action, -1).reshape(-1).tolist()
            )

            dones = 0
            for _ind, _done in enumerate(done):
                if _done:
                    dones += 1
                    metrics["episode_return"].append(
                        info["final_info"][_ind]["episode_return"]
                    )
                    metrics["episode_length"].append(
                        info["final_info"][_ind]["episode_length"]
                    )

                    rollout_metric = {
                        "Metrics/episode_return": info["final_info"][_ind][
                            "episode_return"
                        ],
                        "Metrics/episode_length": info["final_info"][_ind][
                            "episode_length"
                        ],
                    }
                    wandb.log(rollout_metric)

        if not return_metrics:
            return (
                Transition(
                    state=None,
                    observation=jnp.stack(observations, axis=0),
                    termination=jnp.stack(terminations, axis=0),
                    action=jnp.stack(actions, axis=0),
                    reward=jnp.stack(rewards, axis=0),
                    info=None,
                    is_first=jnp.stack(firsts, axis=0),
                ),
                (observation, reward, done, first, info, action),
                rssm_state,
            )

        return metrics, (observation, reward, done, first, info, action), rssm_state

    return rollout


def evaluate(env, agent, config, eval_config: dict):
    env, env_specs = env
    metrics = {"episode_return": [], "episode_length": []}
    rollout_fn = make_atati_step_fn(env, agent, config)
    rssm_state = jax.tree_util.tree_map(
        lambda x: x.reshape((config.num_envs, 1, *x.shape[1:])),
        agent.initial_state_world_model(config.num_envs),
    )
    observation, info = env.reset()
    action = jax.nn.one_hot(
        env.single_action_space.sample(), env_specs["action_space"]
    )[:, jnp.newaxis]
    observation, reward, done, first, info = env.step(
        jnp.argmax(action, -1).reshape(-1).tolist()
    )
    env_data = (observation, reward, done, first, info, action)

    for _ in range(int(eval_config["eval_eps"])):
        ep_metrics, env_data, rssm_state = rollout_fn(
            env_data, rssm_state, None, return_metrics=True
        )
        for metrics_key in metrics:
            metrics[metrics_key].extend(ep_metrics[metrics_key])

    eval_metrics = {}
    for metrics_key in metrics:
        m = metrics[metrics_key]
        eval_metrics[f"{metrics_key}_mean"] = jnp.mean(m)
        eval_metrics[f"{metrics_key}_std"] = jnp.mean(m)

    return eval_metrics


def create_dreamer_agent(
    config: DreamerConfig,
    policy_network_factory: Callable,
    wm_network_factory: Callable,
    env_specs: dict,
    train_ratio: int,
    rollout_fn_factory: Callable = None,
) -> DreamerAgent:

    action_space_dim = env_specs["action_space"]
    obs_space_dim = env_specs["observation_space"]

    actor, critic, slow_critic = policy_network_factory(
        **config.policy_network_config, action_space_dim=action_space_dim
    )
    _world_model_fn = wm_network_factory(
        **config.wm_network_config,
        action_space_dim=action_space_dim,
        obs_space_dim=obs_space_dim,
        world_model_config=config.world_model_config,
    )

    latent_dim = (
        config.wm_network_config["stoch_discrete_dim"]
        * config.wm_network_config["num_categories"]
        + config.wm_network_config["state_dim"]
    )
    config = config._replace(latent_dim=latent_dim)

    return DreamerAgent(
        config=config,
        env=(None, env_specs),
        eval_env=None,
        replay_buffer=create_item_buffer(**config.replay_buffer_config._asdict()),
        world_model_fn=_world_model_fn,
        policy_fn=policy_fn(actor, critic, slow_critic, config.policy_config),
        rollout_fn=rollout_fn_factory,
        train_ratio=train_ratio,
    )


@hydra.main(config_path="./configs/", config_name="atari.yaml", version_base="1.2")
def main(cfg: DictConfig) -> None:
    OmegaConf.set_struct(cfg, False)
    dict_config = OmegaConf.to_container(cfg, resolve=True)

    wandb.init(
        config=dict_config,
        group=f"s_{dict_config['env_id']}_{dict_config['seed']}",
        **dict_config.pop("wandb"),
    )

    env = make_gymnasium_env(
        env_name=dict_config["env_id"], **dict_config["env_kwargs"]
    )
    eval_env = make_gymnasium_env(
        env_name=dict_config["env_id"], **dict_config["env_kwargs"]
    )

    env_specs = {
        "observation_space": env.single_observation_space.shape[1:],
        "action_space": env.single_action_space.nvec[0],
    }

    config = DreamerConfig(
        policy_config=PolicyConfig(**dict_config["policy"]),
        world_model_config=WMConfig(
            **dict_config["wm"], num_actions=env_specs["action_space"]
        ),
        replay_buffer_config=ReplayBufferConfig(
            **dict_config["replay_buffer"],
            add_batch_size=dict_config["env_kwargs"]["num_envs"],
        ),
        policy_network_config=dict_config["policy_network"],
        wm_network_config=dict_config["wm_network"],
        num_envs=dict_config["env_kwargs"]["num_envs"],
        num_agents=1,
    )
    min_step = int(
        config.replay_buffer_config.sample_batch_size
        * config.replay_buffer_config.sample_sequence_length
    )
    eval_freq = dict_config["eval"].pop("eval_every")
    train_ratio = dict_config["train_ratio"] / min_step
    agent = create_dreamer_agent(
        config,
        create_discrete_lin_ppo_policy,
        create_conv_rssm_model,
        env_specs,
        train_ratio,
        None,
    )

    rssm_state = jax.tree_util.tree_map(
        lambda x: x.reshape((config.num_envs, 1, *x.shape[1:])),
        agent.initial_state_world_model(config.num_envs),
    )

    rollout_fn = make_atati_step_fn(env, agent, config)

    sample_transition = Transition(
        state=None,
        observation=jnp.zeros((1, *env_specs["observation_space"])),
        termination=jnp.zeros((1,), dtype=jnp.bool_),
        action=jnp.zeros((1, env_specs["action_space"])),
        reward=jnp.zeros((1,)),
        is_first=jnp.ones((1,), dtype=jnp.bool_),
    )

    buffer_state = agent.replay_buffer.init(sample_transition)

    observation, info = env.reset()
    action = jax.nn.one_hot(
        env.single_action_space.sample(), env_specs["action_space"]
    )[:, jnp.newaxis]
    observation, reward, done, first, info = env.step(
        jnp.argmax(action, -1).reshape(-1).tolist()
    )
    env_data = (observation, reward, done, first, info, action)

    env_steps = 0
    rng = jax.random.key(dict_config["seed"] + 1)
    min_step = int(
        config.replay_buffer_config.sample_batch_size
        * config.replay_buffer_config.sample_sequence_length
    )
    rollout_len = dict_config["env_kwargs"]["max_episode_steps"]
    # Train
    for step in range(1, int(dict_config["total_steps"]) + 1):

        rollout, env_data, rssm_state = rollout_fn(env_data, rssm_state, rollout_len)

        if not buffer_state.is_full.all():
            buffer_state = agent.replay_buffer.add(
                buffer_state,
                jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), rollout),
            )

        wandb.log({"step": step})
        env_steps += rollout_len
        wandb.log({"env_step": env_steps})

        _, rng = jax.random.split(rng)

        _, train_metric = agent.train(
            buffer_state,
            rng,
            None,
            step,
            dict_config["model_epochs"],
            dict_config["policy_epochs"],
            dict_config["policy_update_per_epoch"],
        )

        if (env_steps + 1) % (
            train_ratio * rollout_len
        ) == 0 and train_metric is not None:
            wandb.log(
                {
                    f"WM/{key}": m_value
                    for key, m_value in jax.tree_map(jnp.mean, train_metric[0])
                    ._asdict()
                    .items()
                }
            )
            wandb.log(
                {
                    f"AC/{key}": m_value
                    for key, m_value in jax.tree_map(jnp.mean, train_metric[1])
                    ._asdict()
                    .items()
                }
            )

        # if step % eval_freq == 0:
        #     wandb.log({f"EvalMetric/{key}": m_value for key, m_value in evaluate((eval_env, env_specs), agent, config, dict_config["eval"])})


if __name__ == "__main__":
    main()

    # TODO: clean training

# def make_train(
#     env: Environment,
#     env_params: EnvParams,
#     config: TrainConfig,
# ):
#     def train(
#         rng: jax.Array,
#         train_state: TrainState,
#     ):
#         # INIT ENV
#         rng, _rng = jax.random.split(rng)
#         reset_rng = jax.random.split(_rng, config.num_envs)

#         timestep = jax.vmap(env.reset, in_axes=(None, 0))(env_params, reset_rng)
#         prev_action = jnp.zeros(config.num_envs, dtype=jnp.int32)
#         grid_sh = (env_params.height, env_params.width)
#         stats = Stats(
#             jnp.zeros(grid_sh), jnp.zeros(grid_sh), jnp.zeros(grid_sh),
#             jnp.zeros(grid_sh)
#         )
#         int_state = (train_state, stats)

#         # TRAIN LOOP
#         @scan_tqdm(config.num_updates)
#         def _update_step(runner_state, _):
#             # COLLECT TRAJECTORIES
#             def _env_step(runner_state, _):
#                 rng, int_state, prev_timestep, prev_action = runner_state

#                 # SELECT ACTION
#                 rng, _rng = jax.random.split(rng)
#                 logits = jnp.zeros((
#                     config.num_envs, env.num_actions(env_params)
#                 ))
#                 action = jax.random.categorical(_rng, logits)

#                 # STEP ENV
#                 timestep = jax.vmap(env.step, in_axes=(None, 0, 0))(
#                     env_params, prev_timestep, action
#                 )
#                 transition = Transition(
#                     is_first=prev_timestep.first(),
#                     action=action,
#                     prev_action=prev_action,
#                     reward=prev_timestep.reward,
#                     obs=prev_timestep.observation,
#                     is_last=prev_timestep.last(),
#                     is_terminal=jnp.logical_not(prev_timestep.discount),
#                     pos=prev_timestep.state.agent.position,
#                     dir=prev_timestep.state.agent.direction,
#                 )
#                 runner_state = (rng, int_state, timestep, action)
#                 return runner_state, transition

#             # transitions: [seq_len, batch_size, ...]
#             runner_state, transitions = jax.lax.scan(
#                 _env_step, runner_state, None, config.num_steps
#             )

#             rng, int_state, timestep, prev_action = runner_state

#             # UPDATE NETWORK
#             def _update_epoch(update_state, _):
#                 def _update_minibatch(mini_state, batch_info):
#                     rng, (train_state, stats) = mini_state
#                     transitions = batch_info

#                     new_int_state, update_info, rng = update_wm(
#                         train_state=train_state,
#                         transitions=transitions,
#                         stats=stats,
#                         rng=rng,
#                     )
#                     mini_state = (rng, new_int_state)
#                     return mini_state, update_info

#                 rng, int_state, transitions = update_state

#                 # MINIBATCHES PREPARATION
#                 rng, _rng = jax.random.split(rng)
#                 permutation = jax.random.permutation(_rng, config.num_envs)
#                 # [seq_len, batch_size, ...]
#                 batch = transitions
#                 # [batch_size, seq_len, ...], as our model assumes
#                 batch = jax.tree_map(lambda x: x.swapaxes(0, 1), batch)

#                 shuffled_batch = jax.tree_map(
#                     lambda x: jnp.take(x, permutation, axis=0), batch
#                 )
#                 # [num_minibatches, minibatch_size, ...]
#                 minibatches = jax.tree_map(
#                     lambda x: jnp.reshape(
#                         x, (config.num_minibatches, -1) + x.shape[1:]
#                     ), shuffled_batch
#                 )

#                 mini_state = (rng, int_state)
#                 mini_state, update_info = jax.lax.scan(
#                     _update_minibatch, mini_state, minibatches,
#                 )
#                 rng, int_state = mini_state

#                 update_state = (rng, int_state, transitions)
#                 return update_state, update_info

#             # [seq_len, batch_size, num_layers, hidden_dim]
#             update_state = (rng, int_state, transitions)
#             update_state, loss_info = jax.lax.scan(
#                 _update_epoch, update_state, None, config.update_epochs
#             )

#             # averaging over minibatches then over epochs
#             loss_info = jax.tree_map(lambda x: x.mean(-1).mean(-1), loss_info)

#             rng, int_state = update_state[:2]

#             runner_state = (rng, int_state, timestep, prev_action)
#             return runner_state, loss_info

#         runner_state = (rng, int_state, timestep, prev_action)
#         runner_state, loss_info = jax.lax.scan(
#             _update_step, runner_state, jnp.arange(config.num_updates)
#         )
#         return {
#             "params": train_state.params,
#             "runner_state": runner_state,
#             "loss_info": loss_info
#         }

#     return jax.jit(train)


if __name__ == "__main__":
    main()
