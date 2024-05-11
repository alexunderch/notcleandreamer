import math
from typing import Any, Callable, Tuple

import gymnasium as gym
import hydra
import jax
import jax.numpy as jnp
import numpy as np
import pogema
import wandb
from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm

from noncleandreamer.ac import create_ac_dreamer as policy_fn
from noncleandreamer.custom_types import BaseDataType, Transition
from noncleandreamer.dreamer import DreamerAgent, create_item_buffer
from noncleandreamer.networks import (
    create_discrete_lin_ppo_policy,
    create_lin_rssm_model,
)


class PogemaEnv:

    def __init__(self, env) -> None:
        self.env = env
        self.num_agents = env.unwrapped.get_num_agents()
        self._done: bool = True
        self._episode_return = jnp.zeros((self.num_agents,))
        self._episode_length = jnp.zeros((self.num_agents,), dtype=jnp.uint32)

    def reset(self, seed=None, options=None):
        return self.env.reset()

    def step(self, action: int):
        if self._done:
            # Reset the environment.
            obs, info = self.env.reset()
            reward = [0.0 for _ in range(self.num_agents)]
            done = [False for _ in range(self.num_agents)]
            first = True

            # Update the statistics.
            self._done = np.all(done)
            self._episode_return = jnp.zeros((self.num_agents,))
            self._episode_length = jnp.zeros((self.num_agents,), dtype=jnp.uint32)

        else:
            # Step the environment.
            obs, reward, terminated, truncated, info = self.env.step(action)
            done = np.maximum(terminated, truncated)
            first = False

            # Update the statistics.
            self._done = np.all(done)
            self._episode_return += jnp.array(reward)
            self._episode_length += 1
            # Return the episode return and length.
            for _ind, _done in enumerate(done):
                if _done:
                    if "metrics" not in info[0]:
                        info[0]["metrics"] = {}

                    info[0]["metrics"][f"episode_return_{_ind}"] = self._episode_return[
                        _ind
                    ].astype(float)
                    info[0]["metrics"][f"episode_length_{_ind}"] = self._episode_length[
                        _ind
                    ].astype(float)

        return obs, reward, done, first, info


def make_pogema_env(env_name, **config_kwargs):
    str2env = {
        "Easy8x8": pogema.Easy8x8,
        "Normal8x8": pogema.Normal8x8,
        "Hard8x8": pogema.Hard8x8,
        "ExtraHard8x8": pogema.ExtraHard8x8,
        "Easy16x16": pogema.Easy16x16,
        "Normal16x16": pogema.Normal16x16,
        "Hard16x16": pogema.Hard16x16,
        "ExtraHard16x16": pogema.ExtraHard16x16,
        "Easy32x32": pogema.Easy32x32,
        "Normal32x32": pogema.Normal32x32,
        "Hard32x32": pogema.Hard32x32,
        "ExtraHard32x32": pogema.ExtraHard32x32,
        "Easy64x64": pogema.Easy64x64,
        "Normal64x64": pogema.Normal64x64,
        "Hard64x64": pogema.Hard64x64,
        "ExtraHard64x64": pogema.ExtraHard64x64,
    }
    return PogemaEnv(pogema.pogema_v0(grid_config=str2env[env_name](**config_kwargs)))


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


def make_pogema_step_fn(
    env: Any, agent: DreamerAgent, config: DreamerConfig
) -> Callable:

    def rollout(data, rssm_state, rollout_len: int) -> Tuple[Transition, bool]:
        observations = []
        terminations = []
        actions = []
        rewards = []
        firsts = []
        transform = lambda x: x.reshape(
            config.num_envs, config.num_agents, *x.shape[1:]
        )  # noqa: E731

        observation, reward, done, first, info, action = data

        for _ in tqdm(
            range(rollout_len), total=rollout_len, postfix="Collecting a rollout"
        ):
            observation = jnp.array(observation)
            observations.append(
                transform(observation).reshape(
                    config.num_envs, *observation.shape[:-3], -1
                )
            )
            terminations.append(transform(jnp.array(done)))
            rewards.append(transform(jnp.array(reward)))
            actions.append(action.reshape(config.num_envs, config.num_agents, -1))
            firsts.append(
                jnp.array([first] * env.num_agents).reshape(
                    config.num_envs, config.num_agents
                )
            )

            act_out, rssm_state = agent.act(
                transform(jnp.array(observation)).reshape(
                    config.num_envs, *observation.shape[:-3], -1
                ),
                jnp.array(done).reshape(config.num_envs, config.num_agents),
                jnp.array([first] * env.num_agents).reshape(
                    config.num_envs, config.num_agents
                ),
                rssm_state,
            )

            action = act_out.action
            observation, reward, done, first, info = env.step(
                jnp.argmax(action, -1).reshape(-1).tolist()
            )
            if "metrics" in info[0]:
                wandb.log(
                    {
                        k: v.item() if isinstance(v, jnp.ndarray) else v
                        for k, v in info[0]["metrics"].items()
                    }
                )

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

    return rollout


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


@hydra.main(config_path="./configs/", config_name="pogema.yaml", version_base="1.2")
def main(cfg: DictConfig) -> None:
    OmegaConf.set_struct(cfg, False)
    dict_config = OmegaConf.to_container(cfg, resolve=True)

    wandb.init(
        config=dict_config,
        group=f"s_{dict_config['env_id']}_{dict_config['seed']}",
        **dict_config.pop("wandb"),
    )

    env = make_pogema_env(
        env_name=dict_config["env_id"], on_target=dict_config["on_target"]
    )
    env_specs = {
        "observation_space": (math.prod(env.env.observation_space.shape),),
        "action_space": env.env.action_space.n,
    }

    config = DreamerConfig(
        policy_config=PolicyConfig(**dict_config["policy"]),
        world_model_config=WMConfig(
            **dict_config["wm"], num_actions=env_specs["action_space"]
        ),
        replay_buffer_config=ReplayBufferConfig(
            **dict_config["replay_buffer"], add_batch_size=1
        ),
        policy_network_config=dict_config["policy_network"],
        wm_network_config=dict_config["wm_network"],
        num_envs=1,
        num_agents=env.num_agents,
    )
    min_step = int(
        config.replay_buffer_config.sample_batch_size
        * config.replay_buffer_config.sample_sequence_length
    )
    train_ratio = dict_config["train_ratio"] / min_step

    agent = create_dreamer_agent(
        config,
        create_discrete_lin_ppo_policy,
        create_lin_rssm_model,
        env_specs,
        train_ratio,
        None,
    )

    rssm_state = jax.tree_util.tree_map(
        lambda x: x.reshape((1, env.num_agents, *x.shape[1:])),
        agent.initial_state_world_model(env.num_agents),
    )

    rollout_fn = make_pogema_step_fn(env, agent, config)

    sample_transition = Transition(
        state=None,
        observation=jnp.zeros((config.num_agents, *env_specs["observation_space"])),
        termination=jnp.zeros((config.num_agents,), dtype=jnp.bool_),
        action=jnp.zeros((config.num_agents, env_specs["action_space"])),
        reward=jnp.zeros((config.num_agents,)),
        is_first=jnp.ones((config.num_agents,), dtype=jnp.bool_),
    )

    buffer_state = agent.replay_buffer.init(sample_transition)

    observation, info = env.reset()
    action = jnp.tile(
        jax.nn.one_hot(env.env.action_space.sample(), env_specs["action_space"])[
            jnp.newaxis
        ],
        (1, env.num_agents, 1),
    )
    observation, reward, done, first, info = env.step(
        jnp.argmax(action, -1).reshape(-1).tolist()
    )
    env_data = (observation, reward, done, first, info, action)

    env_steps = 0
    rng = jax.random.key(dict_config["seed"] + 1)
    rollout_len = dict_config["env_kwargs"]["max_episode_steps"]
    # rollout_len = env.env.unwrapped.grid_config.max_episode_steps
    # train_ratio = dict_config["train_ratio"] // rollout_len
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


if __name__ == "__main__":
    main()
