import math
from typing import Callable, Optional

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp

from noncleandreamer.custom_types import (
    ActorOutput,
    BaseDataType,
    CriticOutput,
    Observation,
)
from noncleandreamer.distributions import Discrete, OneHotCategorical
from noncleandreamer.modules import (
    Precision,
    conv_decoder_model,
    conv_encoder_model,
    head_fn,
    initializers,
    latent_repr_fn2,
    linear_decoder_model,
    linear_encoder_model,
)
from noncleandreamer.rssm import posterior_fn, prior_fn, rssm_fn
from noncleandreamer.rssm import seq_proc_fn as rec_seq_proc_fn
from noncleandreamer.rssm import world_model_fn


class FeedForwardActor(nn.Module):
    representation_fn: Callable
    actor_fn: Callable
    unimix_ratio: float
    action_distribution: distrax.Distribution

    @nn.compact
    def __call__(
        self, x: Observation, action: Optional[jax.Array] = None
    ) -> ActorOutput:
        latent_repr = self.representation_fn()(x.agent_observation)
        actor_logits = self.actor_fn()(latent_repr).astype(jnp.float32)

        action_distribution = self.action_distribution(
            actor_logits, unimix_ratio=self.unimix_ratio
        )

        if x.action_mask is not None:
            unavail_actions = 1 - x.action_mask
            actor_logits = actor_logits - (unavail_actions * 1e10)

        if action is None:
            rng = self.make_rng("action")
            action = action_distribution.sample(seed=rng)
            log_prob = action_distribution.log_prob(action)
        else:
            log_prob = action_distribution.log_prob(action.astype(jnp.float32))

        entropy = action_distribution.entropy()
        return ActorOutput(None, actor_logits, action, log_prob, entropy)


class FeedForwardCritic(nn.Module):
    representation_fn: Callable
    critic_fn: Callable
    use_global_state: bool = False

    @nn.compact
    def __call__(self, x: Observation) -> CriticOutput:

        latent_repr = self.representation_fn()(
            x.agent_observation if not self.use_global_state else x.global_observation
        )
        critic_value = self.critic_fn()(latent_repr)

        return CriticOutput(
            None, Discrete(critic_value, use_symlog=True)
        )  # TODO: critic explodes


def create_discrete_lin_ppo_policy(
    base_ff_layers: int,
    hidden_dim: int,
    action_space_dim: int,
    num_discrete_bins: int,
    unimix_ratio: float,
    use_global_state: bool,
):
    representation_fn = linear_encoder_model(
        hidden_dims=[hidden_dim] * base_ff_layers,
        initializer=initializers.orthogonal(jnp.sqrt(2.0)),
        act_fn=nn.silu,
        use_bias=False,
        norm="layer"
    )

    actor_fn = head_fn(
        hidden_dims=[hidden_dim],
        act_fn=nn.silu,
        output_dim=action_space_dim,
        initializer=initializers.orthogonal(jnp.sqrt(2.0)),
        norm="none",
        use_bias=False
    )

    critic_fn = head_fn(
        hidden_dims=[hidden_dim],
        output_dim=num_discrete_bins,
        initializer=initializers.variance_scaling(
            0.0, mode="fan_avg", distribution="truncated_normal"
        ),
        act_fn=nn.silu,
        norm="none",
        use_bias=False
    )

    actor = FeedForwardActor(
        representation_fn=representation_fn,
        actor_fn=actor_fn,
        unimix_ratio=unimix_ratio,
        action_distribution=OneHotCategorical,
    )

    critic = FeedForwardCritic(
        representation_fn=representation_fn,
        critic_fn=critic_fn,
        use_global_state=use_global_state,
    )

    slow_critic = FeedForwardCritic(
        representation_fn=representation_fn,
        critic_fn=critic_fn,
        use_global_state=use_global_state,
    )

    return actor, critic, slow_critic


def create_conv_rssm_model(
    state_dim: int,
    stoch_discrete_dim: int,
    num_categories: int,
    unimix_ratio: float,
    reset_on_termination: bool,
    lstm_seed: int,
    hidden_dim: int,
    action_space_dim: int,
    obs_space_dim: tuple,
    base_channels: int,
    base_ff_layers: int,
    num_discrete_bins: int,
    min_res: int,
    world_model_config: BaseDataType,
) -> Callable:
    # TODO: add action and action mask decoders

    representation_fn = conv_encoder_model(
        base_shape=obs_space_dim,
        base_channels=base_channels,
        min_res=min_res,
        stride=2,
        kernel_size=4,
        initializer=initializers.xavier_normal(),
        act_fn=nn.silu,
        use_bias=True,
        norm="layer"
    )

    observation_decoder_fn = conv_decoder_model(
        base_shape=obs_space_dim,
        base_channels=base_channels,
        min_res=min_res,
        initializer=initializers.xavier_normal(),
        lin_initializer=initializers.orthogonal(jnp.sqrt(2.0)),
        norm="layer",
        kernel_size=4,
        stride=2,
        use_bias=True,
        act_fn=nn.silu,
    )

    prior_latent_repr_enc = head_fn(
        hidden_dims=[hidden_dim] * base_ff_layers,
        output_dim=hidden_dim,
        initializer=initializers.xavier_normal(),
        act_fn=nn.silu,
        name="Prior",
        use_bias=False,
        norm="none"
    )

    rssm_proj = head_fn(
        hidden_dims=[hidden_dim],
        output_dim=num_categories * stoch_discrete_dim,
        initializer=initializers.xavier_normal(),
        act_fn=nn.silu,
        use_bias=False,
        norm="none",
        name="RSSMProj",
    )

    posterior_latent_repr_enc = latent_repr_fn2(
        hidden_dims=[hidden_dim] * base_ff_layers,
        initializer=initializers.xavier_normal(),
        act_fn=nn.silu,
        norm="layer",
        use_bias=False,
        name=None,
    )

    prior_decoder_fn = head_fn(
        hidden_dims=[hidden_dim] * base_ff_layers,
        output_dim=hidden_dim,
        initializer=initializers.xavier_normal(),
        act_fn=lambda x: x,
        name="Prior",
        norm_type="layer",
    )

    posterior_decoder_fn = head_fn(
        hidden_dims=[hidden_dim] * base_ff_layers,
        output_dim=stoch_discrete_dim * stoch_discrete_dim,
        initializer=initializers.xavier_normal(),
        act_fn=nn.silu,
        use_bias=True,
        norm_type="layer",
        name="Posterior"
    )

    reward_decoder_fn = head_fn(
        hidden_dims=[hidden_dim],
        output_dim=num_discrete_bins,
        initializer=initializers.variance_scaling(0.0, "fan_avg", "truncated_normal"),
        act_fn=nn.silu,
        use_bias=True,
        norm="layer",
        name="Reward",
    )

    terminal_decoder_fn = head_fn(
        hidden_dims=[hidden_dim],
        output_dim=1,
        initializer=initializers.xavier_normal(),
        act_fn=nn.silu,
        use_bias=True,
        norm="layer",
        name="Reward",
    )

    ind_action_decoder_fn = head_fn(
        hidden_dims=[hidden_dim],
        output_dim=action_space_dim,
        initializer=initializers.xavier_normal(),
        act_fn=nn.silu,
        use_bias=True,
        norm="layer",
        name="IndAction",
    )

    lstm_config = dict(
        hidden_dim=state_dim,
        reset_on_termination=reset_on_termination,
        kernel_initializer=initializers.xavier_normal(),
        recurrent_initializer=initializers.xavier_normal(),
    )

    seq_proc_fn, seq_init_fn = rec_seq_proc_fn(lstm_seed, lstm_config)

    _prior_init_fn, _prior_fn = prior_fn(
        prior_latent_repr_enc, seq_init_fn, seq_proc_fn, prior_decoder_fn
    )
    _posterior_fn = posterior_fn(posterior_latent_repr_enc, posterior_decoder_fn)

    return world_model_fn(
        world_model_config,
        rssm_fn(
            (_prior_init_fn, _prior_fn),
            _posterior_fn,
            rssm_proj,
            state_dim,
            stoch_discrete_dim,
            num_categories,
            unimix_ratio,
        ),
        representation_fn,
        observation_decoder_fn,
        terminal_decoder_fn,
        reward_decoder_fn,
        False,
        ind_action_decoder_fn,
    )


def create_lin_rssm_model(
    state_dim: int,
    stoch_discrete_dim: int,
    num_categories: int,
    unimix_ratio: float,
    reset_on_termination: bool,
    lstm_seed: int,
    hidden_dim: int,
    action_space_dim: int,
    base_ff_layers: int,
    base_channels: int,
    num_discrete_bins: int,
    min_res: int,
    obs_space_dim: int,
    world_model_config: BaseDataType,
) -> Callable:
    # TODO: add action and action mask decoders

    representation_fn = linear_encoder_model(
        hidden_dims=[
            2**i * base_channels
            for i in range(int(math.log2(obs_space_dim[0] // min_res)))
        ],
        initializer=initializers.orthogonal(jnp.sqrt(2.0)),
        act_fn=nn.silu,
        norm="layer",
        use_bias=True,
        name="ObservationEncoder",
    )

    observation_decoder_fn = linear_decoder_model(
        hidden_dims=[
            2 ** (i - 1) * base_channels
            for i in reversed(range(int(math.log2(obs_space_dim[0] // min_res))))
        ]
        + [*obs_space_dim],
        initializer=initializers.xavier_normal(),
        act_fn=nn.silu,
        norm="layer",
        use_bias=True,
        name="ObservationDecoder",
    )

    prior_latent_repr_enc = head_fn(
        hidden_dims=[hidden_dim] * (base_ff_layers - 1),
        output_dim=hidden_dim,
        initializer=initializers.xavier_normal(),
        act_fn=nn.silu,
        norm="layer",
        use_bias=True,
        name="Prior",
    )

    posterior_latent_repr_enc = latent_repr_fn2(
        hidden_dims=[hidden_dim] * base_ff_layers,
        initializer=initializers.xavier_normal(),
        act_fn=nn.silu,
        norm="layer",
        use_bias=True,
        name="Posterior",
    )

    prior_decoder_fn = head_fn(
        hidden_dims=[hidden_dim] * base_ff_layers,
        output_dim=hidden_dim,
        initializer=initializers.xavier_normal(),
        act_fn=lambda x: x,
        name="Prior",
        norm_type="layer",
    )

    posterior_decoder_fn = head_fn(
        hidden_dims=[hidden_dim] * base_ff_layers,
        output_dim=stoch_discrete_dim * stoch_discrete_dim,
        initializer=initializers.xavier_normal(),
        act_fn=nn.silu,
        use_bias=True,
        norm_type="layer",
        name="Posterior"
    )

    reward_decoder_fn = head_fn(
        hidden_dims=[hidden_dim],
        output_dim=num_discrete_bins,
        initializer=initializers.variance_scaling(0.0, "fan_avg", "truncated_normal"),
        act_fn=nn.silu,
        use_bias=True,
        norm="layer",
        name="Reward",
    )

    terminal_decoder_fn = head_fn(
        hidden_dims=[hidden_dim],
        output_dim=1,
        initializer=initializers.xavier_normal(),
        act_fn=nn.silu,
        use_bias=True,
        norm="layer",
        name="Reward",
    )

    ind_action_decoder_fn = head_fn(
        hidden_dims=[hidden_dim],
        output_dim=action_space_dim,
        initializer=initializers.xavier_normal(),
        act_fn=nn.silu,
        use_bias=True,
        norm="layer",
        name="IndAction",
    )


    lstm_config = dict(
        hidden_dim=state_dim,
        reset_on_termination=reset_on_termination,
        kernel_initializer=initializers.orthogonal(jnp.sqrt(1)),
        recurrent_initializer=initializers.xavier_normal(),
    )

    seq_proc_fn, seq_init_fn = rec_seq_proc_fn(lstm_seed, lstm_config)

    _prior_init_fn, _prior_fn = prior_fn(
        prior_latent_repr_enc, seq_init_fn, seq_proc_fn, prior_decoder_fn
    )
    _posterior_fn = posterior_fn(posterior_latent_repr_enc, posterior_decoder_fn)

    return world_model_fn(
        world_model_config,
        rssm_fn(
            (_prior_init_fn, _prior_fn),
            _posterior_fn,
            state_dim,
            stoch_discrete_dim,
            num_categories,
            unimix_ratio,
        ),
        representation_fn,
        observation_decoder_fn,
        terminal_decoder_fn,
        reward_decoder_fn,
        False,
        ind_action_decoder_fn,
    )
