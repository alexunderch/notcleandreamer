import functools
import math
from typing import Any, Callable, List, Tuple

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.linen import initializers
from jax.lax import Precision

"""
Developers note: these nn declarations were done in hurry, so better to be reworked.
`norm_type` could be only "layer" or None if what.
"""


class LinReprBlock(nn.Module):
    hidden_dim: int
    act_fn: Callable
    initializer: initializers.Initializer
    precision: Any
    norm_type: str = "layer"

    @nn.compact
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        x = nn.Dense(
            self.hidden_dim,
            kernel_init=self.initializer,
            use_bias=self.norm_type is None,
            precision=Precision(self.precision),
            dtype=jnp.float16,
        )(inputs)
        if self.norm_type is not None:
            x = nn.LayerNorm(dtype=jnp.float16)(x)
        if self.act_fn is not None:
            x = self.act_fn(x)
        return x


class ConvReprBlock(nn.Module):
    hidden_dim: int
    act_fn: Callable
    initializer: initializers.Initializer
    precision: Any
    norm_type: str = "layer"

    @nn.compact
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        x = nn.Conv(
            features=self.hidden_dim,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding="SAME",
            kernel_init=self.initializer,
            use_bias=self.norm_type is None,
            precision=None,
            dtype=jnp.float16,
        )(inputs)
        if self.norm_type is not None:
            x = nn.LayerNorm(dtype=jnp.float16)(x)
        if self.act_fn is not None:
            x = self.act_fn(x)
        return x


class ConvTransposeReprBlock(nn.Module):
    hidden_dim: int
    act_fn: Callable
    initializer: initializers.Initializer
    precision: Any
    norm_type: str = "layer"

    @nn.compact
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:

        x = nn.ConvTranspose(
            features=self.hidden_dim,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding="SAME",
            kernel_init=self.initializer,
            use_bias=self.norm_type is None,
            precision=None,
            dtype=jnp.float16,
        )(inputs)

        if self.norm_type is not None:
            x = nn.LayerNorm(dtype=jnp.float16)(x)
        if self.act_fn is not None:
            x = self.act_fn(x)
        return x


def linear_encoder_model(
    hidden_dims: jnp.ndarray,
    initializer: initializers.Initializer,
    activation_fn: Callable,
    precision: Any,
    name: str = None,
) -> Callable:

    if name is None:
        name = ""
    else:
        name += "_"

    def call() -> nn.Module:

        return nn.Sequential(
            [
                LinReprBlock(
                    int(hs),
                    act_fn=activation_fn,
                    initializer=initializer,
                    precision=precision,
                    name=f"ReprLinearEncoder_{name}{i}",
                )
                for i, (hs) in enumerate(hidden_dims)
            ]
        )

    return call


def linear_decoder_model(
    hidden_dims: jnp.ndarray,
    initializer: initializers.Initializer,
    activation_fn: Callable,
    precision: Any,
    name: str = None,
) -> Callable:

    if name is None:
        name = ""
    else:
        name += "_"

    def call() -> nn.Module:

        return nn.Sequential(
            [
                LinReprBlock(
                    int(hs),
                    act_fn=activation_fn,
                    initializer=initializer,
                    precision=precision,
                    name=f"ReprLinearEncoder_{name}{i}",
                )
                for i, (hs) in enumerate(hidden_dims)
            ]
        )

    return call


def conv_encoder_model(
    base_shape: Tuple,
    base_channels: int,
    min_res: int,
    initializer: initializers.Initializer,
    activation_fn: Callable,
    precision: Any,
) -> Callable:

    def call() -> nn.Module:
        num_layers = int(math.log2(base_shape[0] // min_res))
        hidden_dims = [2**i * base_channels for i in range(num_layers)]

        return nn.Sequential(
            [lambda x: x.astype(jnp.float32) - 0.5]
            + [
                ConvReprBlock(
                    int(hs),
                    act_fn=activation_fn,
                    initializer=initializer,
                    precision=precision,
                    name=f"ReprConvEncoder_{i}",
                )
                for i, hs in enumerate(hidden_dims)
            ]
        )

    return call


def conv_decoder_model(
    base_shape: Tuple,
    base_channels: int,
    min_res: int,
    initializer: initializers.Initializer,
    activation_fn: Callable,
    precision: Any,
) -> Callable:

    def call() -> nn.Module:

        num_layers = int(math.log2(base_shape[0] // min_res))
        in_chan = 2 ** (num_layers - 1) * base_channels
        in_shape = (min_res, min_res, in_chan)

        # Convolutional layers
        hidden_dims = [
            2 ** (i - 1) * base_channels for i in reversed(range(num_layers))
        ]
        hidden_dims[-1] = base_shape[-1]

        last_layer = lambda ind: ind == len(hidden_dims) - 1  # noqa: E731

        return nn.Sequential(
            [
                LinReprBlock(
                    math.prod(in_shape),
                    act_fn=None,
                    norm_type=None,
                    initializer=initializer,
                    precision=precision,
                    name="ReprReshape",
                )
            ]
            + [lambda x: jnp.reshape(x, (-1, *in_shape))]
            + [
                ConvTransposeReprBlock(
                    int(hs),
                    act_fn=None if last_layer(i) else activation_fn,
                    initializer=initializer,
                    precision=precision,
                    name=f"ReprConvDecoder_{i}",
                    norm_type=None if last_layer(i) else "layer",
                )
                for i, hs in enumerate(hidden_dims)
            ]
            + [lambda x: x + 0.5]
        )

    return call


def latent_repr_fn(
    hidden_dims: list,
    initializer: initializers.Initializer,
    activation_fn: Callable,
    precision: Precision,
    name: str = None,
) -> Callable:

    if name is None:
        name = ""
    else:
        name += "_"

    last_layer = lambda ind: ind == len(hidden_dims) - 1  # noqa: E731

    def call() -> nn.Module:
        return nn.Sequential(
            [lambda x: x.reshape(x.shape[0], -1)]
            + [
                LinReprBlock(
                    int(hs),
                    act_fn=None if last_layer(i) else activation_fn,
                    norm_type=None if last_layer(i) else "layer",
                    initializer=initializer,
                    precision=precision,
                    name=f"LatentEnc_{name}{i}",
                )
                for i, (hs) in enumerate(hidden_dims)
            ]
        )

    return call


def latent_repr_fn2(
    hidden_dims: list,
    initializer: initializers.Initializer,
    activation_fn: Callable,
    precision: Precision,
    name: str = None,
) -> Callable:

    if name is None:
        name = ""
    else:
        name += "_"

    last_layer = lambda ind: ind == len(hidden_dims) - 1  # noqa: E731

    def call() -> nn.Module:
        return nn.Sequential(
            [
                lambda x: x.reshape(
                    -1,
                )
            ]
            + [
                LinReprBlock(
                    int(hs),
                    act_fn=None if last_layer(i) else activation_fn,
                    norm_type=None if last_layer(i) else "layer",
                    initializer=initializer,
                    precision=precision,
                    name=f"ReprLinearEncoder_{name}{i}",
                )
                for i, (hs) in enumerate(hidden_dims)
            ]
        )

    return call


def head_fn(
    hidden_dims: list,
    output_dim: int,
    initializer: initializers.Initializer,
    activation_fn: Callable,
    precision: Precision,
    name: str = None,
    norm_type: str = None,
) -> Callable:

    if name is not None:
        module_name = name
    else:
        module_name = "values" if output_dim == 1 else "logits"

    def call() -> nn.Module:

        return nn.Sequential(
            [
                LinReprBlock(
                    int(hs),
                    act_fn=activation_fn,
                    initializer=initializer,
                    precision=precision,
                    name=f"Head_{name}{i}",
                    norm_type=norm_type,
                )
                for i, (hs) in enumerate(hidden_dims)
            ]
            + [
                nn.Dense(
                    output_dim,
                    precision=precision,
                    kernel_init=initializer,
                    name=module_name,
                    use_bias=False,
                    dtype=jnp.float16,
                ),
                # nn.LayerNorm(dtype=jnp.float16) if norm_type is not None else lambda x: x
            ]
        )

    return call
