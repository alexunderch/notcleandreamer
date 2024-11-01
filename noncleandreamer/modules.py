import math
from typing import Callable, Tuple, Literal

import flax.linen as nn
import jax.numpy as jnp
from flax.linen import initializers
from noncleandreamer.custom_types import base_jnp_type
import chex

"""
Developers note: these nn declarations were done in hurry, so better to be reworked.
`norm_type` could be only "layer" or None if what.
"""

STR2NORM = {
    "layer": nn.LayerNorm(dtype=base_jnp_type),
    "instance": nn.InstanceNorm(dtype=base_jnp_type),
    "none": lambda x: x
}

class LinearLayer(nn.Module):
    hidden_dim: chex.Numeric
    act_fn: Callable
    initializer: initializers.Initializer
    norm: Literal["layer", "instance", "none"]
    use_bias: bool

    @nn.compact
    def __call__(self, inputs: chex.ArrayDevice) -> chex.ArrayDevice:
        x = nn.Dense(
            self.hidden_dim, 
            use_bias=self.use_bias,
            dtype=base_jnp_type,
            kernel_init=self.initializer
        )(inputs)
        return STR2NORM[self.norm](self.act_fn(x))
    
class ConvLayer(nn.Module):
    hidden_dim: chex.Numeric
    kernel_size: chex.Numeric
    stride: chex.Numeric
    act_fn: Callable
    initializer: initializers.Initializer
    norm: Literal["layer", "instance", "none"]
    use_bias: bool

    @nn.compact
    def __call__(self, inputs: chex.ArrayDevice) -> chex.ArrayDevice:
        x = nn.Conv(
            features=self.hidden_dim,
            kernel_size=(self.kernel_size, self.kernel_size),
            strides=(self.stride, self.stride),
            padding="SAME",
            kernel_init=self.initializer,
            use_bias=self.use_bias,
            dtype=base_jnp_type,
        )(inputs)
        return STR2NORM[self.norm](self.act_fn(x))

class ConvTransposeLayer(nn.Module):
    hidden_dim: chex.Numeric
    kernel_size: chex.Numeric
    stride: chex.Numeric
    act_fn: Callable
    initializer: initializers.Initializer
    norm: Literal["layer", "instance", "none"]
    use_bias: bool

    @nn.compact
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:

        x = nn.ConvTranspose(
            features=self.hidden_dim,
            kernel_size=(self.kernel_size, self.kernel_size),
            strides=(self.stride, self.stride),
            padding="SAME",
            kernel_init=self.initializer,
            use_bias=self.use_bias,
            dtype=base_jnp_type,
        )(inputs)
        return STR2NORM[self.norm](self.act_fn(x))

def linear_encoder_model(
    hidden_dims: chex.Numeric,
    act_fn: Callable,
    initializer: initializers.Initializer,
    norm: Literal["layer", "instance", "none"],
    use_bias: bool,
    name: str = None,
) -> Callable:
    
    name = "" if name is None else f"_{name}"
    
    def _call() -> nn.Module:

        layers = [
            LinearLayer(
                hidden_dim, act_fn, initializer, norm, use_bias, name=f"LinEncoderLayer{name}{i}")
            for i, hidden_dim in enumerate(hidden_dims)
        ]
        return nn.Sequential(layers)

    return _call

def linear_decoder_model(
    hidden_dims: chex.Numeric,
    act_fn: Callable,
    initializer: initializers.Initializer,
    norm: Literal["layer", "instance", "none"],
    use_bias: bool,
    name: str = None,
) -> Callable:

    name = "" if name is None else f"_{name}"

    def _call() -> nn.Module:

        layers = [
            LinearLayer(
                hidden_dim, act_fn, initializer, norm, use_bias, name=f"LinDecoderLayer{name}{i}")
            for i, hidden_dim in enumerate(hidden_dims)
        ]
        return nn.Sequential(layers)

    return _call


def conv_encoder_model(
    base_shape: Tuple,
    base_channels: int,
    min_res: int,
    kernel_size: chex.Numeric,
    stride: chex.Numeric,
    act_fn: Callable,
    initializer: initializers.Initializer,
    norm: Literal["layer", "instance", "none"],
    use_bias: bool,
    name: str = None
) -> Callable:

    name = "" if name is None else f"_{name}"

    def _call() -> nn.Module:
        num_layers = int(math.log2(base_shape[0] // min_res))
        hidden_dims = [2**i * base_channels for i in range(num_layers)]

        layers = [
                ConvLayer(hidden_dim, kernel_size, stride, act_fn, initializer, norm, use_bias, name=f"ConvEncoder{name}{i}")
                for i, hidden_dim in enumerate(hidden_dims)
            ]
        return nn.Sequential(layers)

    return _call


def conv_decoder_model(
    base_shape: Tuple,
    base_channels: int,
    min_res: int,
    kernel_size: chex.Numeric,
    stride: chex.Numeric,
    act_fn: Callable,
    lin_initializer: initializers.Initializer,
    initializer: initializers.Initializer,
    norm: Literal["layer", "instance", "none"],
    use_bias: bool,
    name: str = None
) -> Callable:
    
    name = "" if name is None else f"_{name}"

    def _call() -> nn.Module:

        num_layers = int(math.log2(base_shape[0] // min_res))
        in_chan = 2 ** (num_layers - 1) * base_channels
        in_shape = (min_res, min_res, in_chan)

        # Convolutional layers
        hidden_dims = [
            2 ** (i - 1) * base_channels for i in reversed(range(num_layers))
        ]
        hidden_dims[-1] = base_shape[-1]

        layers = [
            LinearLayer(math.prod(in_shape), lambda x: x, lin_initializer, "none", False, f"PreProjection{name}"),
            lambda x: jnp.reshape(-1, *in_shape)
        ]

        for i, hidden_dim in enumerate(hidden_dims):
            lnorm = norm if i != len(hidden_dims)-1 else "none"
            layers += [
                ConvTransposeLayer(
                    hidden_dim, kernel_size, stride,
                    act_fn, initializer, lnorm, use_bias,
                    name=f"ConvDecoder{name}{i}"
                )
            ]
        
        return nn.Sequential(layers)
    
    return _call

def latent_repr_fn2(
    hidden_dims: chex.Numeric,
    act_fn: Callable,
    initializer: initializers.Initializer,
    norm: Literal["layer", "instance", "none"],
    use_bias: bool,
) -> Callable:

    def call() -> nn.Module:
        return nn.Sequential([
                lambda x: x.reshape(-1),
                linear_encoder_model(hidden_dims, act_fn, initializer, norm, use_bias, "ReprFn")
            ]
        )

    return call


def head_fn(
    output_dim: chex.Numeric,
    hidden_dims: chex.Numeric,
    act_fn: Callable,
    initializer: initializers.Initializer,
    norm: Literal["layer", "instance", "none"],
    use_bias: bool,
    name: str = None
) -> Callable:

    if name is not None:
        module_name = name
    else:
        module_name = "Values" if output_dim == 1 else "Logits"

    def _call() -> nn.Module:

        return nn.Sequential(
            [
               linear_decoder_model(hidden_dims, act_fn, initializer, norm, use_bias, f"HeadFn{module_name}"),
                nn.Dense(
                    output_dim,
                    kernel_init=initializer,
                    name=module_name,
                    use_bias=False,
                    dtype=base_jnp_type,
                ),
            ]
        )

    return _call
