from typing import Tuple

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp

"""
Adopted (almost 100% copied) from https://github.com/symoon11/dreamerv3-flax
"""

class Discrete(distrax.Categorical):
    """Discrete distribution."""

    def __init__(
        self,
        logits: jax.Array,
        low: float = -20.0,
        high: float = 20.0,
        use_symlog: bool = True,
    ):
        """Initializes a distribution."""
        super().__init__(logits)

        # Bins
        self.bins = jnp.linspace(low, high, num=logits.shape[-1], dtype=jnp.float32)

        def symlog(x: jax.Array) -> jax.Array:
            """Defines the symlog function."""
            return jnp.sign(x) * jnp.log(1 + jnp.abs(x))

        def symexp(x: jax.Array) -> jax.Array:
            """Defines the symexp function."""
            return jnp.sign(x) * (jnp.exp(jnp.abs(x)) - 1)

        if not use_symlog:
            self.trans = lambda x: x
            self.trans_inv = lambda x: x
        else:
            self.trans = symlog
            self.trans_inv = symexp

    def mean(self) -> jax.Array:
        """Calculates the mean."""
        # Calculate the mean.
        mean = jnp.sum(self.probs * self.bins, axis=-1)

        # Apply the inverse transform.
        mean = self.trans_inv(mean)

        return mean

    def log_prob(self, value: jax.Array) -> jax.Array:
        """Calculates the log probability."""
        # Apply the transform.
        value = self.trans(value)

        # Calculate the largest bin index below the value.
        below = self.bins <= value[..., None]
        below = jnp.sum(jnp.astype(below, jnp.int32), axis=-1) - 1
        below = jnp.clip(below, 0, len(self.bins) - 1)

        # Calculate the smallest bin index above the value.
        above = self.bins > value[..., None]
        above = len(self.bins) - jnp.sum(jnp.astype(above, jnp.int32), axis=-1)
        above = jnp.clip(above, 0, len(self.bins) - 1)

        # Calculate the distance between the value and each of the bins.
        equal = below == above
        dist_to_below = jnp.where(equal, 1, jnp.abs(self.bins[below] - value))
        dist_to_above = jnp.where(equal, 1, jnp.abs(self.bins[above] - value))

        # Calculate the weight for each of the bins.
        total = dist_to_below + dist_to_above
        weight_below = dist_to_above / total
        weight_above = dist_to_below / total

        # Calculate the target.
        target_below = nn.one_hot(below, len(self.bins)) * weight_below[..., None]
        target_above = nn.one_hot(above, len(self.bins)) * weight_above[..., None]
        target = target_below + target_above

        # Calculate the log probability.
        log_prob = jnp.sum(target * self.logits, axis=-1)

        return log_prob


class OneHotCategorical(distrax.OneHotCategorical):
    """One-hot categorical distribution."""

    def __init__(self, logits: jax.Array, unimix_ratio: float):
        """Initializes a distribution."""
        if unimix_ratio:
            # Calculate the probability.
            probs = nn.softmax(logits, axis=-1)

            # Define the uniform distribution.
            uniform = jnp.ones_like(probs) / probs.shape[-1]

            # Mix the probability with the uniform distribution.
            probs = (1.0 - unimix_ratio) * probs + unimix_ratio * uniform

            # Calculate the logit.
            logits = jnp.log(probs)

        super().__init__(logits)

    def _sample_n(self, key: jax.Array, n: int) -> jax.Array:
        """Returns samples."""
        # Get samples.
        sample = super()._sample_n(key, n)

        # Calculate the straight-through estimator.
        sample += self.probs - jax.lax.stop_gradient(self.probs)

        return sample


class MSE(distrax.Distribution):
    """MSE distribution."""

    def __init__(self, loc: jax.Array, use_symlog: bool = True):
        super().__init__()
        self._loc = loc

        def symlog(x: jax.Array) -> jax.Array:
            """Defines the symlog function."""
            return jnp.sign(x) * jnp.log(1 + jnp.abs(x))

        def symexp(x: jax.Array) -> jax.Array:
            """Defines the symexp function."""
            return jnp.sign(x) * (jnp.exp(jnp.abs(x)) - 1)

        # Location

        if not use_symlog:
            self.trans = lambda x: x
            self.trans_inv = lambda x: x
        else:
            self.trans = symlog
            self.trans_inv = symexp

    @property
    def event_shape(self) -> Tuple[int, ...]:
        return ()

    @property
    def loc(self) -> jax.Array:
        return self._loc

    def _sample_n(self, key: jax.Array, n: int) -> jax.Array:
        """Returns samples."""
        # Get samples.
        sample = jnp.repeat(self._loc[jnp.newaxis], n, axis=0)

        # Apply the inverse transform.
        sample = self.trans_inv(sample)

        return sample

    def mode(self) -> jax.Array:
        """Returns the mode"""
        # Apply the inverse transform.
        mode = self.trans_inv(self._loc)

        return mode

    def log_prob(self, value: jax.Array) -> jax.Array:
        """Calculates the log probability."""
        # Calculate the negative MSE.
        return -jnp.square(self._loc - self.trans(value))


class Normalizer(nn.Module):
    """Normalizer module."""

    decay: float = 0.99
    max_scale: float = 1.0
    q_low: float = 5.0
    q_high: float = 95.0

    def setup(self):
        # Statistics
        self.low = self.variable("aux", "low", jnp.zeros, (), jnp.float32)
        self.high = self.variable("aux", "high", jnp.zeros, (), jnp.float32)

    def __call__(self, x: jax.Array) -> Tuple[jax.Array, jax.Array]:
        # Update the statistics.
        self.update_stat(x)

        # Get the statistics.
        offset, inv_scale = self.get_stat()

        return offset, inv_scale

    def update_stat(self, x: jax.Array) -> None:
        # Get the percentiles.
        low = jnp.percentile(x, self.q_low)
        high = jnp.percentile(x, self.q_high)

        # Update the statistics.
        self.low.value = self.decay * self.low.value + (1 - self.decay) * low
        self.high.value = self.decay * self.high.value + (1 - self.decay) * high

    def get_stat(self) -> Tuple[jax.Array, jax.Array]:
        # Get the statistics.
        offset = self.low.value
        inv_scale = jnp.maximum(1.0 / self.max_scale, self.high.value - self.low.value)

        return offset, inv_scale
