import chex
import jax
import jax.numpy as jnp
import numpy as np
from gymnax.environments.spaces import Space


class MultiDiscrete(Space):
    """Minimal jittable class for multi discrete gymnax spaces."""

    def __init__(self, low: np.ndarray, high: np.ndarray):
        self.low = low
        self.high = high
        self.dist = self.high - self.low
        assert low.shape == high.shape
        self.shape = low.shape
        self.dtype = jnp.int16

    def sample(self, rng: chex.PRNGKey) -> chex.Array:
        return (
            jax.random.uniform(rng, shape=self.shape, minval=0, maxval=1) * self.dist
            + self.low
        ).astype(self.dtype)

    def contains(self, x) -> jnp.ndarray:
        """Check whether specific object is within space."""
        # type_cond = isinstance(x, self.dtype)
        # shape_cond = (x.shape == self.shape)
        range_cond = jnp.logical_and(x >= 0, x < self.n)
        return range_cond
