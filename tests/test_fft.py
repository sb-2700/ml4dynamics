import jax
import jax.numpy as jnp
import pytest
from jax import random


jax.config.update('jax_enable_x64', True)

# @pytest.mark.parametrize(
#     ("hw", "param_count"),
#     [
#         ((128, 128), 34_491_599),
#         # It's fully convolutional => same parameter number.
#         ((256, 256), 34_491_599),
#     ],
#   )


# @pytest.mark.skip
def test_embed_rfft_mat():

  def extend(f_hat):
    """specific to 2D"""
    n = f_hat.shape[0]
    f2_hat = jnp.zeros((n * 2, n + 1), dtype=f_hat.dtype)
    f2_hat = f2_hat.at[:n // 2, :n // 2 + 1].set(f_hat[:n // 2] * 4)
    f2_hat = f2_hat.at[-n // 2:, :n // 2 + 1].set(f_hat[n // 2:] * 4)
    return f2_hat
  
  def func(x, y):
    return jnp.sin(4 * x) * jnp.sin(3 * y) + jnp.cos(x) + jnp.sin(y) + jnp.cos(2 * x) * jnp.sin(2 * y) +\
      jnp.cos(3 * x) * jnp.sin(4 * y) + jnp.cos(2 * x) * jnp.cos(3 * y) + jnp.sin(2 * x) * jnp.cos(3 * y)
  
  n = 16
  x1 = jnp.linspace(0, 2 * jnp.pi, n, endpoint=False)
  x1, y1 = jnp.meshgrid(x1, x1)
  f1 = func(x1, y1)
  f1_hat = jnp.fft.rfftn(f1)
  x2 = jnp.linspace(0, 2 * jnp.pi, n * 2, endpoint=False)
  x2, y2 = jnp.meshgrid(x2, x2)
  f2 = func(x2, y2)
  f2_hat = jnp.fft.rfftn(f2)
  f2_hat_ = extend(f1_hat)
  assert jnp.allclose(f2_hat.real, f2_hat_.real)

  shift_f1_hat = jnp.fft.fftshift(f1_hat.real, tuple(range(f1_hat.ndim-1)))
  shift_f2_hat = jnp.zeros_like(f2_hat_)
  shift_f2_hat = shift_f2_hat.at[n // 2:-(n // 2), :n // 2 + 1].set(shift_f1_hat * 4)
  assert jnp.allclose(shift_f2_hat.real, jnp.fft.fftshift(f2_hat, tuple(range(f1_hat.ndim-1))).real)
