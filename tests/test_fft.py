import jax
import jax.numpy as jnp
import pytest
import yaml
from box import Box

from ml4dynamics.utils import create_fine_coarse_simulator


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


def test_convection_ns():
  r"""Test the implementation of the convection term in the NS HIT model.

  .. math::

    \omega(x, y) = \cos(x + y)
    \psi(x, y) = \cos(x + y) / 2
    u(x, y) = \partial_y \psi(x, y) = -\sin(x + y) / 2
    v(x, y) = -\partial_x \psi(x, y) = \sin(x + y) / 2
    J(x, y) = u(x, y) \partial_x \omega(x, y) + v(x, y) \partial_y \omega(x, y) = 0

  """

  with open(f"config/ns_hit.yaml", "r") as file:
    config_dict = yaml.safe_load(file)
  config = Box(config_dict)
  model, _ = create_fine_coarse_simulator(config)
  n = model.N

  w_hat_true = jnp.zeros((n, n // 2 + 1), dtype=jnp.complex128)
  w_hat_true = w_hat_true.at[1, 1].set(n**2 / 2)
  psi_hat2_true = jnp.zeros((n * 2, n + 1), dtype=jnp.complex128)
  psi_hat2_true = psi_hat2_true.at[1, 1].set(n**2)
  convect_true = jnp.zeros_like(w_hat_true)

  x = jnp.linspace(0, 2 * jnp.pi, n, endpoint=False)
  y = jnp.linspace(0, 2 * jnp.pi, n, endpoint=False)
  x, y = jnp.meshgrid(x, y)
  # w = jnp.cos(x - y)
  w = jnp.cos(x) * jnp.cos(y)
  w_hat = jnp.fft.rfft2(w)
  w_hat2 = jnp.zeros((n * 2, n + 1), dtype=jnp.complex128)
  psi_hat2 = jnp.zeros((n * 2, n + 1), dtype=jnp.complex128)
  w_hat2 = w_hat2.at[:n // 2, :n // 2 + 1].set(w_hat[:n // 2] * 4)
  w_hat2 = w_hat2.at[-n // 2:, :n // 2 + 1].set(w_hat[n // 2:] * 4)
  psi_hat2 = psi_hat2.at[:n // 2, :n // 2 + 1].set(
    -(w_hat / model.laplacian_)[:n // 2] * 4
  )
  psi_hat2 = psi_hat2.at[-n // 2:, :n // 2 + 1].set(
    -(w_hat / model.laplacian_)[n // 2:] * 4
  )
  wx2 = jnp.fft.irfft2(1j * w_hat2 * model.k2x)
  wy2 = jnp.fft.irfft2(1j * w_hat2 * model.k2y)
  psix2 = jnp.fft.irfft2(1j * psi_hat2 * model.k2x)
  psiy2 = jnp.fft.irfft2(1j * psi_hat2 * model.k2y)
  convect = jnp.zeros_like(w_hat)
  convect_ = jnp.fft.rfft2(wx2 * psiy2 - wy2 * psix2)
  convect = convect.at[:n // 2].set(convect_[:n // 2, :n // 2 + 1] / 4)
  convect = convect.at[n // 2:].set(convect_[-n // 2:, :n // 2 + 1] / 4)

  # assert jnp.allclose(w_hat, w_hat_true)
  # assert jnp.allclose(psi_hat2, psi_hat2_true)
  assert jnp.allclose(convect, convect_true)
