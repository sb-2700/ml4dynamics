from functools import partial

import jax
import jax.numpy as jnp
import numpy as np


def corr(u):
  """Calculate the spatial correlation of the field.

  Args:
  U (Float[Array, 'batch_size nx ny']): field variable, the batch dimension can
  contains both time snapshots or other dimensions.
  """
  corr = np.zeros(*u.shape[1:])
  for i in range(corr.shape[0]):
    for j in range(corr.shape[1]):
      corr[i, j] = jnp.mean(jnp.roll(u, [i, j], axis=[1, 2]) * u)
  return corr


@partial(jax.jit, static_argnums=(1, ))
def calc_reynolds_stress(U: jnp.array, nt: int = 10):
  """Reynolds stress

  Args:
  U (Float[Array, 'nt nx ny nz DIM']): velocity, nz = 1 for 2D case.
  nt (int): number of time steps for averaging.

  Returns:
  tau (Float[Array, 'nx ny nz DIM**2']): Reynolds stresses.
  """

  u = U[-nt:] - jnp.mean(U[-nt:], axis=(0, ))[None]
  tau = jnp.mean(u[..., None] * u[..., None, :], axis=(0, ))
  return tau


def power_spec_over_t(U: jnp.array, dx: list):

  @jax.jit
  def power_spec(u):
    energy_spectral = 0
    for component in range(u.shape[-1]):
      energy_spectral += 0.5 * (jnp.abs(jnp.fft.fftn(u[..., component]))**2)
    hist = jnp.histogram(k_mag, bins=k_bins, weights=energy_spectral)[0]
    k_centers = 0.5 * (k_bins[1:] + k_bins[:-1])
    area = 2 * jnp.pi * k_centers * bin_width
    area = 1
    E_k = hist / (area + 1e-12)
    return E_k

  k = []
  for component in range(U.shape[-1]):
    n = U.shape[1 + component]
    k.append(jnp.fft.fftfreq(n, d=dx[component]) * 2 * jnp.pi)
  k_mesh = jnp.meshgrid(*k, indexing='ij')
  k2 = 0
  for k in k_mesh:
    k2 += k**2
  k_mag = jnp.sqrt(k2)
  k_max = jnp.max(k_mag)
  k_bins = jnp.linspace(0, k_max, num=int(n // 2))
  bin_width = k_bins[1] - k_bins[0]

  interval = 1000
  u = U[-interval:] - jnp.mean(U[-interval:], axis=(0, ), keepdims=True)
  E_k_all = jax.vmap(power_spec)(u)
  E_k_avg = jnp.mean(E_k_all, axis=0)
  return k_bins[:-1], E_k_avg
