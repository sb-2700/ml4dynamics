import jax
import jax.numpy as jnp
import numpy as np
import yaml
from box import Box
from jax import random
from matplotlib import pyplot as plt

from ml4dynamics.utils import plot_with_horizontal_colorbar
from ml4dynamics.dynamics import ns_hit
from pyfoam.utils import calc_utils

with open(f"config/ns_hit.yaml", "r") as file:
  config_dict = yaml.safe_load(file)
config = Box(config_dict)
Re = config.sim.Re
n = config.sim.n
T = config.sim.T
dt = config.sim.dt
L = config.sim.L
model = ns_hit(L=L * jnp.pi, N=n, nu=1/Re, T=T, dt=dt, init_scale=n**(1.5))
case_num = config.sim.case_num
writeInterval = 1

model.w_hat = jnp.zeros((model.N, model.N//2+1))
# f0 = int(jnp.sqrt(n/2)) # init frequency
f0 = 32
model.w_hat = model.w_hat.at[:f0, :f0].set(
  random.normal(random.PRNGKey(0), (f0, f0)) * model.init_scale
)
model.set_x_hist(model.w_hat, model.CN)

n_plot = 3
plot_interval = model.step_num // n_plot**2
im_array = np.zeros((n_plot, n_plot, n, n))
title_array = []
for i in range(n_plot**2):
  im_array[i//3, i%3] = model.x_hist[i * plot_interval]
  title_array.append(f"t={i * plot_interval * dt:.2f}")

plot_with_horizontal_colorbar(
  im_array, (12, 12), title_array, "ns_hit.png"
)

tau = calc_utils.calc_reynolds_stress(model.u_hist[..., None, :], 900)
im_array = np.zeros((2, 2, n, n))
title_array = [r"$\tau_{xx}$", r"$\tau_{xy}$", r"$\tau_{yx}$", r"$\tau_{yy}$"]
for i in range(2):
  for j in range(2):
    im_array[i, j] = tau[:, :, 0, i, j]
plot_with_horizontal_colorbar(
  im_array, (12, 12), title_array, "ns_hit_reynolds.png"
)

y_grid = jnp.linspace(0, L * jnp.pi, n, endpoint=False)

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

def power_spec_2d_over_t(U: jnp.array, dx: float, dy: float):
  
  def power_spec_2d(u_t):
    u_hat_x = jnp.fft.rfftn(u_t[..., 0])
    u_hat_y = jnp.fft.rfftn(u_t[..., 1])
    energy_spectral = 0.5 * (jnp.abs(u_hat_x)**2 + jnp.abs(u_hat_y)**2)
    hist = jnp.histogram(k_mag, bins=k_bins, weights=energy_spectral)[0]
    k_centers = 0.5*(k_bins[1:] + k_bins[:-1])
    area = 2 * jnp.pi * k_centers * bin_width
    E_k = hist / (area + 1e-12)
    return E_k
  
  nx, ny = U.shape[1:3]
  kx = jnp.fft.fftfreq(nx, d=dx) * 2 * jnp.pi
  ky = jnp.fft.rfftfreq(ny, d=dx) * 2 * jnp.pi
  kx_grid, ky_grid = jnp.meshgrid(kx, ky, indexing='ij')
  k_mag = jnp.sqrt(kx_grid**2 + ky_grid**2)
  k_max = jnp.max(k_mag)
  k_bins = jnp.linspace(0, k_max, num=int(nx//2))
  bin_width = k_bins[1] - k_bins[0]

  u = U - jnp.mean(U, axis=(0, ), keepdims=True)
  E_k_all = jax.vmap(power_spec_2d)(u)
  E_k_avg = jnp.mean(E_k_all, axis=0)
  return k_bins[:-1], E_k_avg

k_bins, E_k_avg = power_spec_2d_over_t(
  model.u_hist, model.dx, model.dx
)
plt.plot(k_bins, E_k_avg)
plt.xlabel("k")
plt.ylabel("E(k)")
plt.xscale("log")
plt.yscale("log")
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.title("2D Power Spectrum")
plt.savefig("power_spec_2d.png")
plt.clf()


breakpoint()


calc_utils.power_spec(
  model.u_hist[..., None, :], y_grid, model.dx, 1, (1, ), "psd.png"
)


breakpoint()
