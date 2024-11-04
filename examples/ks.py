"""
Examples with Kuramoto–Sivashinsky equation

$$
  u_t + (c + u)u_c + uu_x + u_{xx} + \nu u_{xxxx} = 0.
$$

We perform sensitivity analysis of several statistics of the KS equation
w.r.t. the parameter c. We reproduce the fig. 1 in reference [1]. This
statistics can also be used to evaluate the ROM we proposed. To reproduce the
figure, we use the following parameters in config file:

ks:
  nu: 1
  c: 0.8
  L: 128
  T: 200
  init_scale: 1.
  nx: 1024
  dt: .1
  BC: Dirichlet-Neumann

reference:
1. https://arxiv.org/pdf/1307.8197
"""

from functools import partial
from typing import Iterator, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
import yaml
from box import Box
from jax import random as random
from jaxtyping import Array
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.dynamics import KS
from src.types import Batch, OptState, PRNGKey
from src.utils import plot_with_horizontal_colorbar


with open("config/simulation.yaml", "r") as file:
  config_dict = yaml.safe_load(file)
config = Box(config_dict)
# model parameters
nu = config.ks.nu
c = config.ks.c
L = config.ks.L
T = config.ks.T
init_scale = config.ks.init_scale
# solver parameters
N = config.ks.nx
dt = config.ks.dt
n_sample = 5
key = random.PRNGKey(config.sim.seed)

# KS simulator with Dirichlet Neumann BC
ks1 = KS(
  N=N - 1,
  T=T,
  dt=dt,
  init_scale=init_scale,
  L=L,
  nu=nu,
  c=c,
  BC="Dirichlet-Neumann",
  key=key,
)
ks2 = KS(
  N=N // 2 - 1,
  T=T,
  dt=dt,
  init_scale=init_scale,
  L=L,
  nu=nu,
  c=c,
  BC="Dirichlet-Neumann",
  key=key,
)
ks3 = KS(
  N=N // 4 - 1,
  T=T,
  dt=dt,
  init_scale=init_scale,
  L=L,
  nu=nu,
  c=c,
  BC="Dirichlet-Neumann",
  key=key,
)
ks_models = [ks1, ks2, ks3]

c_array = jnp.linspace(0.0, 2.0, 20)
dx = L / N
x = jnp.linspace(dx, L - dx, N - 1)
ubar = jnp.zeros((3, c_array.shape[0]))
u2bar = jnp.zeros((3, c_array.shape[0]))
fig, axs = plt.subplots(1, 2, figsize=(10, 4))
color_array = ["r", "b", "g"]

for i in range(c_array.shape[0]):
  print(f"{c_array[i]:.2f}")
  for _ in range(n_sample):
    key, subkey = random.split(key)
    # r = random.uniform(subkey) * 20 + 44
    # u0 = jnp.exp(-(x - r)**2 / r**2 * 4)
    u0 = random.uniform(subkey) * jnp.sin(8 * jnp.pi * x / 128) +\
      random.uniform(key) * jnp.sin(16 * jnp.pi * x / 128)
    for ks in ks_models:
      ks.c = c_array[i]
      ks.assembly_matrix()

    ks_models[0].run_simulation(u0, ks_models[0].CN_FEM)
    ks_models[1].run_simulation(u0[1::2], ks_models[1].CN_FEM)
    ks_models[2].run_simulation(u0[3::4], ks_models[2].CN_FEM)

    for j in range(len(ks_models)):
      umean = jnp.mean(ks_models[j].x_hist[-500:])
      u2mean = jnp.mean(ks_models[j].x_hist[-500:]**2)
      axs[0].scatter(c_array[i], umean, c=color_array[j], s=2)
      axs[1].scatter(c_array[i], u2mean, c=color_array[j], s=2)
      ubar = ubar.at[j, i].add(umean)
      u2bar = u2bar.at[j, i].add(u2mean)

ubar /= n_sample
u2bar /= n_sample
axs[0].plot(c_array, ubar[0], label=r"$N = {}$".format(N), c="r")
axs[1].plot(c_array, u2bar[0], label=r"$N = {}$".format(N), c="r")
axs[0].plot(c_array, ubar[1], label=r"$N = {}$".format(N//2), c="b")
axs[1].plot(c_array, u2bar[1], label=r"$N = {}$".format(N//2), c="b")
axs[0].plot(c_array, ubar[2], label=r"$N = {}$".format(N//4), c="g")
axs[1].plot(c_array, u2bar[2], label=r"$N = {}$".format(N//4), c="g")
axs[0].set_xlabel(r"$c$")
axs[1].set_xlabel(r"$c$")
axs[0].set_ylabel(r"$\langle \overline{u} \rangle$")
axs[1].set_ylabel(r"$\langle \overline{u^2} \rangle$")
axs[0].legend()
axs[1].legend()
plt.savefig("results/fig/ks_c_stats.pdf")
