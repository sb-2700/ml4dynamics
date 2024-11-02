"""
Examples with Kuramoto–Sivashinsky equation

$$
  u_t + (c + u)u_c + uu_x + u_{xx} + \nu u_{xxxx} = 0.
$$

We perform sensitivity analysis of several statistics of the KS equation
w.r.t. the parameter c.



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
N1 = config.ks.nx
N2 = N1 // 2
dt = config.ks.dt
r = N1 // N2
key = random.PRNGKey(config.sim.seed)

# fine simulation with Dirichlet Neumann BC
ks_fine = KS(
  N=N1,
  T=T,
  dt=dt,
  tol=1e-8,
  init_scale=init_scale,
  tv_scale=1e-8,
  L=L,
  nu=1.0,
  c=0.8,
  BC="Dirichlet-Neumann",
  key=key,
)

dx = L / (N1 + 1)
x = jnp.linspace(dx, L - dx, N1)
u0 = jnp.exp(-(x - L/2)**2 / (L / 2)**2 * 10)
ks_fine.run_simulation(u0, ks_fine.CN_FEM)
plt.imshow(ks_fine.x_hist)
plt.savefig("ks.pdf")