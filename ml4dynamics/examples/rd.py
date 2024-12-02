import jax.numpy as jnp
import torch
import yaml
from box import Box
from jax import random as random
from matplotlib import cm
from matplotlib import pyplot as plt

from ml4dynamics.dynamics import RD


with open("config/simulation.yaml", "r") as file:
  config_dict = yaml.safe_load(file)
config = Box(config_dict)
# model parameters
warm_up = config.sim.warm_up
Lx = Ly = config.react_diff.Lx
nx = config.react_diff.nx
ny = config.react_diff.nx
dx = dy = Lx / nx
T = config.react_diff.T
dt = config.react_diff.dt
step_num = int(T / dt)
alpha = config.react_diff.alpha
beta = config.react_diff.beta
gamma = config.react_diff.gamma
d = config.react_diff.d

# KS simulator with Dirichlet Neumann BC
rd_fine = RD(
  L=Lx,
  N=nx**2 * 2,
  T=T,
  dt=dt,
  alpha=alpha,
  beta=beta,
  gamma=gamma,
  d=d,
  tol=1e-8,
  init_scale=4,
  tv_scale=1e-8,
)

u_fft = jnp.zeros((2, nx, nx))
u_fft = u_fft.at[:, :10, :10].set(
  random.normal(key=random.PRNGKey(0), shape=(2, 10, 10))
)
u0 = jnp.real(jnp.fft.fftn(u_fft, axes=(1, 2)).reshape(-1)) / nx
# u0 = jnp.zeros(nx**2 * 2)
# u0 = random.normal(key=random.PRNGKey(0), shape=(nx, nx, 2)).reshape(-1)

rd_fine.assembly_matrix()
rd_fine.run_simulation(u0, rd_fine.adi)
n_plot = 3
fig, axs = plt.subplots(n_plot, n_plot)
axs = axs.flatten()
for i in range(n_plot**2):
  axs[i].imshow(rd_fine.x_hist[i * 500, :nx**2].reshape(nx, nx), cmap=cm.jet)
  axs[i].axis("off")
plt.savefig("test.pdf")

breakpoint()
