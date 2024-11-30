import jax.numpy as jnp
import torch
import yaml
from box import Box
from jax import random as random
from matplotlib import pyplot as plt

from ml4dynamics.dynamics import RD


with open("config/simulation.yaml", "r") as file:
  config_dict = yaml.safe_load(file)
config = Box(config_dict)
# model parameters
warm_up = config.sim.warm_up
widthx = (config.react_diff.x_right - config.react_diff.x_left)
widthy = (config.react_diff.y_top - config.react_diff.y_bottom)
nx = config.react_diff.nx
ny = config.react_diff.nx
dx = widthx / nx
dy = widthy / ny
gamma = config.react_diff.gamma
t = config.react_diff.t
alpha = config.react_diff.alpha
beta = config.react_diff.beta
# solver parameters
step_num = config.react_diff.nt
dt = t / step_num

# KS simulator with Dirichlet Neumann BC
rd_fine = RD(
  N=nx,
  T=t,
  dt=dt,
  dx=dx,
  tol=1e-8,
  init_scale=4,
  tv_scale=1e-8,
  L=widthx,
  alpha=alpha,
  beta=beta,
  gamma=gamma,
  device=torch.device('cpu'),
)



