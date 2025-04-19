import numpy as np
import yaml
from box import Box
from matplotlib import cm
from matplotlib import pyplot as plt

from ml4dynamics.dynamics import ns_channel

with open("config/ns_channel.yaml", "r") as file:
  config_dict = yaml.safe_load(file)
config = Box(config_dict)
# model parameters
warm_up = config.sim.warm_up
Lx = Ly = config.ns.Lx
nx = config.ns.nx
ny = config.ns.ny
dx = dy = Lx / nx
T = config.ns.T
Re = config.ns.Re
dt = config.ns.dt
step_num = int(T / dt)

# NS channel simulator with no-slip BC
ns_fine = ns_channel(
  Lx=Lx,
  nx=nx,
  ny=ny,
  T=T,
  dt=dt,
  Re=Re,
)
y0 = 0.325
u = np.zeros([nx + 2, ny + 2])
v = np.zeros([nx + 2, ny + 1])
u[0, 1:-1] = np.exp(-50 * (np.linspace(dy / 2, 1 - dy / 2, ny) - y0)**2)
u0 = np.hstack([u.reshape(-1), v.reshape(-1)])

ns_fine.assembly_matrix()
ns_fine.run_simulation(u0, ns_fine.projection_correction)
n_plot = 3
fig, axs = plt.subplots(n_plot, n_plot)
axs = axs.flatten()
for i in range(n_plot**2):
  axs[i].imshow(
    ns_fine.x_hist[i * 100, :nx * ny].reshape(nx, ny), cmap=cm.twilight
  )
  axs[i].axis("off")
plt.savefig("ns_channel.png")

breakpoint()
