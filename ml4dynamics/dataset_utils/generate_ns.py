import copy
from datetime import datetime

import h5py
import numpy as np
import numpy.random as r
import yaml
from box import Box

import ml4dynamics.utils as utils


def main():
  with open(f"config/ns.yaml", "r") as file:
    config_dict = yaml.safe_load(file)
  config = Box(config_dict)
  nx = config.sim.nx
  ny = config.sim.ny
  Re = config.sim.Re
  dx = 1 / ny
  dy = 1 / ny
  case_num = config.sim.case_num
  eps = 1e-7
  dt = .01
  step_num = 2000
  t = step_num * dt
  patience = 50  # we admit 50 times blow up generations
  warm_up = 500
  writeInterval = 2
  r.seed(0)
  print('Generating NS data with n = {}, Re = {} ...'.format(nx, Re))

  utils.assembly_NSmatrix(nx, ny, dt, dx, dy)
  u_hist_ = np.zeros([case_num, step_num // writeInterval, nx + 2, ny + 2])
  v_hist_ = np.zeros([case_num, step_num // writeInterval, nx + 2, ny + 1])
  p_hist_ = np.zeros([case_num, step_num // writeInterval, nx, ny])

  j = 0
  i = 0
  while j < case_num and i < patience:
    i = i + 1
    print('generating the {}-th trajectory...'.format(j))
    y0 = r.rand() * 0.4 + 0.3
    u = np.zeros([nx + 2, ny + 2])
    v = np.zeros([nx + 2, ny + 1])
    p = np.zeros([nx, ny])  # staggered grid, the size of grid p is undetermined
    divu = np.zeros(
      [nx, ny]
    )  # source term in poisson equation: divergence of the predicted velocity field
    u[0, 1:-1] = np.exp(-50 * (np.linspace(dy / 2, 1 - dy / 2, ny) - y0)**2)
    u_hist = np.zeros([(step_num + warm_up) // writeInterval, nx + 2, ny + 2])
    v_hist = np.zeros([(step_num + warm_up) // writeInterval, nx + 2, ny + 1])
    p_hist = np.zeros([(step_num + warm_up) // writeInterval, nx, ny])

    flag = True
    for k in range(step_num + warm_up):
      t = k * dt
      u, v, p, flag = utils.projection_correction(
        u, v, t, dx=dx, dy=dy, nx=nx, ny=ny, y0=y0, eps=eps, dt=dt,
        Re=Re, flag=flag
      )
      if flag == False:
        break
      if k % writeInterval == 0:
        u_hist[k // writeInterval, :, :] = copy.deepcopy(u)
        v_hist[k // writeInterval, :, :] = copy.deepcopy(v)
        p_hist[k // writeInterval, :, :] = copy.deepcopy(p)

    if flag:
      # successful generating traj
      u_hist_[j] = copy.deepcopy(u_hist[warm_up // writeInterval:])
      v_hist_[j] = copy.deepcopy(v_hist[warm_up // writeInterval:])
      p_hist_[j] = copy.deepcopy(p_hist[warm_up // writeInterval:])
      j = j + 1

  if j == case_num:
    # TODO: need to modified this part to store the whole simulation data on
    # the grid
    # old data size, save all the data on the grid points
    # U = np.zeros([case_num, step_num // writeInterval, 2, nx + 2, ny + 2])
    # U[:, :, 0] = u_hist_
    # U[:, :, 1, :, 1:] = v_hist_
    U = np.zeros([case_num, step_num // writeInterval, nx, ny, 2])
    U[..., 0] = u_hist_[:, :, 1:-1, 1:-1]
    U[..., 1] = v_hist_[:, :, 1:-1, 1:]
    data = {
      "metadata": {
        "type": "ns",
        "t0": 0.0,
        "t1": t,
        "dt": dt * writeInterval,
        "nx": nx,
        "ny": ny,
        "description": "Navier-Stokes PDE dataset",
        "author": "Jiaxi Zhao",
        "creation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
      },
      "data": {
        "input_fine":
        U.reshape(-1, nx, ny, 2),  # shape [case_num * step_num // writeInterval, nx+2, ny+2, 2]
        "output_fine":
        p_hist_.reshape(-1, nx, ny, 1),  # shape [case_num * step_num // writeInterval, nx, ny, 1]
        # "input_coarse":
        # u_coarse,  # shape [case_num * step_num // writeInterval, 2, nx//2, ny//2]
        # "output_coarse":
        # label_coarse,  # shape [case_num, step_num // writeInterval, 2, nx//2, ny//2]
      },
      "config":
      config_dict,
      "readme":
      "This dataset contains the results of a Reaction-Diffusion PDE solver. "
      "The 'input' field represents the velocity, and the 'output' "
      "field represents the nonlinear RHS."
    }

    with h5py.File(
      f"data/ns/Re{Re}_n{case_num}.h5", "w"
    ) as f:
      metadata_group = f.create_group("metadata")
      for key, value in data["metadata"].items():
        metadata_group.create_dataset(key, data=value)

      data_group = f.create_group("data")
      data_group.create_dataset("inputs", data=data["data"]["input_fine"])
      data_group.create_dataset("outputs", data=data["data"]["output_fine"])
      # data_group.create_dataset(
      #   "input_coarse", data=data["data"]["input_coarse"]
      # )
      # data_group.create_dataset(
      #   "output_coarse", data=data["data"]["output_coarse"]
      # )

      config_group = f.create_group("config")
      config_yaml = yaml.dump(data["config"])
      config_group.attrs["config"] = config_yaml

      f.attrs["readme"] = data["readme"]


if __name__ == "__main__":
  main()
