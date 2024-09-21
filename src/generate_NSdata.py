###################################################
#                   finished                      #
###################################################
import argparse
import copy

import hydra
import ml_collections
import numpy as np
import numpy.random as r

import utils


def generate_NS_data(config: ml_collections.ConfigDict):
  print('Generating NS data with n = {}, Re = {} ...'.format(nx, Re))

  nx = config.nx
  ny = config.ny
  dim = 2  # dimension of the problem
  dx = 1 / ny
  dy = 1 / ny
  traj_num = 10
  eps = 1e-7
  dt = .01
  step_num = 2000
  T = step_num * dt
  patience = 50  # we admit 50 times blow up generations
  warm_up = 500
  writeInterval = 2
  r.seed(0)
  utils.assembly_NSmatrix(nx, ny, dt, dx, dy)
  u_hist_ = np.zeros([traj_num, step_num // writeInterval, nx + 2, ny + 2])
  v_hist_ = np.zeros([traj_num, step_num // writeInterval, nx + 2, ny + 1])
  p_hist_ = np.zeros([traj_num, step_num // writeInterval, nx, ny])

  j = 0
  i = 0
  while j < traj_num and i < patience:
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
      u, v, p, flag = utils.projection_method(
        u,
        v,
        t,
        dx=dx,
        dy=dy,
        nx=nx,
        ny=ny,
        y0=y0,
        eps=eps,
        dt=dt,
        Re=Re,
        flag=flag
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

  if j == traj_num:
    label_dim = 1
    U = np.zeros([traj_num, step_num // writeInterval, 2, nx + 2, ny + 2])
    U[:, :, 0] = u_hist_
    U[:, :, 1, :, 1:] = v_hist_
    np.savez(
      '../data/NS/{}-{}.npz'.format(nx, Re),
      arg=[nx, ny, dt * writeInterval, T, label_dim],
      U=U,
      label=p_hist_
    )


@hydra.main(config_path="configs/", config_name="ns_incomp.yaml")
def main(config: ml_collections.ConfigDict):
  """
    This is a starter function of the simulation

    Args:
        config: This function uses hydra configuration for all parameters.
    """

  from src import sim_ns_incomp_2d
  sim_ns_incomp_2d.ns_sim(config=config, **config)


if __name__ == "__main__":
  main()
