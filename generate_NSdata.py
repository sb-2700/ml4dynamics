###################################################
#                   finished                      #
# I am not sure if save data as 128x32 will       #
# influce the result?                             #
###################################################
import utils
import numpy as np
import numpy.random as r
import copy
import argparse


parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--Re', type=int, default=100)
parser.add_argument('--nx', type=int, default=128)
args = parser.parse_args()
Re = args.Re
nx = args.nx
print('Generating NS data with n = {}, Re = {} ...'.format(nx, Re))


dim = 2                     # dimension of the problem
ny = nx//4
dx = 1/ny
dy = 1/ny
traj_num = 10
eps = 1e-7
dt = .01
T = 10
step_num = int(T/dt)
r.seed(0)
utils.assembly_NSmatrix(nx, ny, dt, dx, dy)
u_hist_ = np.zeros([traj_num, step_num, nx, ny])
v_hist_ = np.zeros([traj_num, step_num, nx, ny])
p_hist_ = np.zeros([traj_num, step_num, nx, ny])


for j in range(traj_num):


    print('generating the {}-th trajectory...'.format(j))
    y0 = r.rand()*0.4 + 0.3
    u = np.zeros([nx+2, ny+2])
    v = np.zeros([nx+2, ny+1])
    p = np.zeros([nx, ny])      # staggered grid, the size of grid p is undetermined
    divu = np.zeros([nx, ny])   # source term in poisson equation: divergence of the predicted velocity field
    u[0, 1:-1] = np.exp(-50*(np.linspace(dy/2, 1-dy/2, ny) - y0)**2)
    u_hist = np.zeros([step_num, nx, ny])
    v_hist = np.zeros([step_num, nx, ny])
    p_hist = np.zeros([step_num, nx, ny])


    for i in range(step_num):
        t = i*dt
        u, v = utils.projection_method(u, v, t, dx=dx, dy=dy, nx=nx, ny=ny, y0=y0, eps=eps, dt=dt, Re=Re)
        u_hist[i, :, :] = u[1:-1, 1:-1]
        v_hist[i, :, :] = v[1:-1, :-1]
        p_hist[i, :, :] = p 
    u_hist_[j] = copy.deepcopy(u_hist)
    v_hist_[j] = copy.deepcopy(v_hist)
    p_hist_[j] = copy.deepcopy(p_hist)


u = copy.deepcopy(u_hist_)
v = copy.deepcopy(v_hist_)
p = copy.deepcopy(p_hist_)
np.savez('../data/NS/n{}Re{}.npz'.format(nx, Re), arg=[nx, ny, dt, T], u=u, v=v, label=p)