###################################################
#                   finished                      #
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
traj_num = 1
eps = 1e-7
dt = .01
step_num = 10000
T = step_num * dt
patience = 50                           # we admit 50 times blow up generations
warm_up = 500
writeInterval = 10
r.seed(0)
utils.assembly_NSmatrix(nx, ny, dt, dx, dy)
u_hist_ = np.zeros([traj_num, step_num//writeInterval, nx+2, ny+2])
v_hist_ = np.zeros([traj_num, step_num//writeInterval, nx+2, ny+1])
p_hist_ = np.zeros([traj_num, step_num//writeInterval, nx, ny])


j = 0
i = 0
while j < traj_num and i < patience: 
    i = i+1
    print('generating the {}-th trajectory...'.format(j))
    y0 = r.rand()*0.4 + 0.3
    u = np.zeros([nx+2, ny+2])
    v = np.zeros([nx+2, ny+1])
    p = np.zeros([nx, ny])      # staggered grid, the size of grid p is undetermined
    divu = np.zeros([nx, ny])   # source term in poisson equation: divergence of the predicted velocity field
    u[0, 1:-1] = np.exp(-50*(np.linspace(dy/2, 1-dy/2, ny) - y0)**2)
    u_hist = np.zeros([(step_num+warm_up)//writeInterval, nx+2, ny+2])
    v_hist = np.zeros([(step_num+warm_up)//writeInterval, nx+2, ny+1])
    p_hist = np.zeros([(step_num+warm_up)//writeInterval, nx, ny])


    flag = True
    for k in range(step_num+warm_up):
        t = k*dt
        u, v, p, flag = utils.projection_method(u, v, t, dx=dx, dy=dy, nx=nx, ny=ny, y0=y0, eps=eps, dt=dt, Re=Re, flag=flag)
        if flag == False:
            break
        if k%writeInterval == 0:
            u_hist[k//writeInterval, :, :] = copy.deepcopy(u)
            v_hist[k//writeInterval, :, :] = copy.deepcopy(v)
            p_hist[k//writeInterval, :, :] = copy.deepcopy(p) 


    if flag:
    # successful generating traj
        u_hist_[j] = copy.deepcopy(u_hist[warm_up//writeInterval:])
        v_hist_[j] = copy.deepcopy(v_hist[warm_up//writeInterval:])
        p_hist_[j] = copy.deepcopy(p_hist[warm_up//writeInterval:])
        j = j+1


if j == traj_num:
    label_dim = 1
    np.savez('../data/NS/{}-{}.npz'.format(nx, Re), 
             arg=[nx, ny, dt, T, label_dim], 
             u=u_hist_, 
             v=v_hist_, 
             label=p_hist_)