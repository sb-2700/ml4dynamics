###################################################
#                   finished                      #
###################################################
import utils
import numpy as np
import numpy.random as r
import copy
import argparse
import pdb

# parse the simulation arguments
parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--beta', type=float, default=0.2)
args = parser.parse_args()
beta = args.beta
print('Generating RD data with beta = {:.1f}...'.format(beta))

# set simulation parameters
widthx = 1.0
widthy = 1.0
dt = 0.001
step_num = 1000
T = step_num * dt
alpha = 0.01
gamma = 0.05
warm_up = 200
writeInterval = 1
tol = 1e-7
r.seed(0)

# simulating training trajectories
case_num = 10
traning_u64 = np.zeros([case_num, step_num, 64, 64])
traning_v64 = np.zeros([case_num, step_num, 64, 64])
traning_labelu64 = np.zeros([case_num, step_num, 64, 64])
traning_labelv64 = np.zeros([case_num, step_num, 64, 64])
traning_u128 = np.zeros([case_num, step_num, 128, 128])
traning_v128 = np.zeros([case_num, step_num, 128, 128])
traning_labelu128 = np.zeros([case_num, step_num, 128, 128])
traning_labelv128 = np.zeros([case_num, step_num, 128, 128])
for i in range(case_num):
    print('generating the {}-th trajectory...'.format(i))
    # simulation in 128x128 grid
    n = 128
    dx = widthx/n
    u_hist = np.zeros([step_num+warm_up, n, n])
    v_hist = np.zeros([step_num+warm_up, n, n])
    utils.assembly_RDmatrix(n, dt, dx, beta, gamma)
    u_init = r.randn(n, n)
    v_init = r.randn(n, n)


    u = copy.deepcopy(u_init)
    v = copy.deepcopy(v_init)
    u_hist, v_hist = utils.RD_adi(u, v, alpha=alpha, beta=beta, gamma=gamma, step_num=step_num+warm_up, plot=False)
    traning_u128[i] = copy.deepcopy(u_hist[warm_up:])
    traning_v128[i] = copy.deepcopy(v_hist[warm_up:])
    traning_labelu128[i] = traning_u128[i] - traning_u128[i]**3 - traning_v128[i] - alpha
    traning_labelv128[i] = beta * (traning_u128[i] - traning_v128[i])


    u_solu = copy.deepcopy(u)
    v_solu = copy.deepcopy(v)


    # simulation in 64x64 grid
    # averaging the 128-grid to obtain 64 grid initial condition
    tmp = u_init
    u = (tmp[::2, ::2] + tmp[1::2, ::2] + tmp[::2, 1::2] + tmp[1::2, 1::2])/4
    tmp = v_init
    v = (tmp[::2, ::2] + tmp[1::2, ::2] + tmp[::2, 1::2] + tmp[1::2, 1::2])/4
    n = 64
    dx = widthx/n
    utils.assembly_RDmatrix(n, dt, dx, beta, gamma)
    u_hist = np.zeros([step_num+warm_up, n, n])
    v_hist = np.zeros([step_num+warm_up, n, n])
    u_hist, v_hist = utils.RD_adi(u, v, alpha=alpha, beta=beta, gamma=gamma, step_num=step_num+warm_up, plot=False)
    traning_u64[i] = copy.deepcopy(u_hist[warm_up:])
    traning_v64[i] = copy.deepcopy(v_hist[warm_up:])
    traning_labelu64[i] = traning_u64[i] - traning_u64[i]**3 - traning_v64[i] - alpha
    traning_labelv64[i] = beta * (traning_u64[i] - traning_v64[i])          

# save 64 x 64 data
label = np.concatenate([np.expand_dims(traning_labelu64, axis=2), np.expand_dims(traning_labelv64, axis=2)], axis=2)
label_dim = 2
np.savez('../data/RD/64-{}.npz'.format(int(beta*10)), arg=[n, n, dt, T, label_dim], u=traning_u64, v=traning_v64, label=label)

# save 128 x 128 data
n = 128
label = np.concatenate([np.expand_dims(traning_labelu128, axis=2), np.expand_dims(traning_labelv128, axis=2)], axis=2)
np.savez('../data/RD/128-{}.npz'.format(int(beta*10)), arg=[n, n, dt, T, label_dim], u=traning_u128, v=traning_v128, label=label)