###################################################
#                   finished                      #
###################################################
import utils
import numpy as np
import numpy.random as r
import copy
import argparse


parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--beta', type=float, default=0.2)
args = parser.parse_args()
beta = args.beta
print('Generating RD data with beta = {:.1f}...'.format(beta))


widthx = 6.4
widthy = 6.4
dt = 0.01
T = 2
step_num = 200
alpha = 0.01
gamma = 0.05
warm_up = 5
step_num = warm_up + step_num
tol = 1e-7
r.seed(0)


case_num = 10
traning_u64 = np.zeros([case_num, step_num, 64*64])
traning_v64 = np.zeros([case_num, step_num, 64*64])
traning_labelu64 = np.zeros([case_num, step_num, 64*64])
traning_labelv64 = np.zeros([case_num, step_num, 64*64])
traning_u128 = np.zeros([case_num, step_num, 128*128])
traning_v128 = np.zeros([case_num, step_num, 128*128])
traning_labelu128 = np.zeros([case_num, step_num, 128*128])
traning_labelv128 = np.zeros([case_num, step_num, 128*128])
for i in range(case_num):
    print('generating the {}-th trajectory...'.format(i))
    # simulation in 128x128 grid
    n = 128
    dx = widthx/n
    u_hist = np.zeros([step_num, n*n])
    v_hist = np.zeros([step_num, n*n])
    utils.assembly_RDmatrix(n, dt, dx, beta, gamma)
    u_init = r.randn(n*n)
    v_init = r.randn(n*n)


    u = copy.deepcopy(u_init)
    v = copy.deepcopy(v_init)
    u_hist, v_hist = utils.RD_semi(u, v, alpha=alpha, beta=beta, gamma=gamma, step_num=step_num, plot=False)
    traning_u128[i, :, :] = copy.deepcopy(u_hist)
    traning_v128[i, :, :] = copy.deepcopy(v_hist)
    traning_labelu128[i, :, :] = traning_u128[i, :, :] - traning_u128[i, :, :]**3 - traning_v128[i, :, :] - alpha
    traning_labelv128[i, :, :] = beta * (traning_u128[i, :, :] - traning_v128[i, :, :])


    u_solu = copy.deepcopy(u)
    v_solu = copy.deepcopy(v)


    # simulation in 64x64 grid
    # averaging the 128-grid to obtain 64 grid initial condition
    tmp = u_init.reshape([n, n])
    u = (tmp[::2, ::2] + tmp[1::2, ::2] + tmp[::2, 1::2] + tmp[1::2, 1::2])/4
    tmp = v_init.reshape([n, n])
    v = (tmp[::2, ::2] + tmp[1::2, ::2] + tmp[::2, 1::2] + tmp[1::2, 1::2])/4
    n = 64
    dx = widthx/n
    utils.assembly_RDmatrix(n, dt, dx, beta, gamma)
    u_hist = np.zeros([step_num, n*n])
    v_hist = np.zeros([step_num, n*n])
    u = u.reshape(n*n)
    v = v.reshape(n*n)
    u_hist, v_hist = utils.RD_semi(u, v, alpha=alpha, beta=beta, gamma=gamma, step_num=step_num, plot=False)
    traning_u64[i, :, :] = copy.deepcopy(u_hist)
    traning_v64[i, :, :] = copy.deepcopy(v_hist)
    traning_labelu64[i, :, :] = traning_u64[i, :, :] - traning_u64[i, :, :]**3 - traning_v64[i, :, :] - alpha
    traning_labelv64[i, :, :] = beta * (traning_u64[i, :, :] - traning_v64[i, :, :])          


u = traning_u64[:, warm_up:, :]
v = traning_v64[:, warm_up:, :]
labelu = traning_labelu64[:, warm_up:, :]
labelv = traning_labelv64[:, warm_up:, :]
label = np.concatenate([np.expand_dims(labelu, axis=2), np.expand_dims(labelv, axis=2)], axis=2)
label_dim = 2
np.savez('../data/RD/64-{}.npz'.format(int(beta*10)), arg=[n, n, dt, T, label_dim], u=u, v=v, label=label)


u = traning_u128[:, warm_up:, :]
v = traning_v128[:, warm_up:, :]
labelu = traning_labelu128[:, warm_up:, :]
labelv = traning_labelv128[:, warm_up:, :]
label = np.concatenate([np.expand_dims(labelu, axis=2), np.expand_dims(labelv, axis=2)], axis=2)
n = 128
np.savez('../data/RD/128-{}.npz'.format(int(beta*10)), arg=[n, n, dt, T, label_dim], u=u, v=v, label=label)