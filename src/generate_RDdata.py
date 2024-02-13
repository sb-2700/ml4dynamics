###################################################
#                   finished                      #
###################################################
import utils
import numpy as np
import numpy.random as r
import copy
import argparse
import pdb
import ml_collections


def generate_RD_data():
    # parse the simulation arguments
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('--gamma', type=float, default=0.05)
    args = parser.parse_args()
    gamma = args.gamma
    print('Generating RD data with gamma = {:.1f}...'.format(gamma))

    # set simulation parameters
    widthx = 6.4
    widthy = 6.4
    dt = 0.01
    step_num = 2000
    T = step_num * dt
    alpha = 0.01
    beta = 1.0
    warm_up = 200
    patience = 5                           # we admit 50 times blow up generations
    writeInterval = 2
    tol = 1e-7
    r.seed(0)

    # simulating training trajectories
    case_num = 1
    traning_u64 = np.zeros([case_num, step_num//writeInterval, 64, 64])
    traning_v64 = np.zeros([case_num, step_num//writeInterval, 64, 64])
    traning_labelu64 = np.zeros([case_num, step_num//writeInterval, 64, 64])
    traning_labelv64 = np.zeros([case_num, step_num//writeInterval, 64, 64])
    traning_u128 = np.zeros([case_num, step_num//writeInterval, 128, 128])
    traning_v128 = np.zeros([case_num, step_num//writeInterval, 128, 128])
    traning_labelu128 = np.zeros([case_num, step_num//writeInterval, 128, 128])
    traning_labelv128 = np.zeros([case_num, step_num//writeInterval, 128, 128])
    j = 0
    i = 0
    while i < case_num and j < patience:
        print('generating the {}-th trajectory for gamma = {:.2e}'.format(i, gamma))
        # simulation in 128x128 grid
        n = 128
        dx = widthx/n
        u_hist = np.zeros([(step_num+warm_up)//writeInterval, n, n])
        v_hist = np.zeros([(step_num+warm_up)//writeInterval, n, n])
        utils.assembly_RDmatrix(n, dt, dx, beta, gamma)
        u_init = r.randn(n, n)
        v_init = r.randn(n, n)


        u = copy.deepcopy(u_init)
        v = copy.deepcopy(v_init)
        u_hist, v_hist, flag = utils.RD_adi(u, v, dt, alpha=alpha, beta=beta, gamma=gamma, step_num=step_num+warm_up, writeInterval=writeInterval, plot=False)
        if flag == False:
            j = j+1
            continue
        traning_u128[i] = copy.deepcopy(u_hist[warm_up//writeInterval:])
        traning_v128[i] = copy.deepcopy(v_hist[warm_up//writeInterval:])
        traning_labelu128[i] = traning_u128[i] - traning_u128[i]**3 - traning_v128[i] + alpha
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
        u_hist = np.zeros([(step_num+warm_up)//writeInterval, n, n])
        v_hist = np.zeros([(step_num+warm_up)//writeInterval, n, n])
        u_hist, v_hist, flag = utils.RD_adi(u, v, dt, alpha=alpha, beta=beta, gamma=gamma, step_num=step_num+warm_up, writeInterval=writeInterval, plot=False)
        if flag == False:
            j = j+1
            continue
        traning_u64[i] = copy.deepcopy(u_hist[warm_up//writeInterval:])
        traning_v64[i] = copy.deepcopy(v_hist[warm_up//writeInterval:])
        traning_labelu64[i] = traning_u64[i] - traning_u64[i]**3 - traning_v64[i] + alpha
        traning_labelv64[i] = beta * (traning_u64[i] - traning_v64[i]) 
        i = i+1         

    if i == case_num:
        # save 64 x 64 data
        U = np.concatenate([np.expand_dims(traning_u64, axis=2), np.expand_dims(traning_v64, axis=2)], axis=2)
        label = np.concatenate([np.expand_dims(traning_labelu64, axis=2), np.expand_dims(traning_labelv64, axis=2)], axis=2)
        label_dim = 2
        np.savez('../data/RD/64-{}.npz'.format(int(gamma*20)), arg=[n, n, dt*writeInterval, T, label_dim], U=U, label=label)

        # save 128 x 128 data
        n = 128
        U = np.concatenate([np.expand_dims(traning_u128, axis=2), np.expand_dims(traning_v128, axis=2)], axis=2)
        label = np.concatenate([np.expand_dims(traning_labelu128, axis=2), np.expand_dims(traning_labelv128, axis=2)], axis=2)
        np.savez('../data/RD/128-{}.npz'.format(int(gamma*20)), arg=[n, n, dt*writeInterval, T, label_dim], U=U, label=label)