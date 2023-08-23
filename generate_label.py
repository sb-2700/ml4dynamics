import numpy as np
import scipy.sparse as spa


case_num = 10
step_num = 100
n = 128
dt = 0.01
gamma = 0.05
alpha = 0.01
beta = 0.25


u64 = np.load('../data/u128-10.npy').reshape([case_num, step_num, n, n])
v64 = np.load('../data/v128-10.npy').reshape([case_num, step_num, n, n])
labelu = np.zeros(u64.shape)
labelv = np.zeros(v64.shape)


dx = 6.4 * 2/n
L = np.eye(n//2) * (-2)
for i in range(1, n//2-1):
    L[i, i-1] = 1
    L[i, i+1] = 1
L[0, 1] = 1
L[0, -1] = 1
L[-1, 0] = 1
L[-1, -2] = 1
L = L/(dx**2)
L = spa.csc_matrix(L)
L2 = spa.kron(L, np.eye(n//2)) + spa.kron(np.eye(n//2), L)
A = spa.eye(n//2*n//2) + L2 * gamma * dt 


for i in range(case_num):
    for j in range(step_num-1):
        u = (u64[i, j, ::2, ::2] + u64[i, j, ::2, 1::2] + u64[i, j, 1::2, ::2] + u64[i, j, 1::2, 1::2])/4
        v = (v64[i, j, ::2, ::2] + v64[i, j, ::2, 1::2] + v64[i, j, 1::2, ::2] + v64[i, j, 1::2, 1::2])/4


        #u = A.dot(u.reshape([n//2*n//2, 1]))
        u = (A.dot(u.reshape([n*n//4, 1]))).reshape([n//2, n//2]) + dt * (u - v - u**3 + alpha)
        v = (A.dot(v.reshape([n*n//4, 1]))).reshape([n//2, n//2]) + beta * dt * (u - v)


        u_ = np.zeros([n, n])
        v_ = np.zeros([n, n])
        u_[::2, ::2] = u
        u_[::2, 1::2] = u
        u_[1::2, ::2] = u
        u_[1::2, 1::2] = u
        v_[::2, ::2] = v
        v_[::2, 1::2] = v
        v_[1::2, ::2] = v
        v_[1::2, 1::2] = v


        labelu[i, j, :, :] = (u64[i, j+1, :, :] - u_)
        labelv[i, j, :, :] = (v64[i, j+1, :, :] - v_)


np.save('../data/labelu128-10.npy', labelu)
np.save('../data/labelv128-10.npy', labelv)