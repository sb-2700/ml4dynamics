import sys
from pathlib import Path

ROOT_PATH = str(Path(__file__).parent.parent)
sys.path.append(ROOT_PATH)

import jax.numpy as jnp
from jax.experimental import sparse

case_num = 10
step_num = 100
n = 128
dt = 0.01
gamma = 0.05
alpha = 0.01
beta = 0.25

u64 = jnp.load(ROOT_PATH +
               '/data/u128-10.npy').reshape([case_num, step_num, n, n])
v64 = jnp.load(ROOT_PATH +
               '/data/v128-10.npy').reshape([case_num, step_num, n, n])
labelu = jnp.zeros(u64.shape)
labelv = jnp.zeros(v64.shape)

dx = 6.4 * 2 / n
L = jnp.eye(n // 2) * (-2)
for i in range(1, n // 2 - 1):
  L[i, i - 1] = 1
  L[i, i + 1] = 1
L[0, 1] = 1
L[0, -1] = 1
L[-1, 0] = 1
L[-1, -2] = 1
L = L / (dx**2)
L = sparse.csc_matrix(L)
L2 = sparse.kron(L, jnp.eye(n // 2)) + sparse.kron(jnp.eye(n // 2), L)
A = sparse.eye(n // 2 * n // 2) + L2 * gamma * dt

for i in range(case_num):
  for j in range(step_num - 1):
    u = (
      u64[i, j, ::2, ::2] + u64[i, j, ::2, 1::2] + u64[i, j, 1::2, ::2] +
      u64[i, j, 1::2, 1::2]
    ) / 4
    v = (
      v64[i, j, ::2, ::2] + v64[i, j, ::2, 1::2] + v64[i, j, 1::2, ::2] +
      v64[i, j, 1::2, 1::2]
    ) / 4

    #u = A.dot(u.reshape([n//2*n//2, 1]))
    u = (A.dot(u.reshape([n * n // 4, 1])
               )).reshape([n // 2, n // 2]) + dt * (u - v - u**3 + alpha)
    v = (A.dot(v.reshape([n * n // 4, 1]))).reshape([n // 2, n // 2]
                                                    ) + beta * dt * (u - v)

    u_ = jnp.zeros([n, n])
    v_ = jnp.zeros([n, n])
    u_[::2, ::2] = u
    u_[::2, 1::2] = u
    u_[1::2, ::2] = u
    u_[1::2, 1::2] = u
    v_[::2, ::2] = v
    v_[::2, 1::2] = v
    v_[1::2, ::2] = v
    v_[1::2, 1::2] = v

    labelu[i, j, :, :] = (u64[i, j + 1, :, :] - u_)
    labelv[i, j, :, :] = (v64[i, j + 1, :, :] - v_)

jnp.save('../data/labelu128-10.npy', labelu)
jnp.save('../data/labelv128-10.npy', labelv)
