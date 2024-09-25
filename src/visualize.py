import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm


def plot_from_h5py(case: str):
  '''Plot the fluid field from the h5py file'''

  path = 'data/RD/' + case + '.h5'
  with h5py.File(path, 'r') as file:
    U = file['input'][:]
    case_num, nt, dim, nx, ny = U.shape
    T = file["end_time"][()]
    dt = T / nt

  n1 = 5
  n2 = 5

  for k in range(10):
    random_integer = np.random.randint(1, 101)
    print(random_integer)
    plt.figure(figsize=(20, 20))
    for i in range(n1):
      for j in range(n2):
        plt.subplot(n2, n1, i * n2 + j + 1)
        plt.imshow(U[random_integer, (i * n2 + j)*30, 0, :, :], cmap=cm.jet)
        plt.axis('off')
    plt.savefig('results/fig/traj_' + str(random_integer) + '.pdf')


def plot_from_h5py_cmp():
  '''Plot the fluid field from the h5py file'''

  path = 'data/RD/128-100.h5'
  with h5py.File(path, 'r') as file:
    U128 = file['input'][:]

  path = 'data/RD/64-100.h5'
  with h5py.File(path, 'r') as file:
    U64 = file['input'][:]

  n1 = 5
  n2 = 5

  for k in range(10):
    random_integer = np.random.randint(1, 101)
    print(random_integer)
    plt.figure(figsize=(20, 20))
    for i in range(n1):
      for j in range(n2):
        plt.subplot(n2, n1, i * n2 + j + 1)
        plt.imshow(U128[random_integer, (i * n2 + j)*30, 0, :, :], cmap=cm.jet)
        plt.axis('off')
    plt.savefig('results/fig/traj_128_' + str(random_integer) + '.pdf')
    plt.clf()

    plt.figure(figsize=(20, 20))
    for i in range(n1):
      for j in range(n2):
        plt.subplot(n2, n1, i * n2 + j + 1)
        plt.imshow(U64[random_integer, (i * n2 + j)*30, 0, :, :], cmap=cm.jet)
        plt.axis('off')
    plt.savefig('results/fig/traj_64_' + str(random_integer) + '.pdf')
