import h5py
import matplotlib.pyplot as plt
from matplotlib import cm


def plot_from_h5py(case: str):
  '''Plot the fluid field from the h5py file'''
  path = 'data/' + case + '.h5'
  with h5py.File(path, 'r') as file:
    U = file['data'][:]
    nx, ny, nz = U.shape[-3:]
    start_time = file['start_time'][()]
    end_time = file['end_time'][()]
    write_interval = file['write_interval'][()]

  n1 = 3
  n2 = 8
  for i in range(n1):
    for j in range(n2):
      plt.subplot(n2, n1, i * n2 + j + 1)
      plt.imshow(U[i * n2 + j, :, :, 0, 1].T, cmap=cm.jet)
      plt.axis('off')
  plt.savefig('results/fig/traj_' + case + '.pdf')
