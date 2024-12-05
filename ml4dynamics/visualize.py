import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm


def plot_from_h5py(case: str):
  """Plot the fluid field from the h5py file"""

  path = "data/RD/" + case + ".h5"
  with h5py.File(path, "r") as file:
    U = file["input"][:]
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
        plt.imshow(U[random_integer, (i * n2 + j) * 30, 0, :, :], cmap=cm.jet)
        plt.axis("off")
    plt.savefig("results/fig/traj_" + str(random_integer) + ".pdf")


def plot_from_h5py_cmp():
  """Plot the fluid field from the h5py file"""

  path = "data/RD/128-100.h5"
  with h5py.File(path, "r") as file:
    U128 = file["input"][:]

  path = "data/RD/64-100.h5"
  with h5py.File(path, "r") as file:
    U64 = file["input"][:]

  n1 = 5
  n2 = 5

  for k in range(10):
    random_integer = np.random.randint(1, 101)
    print(random_integer)
    plt.figure(figsize=(20, 20))
    for i in range(n1):
      for j in range(n2):
        plt.subplot(n2, n1, i * n2 + j + 1)
        plt.imshow(
          U128[random_integer, (i * n2 + j) * 30, 0, :, :], cmap=cm.viridis
        )
        plt.axis("off")
    plt.savefig("results/fig/traj_128_" + str(random_integer) + ".pdf")
    plt.clf()

    plt.figure(figsize=(20, 20))
    for i in range(n1):
      for j in range(n2):
        plt.subplot(n2, n1, i * n2 + j + 1)
        plt.imshow(
          U64[random_integer, (i * n2 + j) * 30, 0, :, :], cmap=cm.viridis
        )
        plt.axis("off")
    plt.savefig("results/fig/traj_64_" + str(random_integer) + ".pdf")


def plot_stats(
  t_array: np.array,
  fine_traj: np.ndarray,
  baseline: np.ndarray,
  correction1: np.ndarray,
  correction2: np.ndarray,
  fig_name: str,
):

  avg_length = 500
  r = fine_traj.shape[1] // baseline.shape[1]
  plt.figure(figsize=(6, 6))
  plt.subplot(211)
  plt.plot(t_array, np.mean(fine_traj, axis=1), label="truth")
  plt.plot(
    t_array,
    np.mean(baseline, axis=1),
    label="baseline={:.3e}|MSE={:.3e}".format(
      # np.linalg.norm(
      #   np.mean(fine_traj, axis=1) - np.mean(baseline, axis=1)
      # )
      np.mean(fine_traj[-avg_length:]) - np.mean(baseline[-avg_length:]),
      np.mean((fine_traj[:, r - 1::r] - baseline)**2)
    )
  )
  plt.plot(
    t_array,
    np.mean(correction1, axis=1),
    label="err2={:.3e}|MSE={:.3e}".format(
      # np.linalg.norm(
      #   np.mean(fine_traj, axis=1) - np.mean(correction1, axis=1)
      # )
      np.mean(fine_traj[-avg_length:]) - np.mean(correction1[-avg_length:]),
      np.mean((fine_traj[:, r - 1::r] - correction1)**2)
    )
  )
  if correction2 is not None:
    plt.plot(
      t_array,
      np.mean(correction2, axis=1),
      label="err3={:.3e}|MSE={:.3e}".format(
        # np.linalg.norm(
        #   np.mean(fine_traj, axis=1) - np.mean(correction2, axis=1)
        # )
        np.mean(fine_traj[-avg_length:]) - np.mean(correction2[-avg_length:]),
        np.mean((fine_traj[:, r - 1::r] - correction2)**2)
      )
    )
  plt.legend()
  plt.xlabel(r"$T$")
  plt.ylabel(r"$\overline{u}$")
  plt.subplot(212)
  plt.plot(t_array, np.mean(fine_traj**2, axis=1), label="truth")
  plt.plot(
    t_array,
    np.mean(baseline**2, axis=1),
    label="baseline={:.3e}".format(
      # np.linalg.norm(
      #   np.mean(fine_traj**2, axis=1) - np.mean(baseline**2, axis=1)
      # )
      np.mean(fine_traj[-avg_length:]**2) - np.mean(baseline[-avg_length:]**2),
    )
  )
  plt.plot(
    t_array,
    np.mean(correction1**2, axis=1),
    label="err2={:.3e}".format(
      # np.linalg.norm(
      #   np.mean(fine_traj**2, axis=1) - np.mean(correction1**2, axis=1)
      # )
      np.mean(fine_traj[-avg_length:]**2) -
      np.mean(correction1[-avg_length:]**2),
    )
  )
  if correction2 is not None:
    plt.plot(
      t_array,
      np.mean(correction2**2, axis=1),
      label="err3={:.3e}".format(
        # np.linalg.norm(
        #   np.mean(fine_traj**2, axis=1) - np.mean(correction2**2, axis=1)
        # )
        np.mean(fine_traj[-avg_length:]**2) -
        np.mean(correction2[-avg_length:]**2),
      )
    )
  plt.legend()
  plt.xlabel(r"$T$")
  plt.ylabel(r"$\overline{u^2}$")
  plt.savefig(fig_name)
  print(
    "rmse without correction: {:.3e}".format(
      np.sqrt(np.mean((baseline.T - fine_traj[:, r - 1::r].T)**2))
    )
  )
  print(
    "rmse with correction1: {:.3e}".format(
      np.sqrt(np.mean((correction1.T - fine_traj[:, r - 1::r].T)**2))
    )
  )
  if correction2 is not None:
    print(
      "rmse with correction2: {:.3e}".format(
        np.sqrt(np.mean((correction2.T - fine_traj[:, r - 1::r].T)**2))
      )
    )


def plot_error_cloudmap(
  err: np.ndarray,
  u: np.ndarray,
  u_x: np.ndarray,
  u_xx: np.ndarray,
  u_xxxx: np.ndarray,
  name: str,
):

  plt.subplot(511)
  plt.imshow(u)
  plt.colorbar()
  plt.axis("off")
  plt.ylabel(r"$\Delta \tau$")
  plt.subplot(512)
  plt.imshow(err)
  plt.colorbar()
  plt.axis("off")
  plt.ylabel(r"$u$")
  plt.subplot(513)
  plt.imshow(u_x)
  plt.colorbar()
  plt.axis("off")
  plt.ylabel(r"$u_x$")
  plt.subplot(514)
  plt.imshow(u_xx)
  plt.colorbar()
  plt.axis("off")
  plt.ylabel(r"$u_{xx}$")
  plt.subplot(515)
  plt.imshow(u_xxxx)
  plt.colorbar()
  plt.axis("off")
  plt.ylabel(r"$u_{xxxx}$")
  plt.savefig(f"results/fig/{name}_err_dist.pdf")
  plt.clf()
