import numpy as np
import yaml
from matplotlib import cm
from matplotlib import pyplot as plt

from ml4dynamics.utils import utils


def main():
  case = "ks"
  with open(f"config/{case}.yaml", "r") as file:
    config_dict = yaml.safe_load(file)
  
  if case == "ns_hit":
    config_dict["sim"]["Re"] = 10000
  elif case == "ks":
    config_dict["sim"]["c"] = 0.8
  config_dict["train"]["sgs"] = "coarse_correction"
  inputs_correction, outputs_correction, _, _, _ = utils.load_data(
    config_dict, 100
  )
  config_dict["train"]["sgs"] = "filter"
  inputs_filter, outputs_filter, _, _, _ = utils.load_data(
    config_dict, 100
  )
  assert np.linalg.norm(
    inputs_correction - inputs_filter
  ) < 1e-14, "The input data should be the same"
  dx = config_dict["sim"]["L"] * np.pi / config_dict["sim"]["n"] *\
    config_dict["sim"]["r"]
  outputs_correction /= dx**2
  # _, model = utils.create_fine_coarse_simulator(config_dict)
  # what_filter = np.fft.rfftn(outputs_filter, axes=(1, 2))
  # what_correction = np.fft.rfftn(outputs_correction, axes=(1, 2))
  # what_correction - what_filter / (
  # 1 - model.dt / 2 * model.nu * model.laplacian[None, ..., None]
  # )

  if case == "ks":
    # calculate the derivative of the filtering via finite difference
    tmp = (np.roll(outputs_filter[..., 0], -1, axis=1) -\
      np.roll(outputs_filter[..., 0], 1, axis=1)) / 2 / dx
    tmp[:, 0] = outputs_filter[:, 1, 0] / 2 / dx
    tmp[:, -1] = -outputs_filter[:, -2, 0] / 2 / dx
    outputs_filter[..., 0] = tmp

    shape = outputs_filter.shape
    im_array = np.zeros((4, 1, shape[1], shape[0]))
    im_array[0, 0] = inputs_filter[..., 0].T
    im_array[1, 0] = outputs_filter[..., 0].T
    im_array[2, 0] = outputs_correction[..., 0].T
    ratio = outputs_filter.max() / outputs_correction.max()
    print(ratio)
    im_array[3, 0] = outputs_filter[..., 0].T - outputs_correction[..., 0].T * ratio
    # utils.plot_with_horizontal_colorbar(
    #   im_array, fig_size=(12, 4), title_array=None,
    #   file_path="results/fig/cmp_ks.png",
    #   dpi=100, cmap=cm.coolwarm,
    # )
    
    _, axs = plt.subplots(2, 2, figsize=(12, 8))
    axs = axs.flatten()
    index_array = [500, 1500, 2500, 3500]
    for i in range(len(index_array)):
      axs[i].set_title(f"t = {index_array[i] * config_dict['sim']['dt']:.2f}")
      axs[i].plot(outputs_filter[index_array[i], :, 0], label=r"$\tau^f$")
      axs[i].plot(outputs_correction[index_array[i], :, 0], label=r"$\tau^c$")
      # axs[i].plot(outputs_filter[index_array[i], :, 0] -\
      #   outputs_correction[index_array[i], :, 0] * ratio, label=r"$\tau^f - \tau^c$")
      axs[i].legend()
    c = config_dict["sim"]["c"]
    plt.savefig(f"results/fig/cmp_c{c:.1f}.png", dpi=300)
    plt.close()

  else:
    n_plots = 2
    index_array = [3500, 4500]
    im_array = np.zeros((4, n_plots, *(outputs_filter.shape[1:3])))
    for i in range(n_plots):
      im_array[0, i] = inputs_filter[index_array[i], ..., 0]
      im_array[1, i] = outputs_filter[index_array[i], ..., 0]
      im_array[2, i] = outputs_correction[index_array[i], ..., 0]
      ratio = outputs_filter[index_array[i]].max() /\
        outputs_correction[index_array[i]].max()
      print(ratio)
      im_array[3, i] = outputs_filter[index_array[i], ..., 0] -\
        outputs_correction[index_array[i], ..., 0] * ratio
    utils.plot_with_horizontal_colorbar(
      im_array, fig_size=(n_plots * 3, 12), title_array=None,
      file_path="results/fig/cmp_Re.png",
      dpi=100, cmap=cm.coolwarm,
    )
  breakpoint()


if __name__ == "__main__":
  main()
