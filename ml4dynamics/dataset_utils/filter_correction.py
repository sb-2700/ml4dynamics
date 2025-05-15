import numpy as np
import yaml
from matplotlib import cm


from ml4dynamics.utils import utils


def main():
  with open(f"config/ns_hit.yaml", "r") as file:
    config_dict = yaml.safe_load(file)
  
  config_dict["sim"]["Re"] = 10000
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
  n_plots = 2
  im_array = np.zeros((4, n_plots, 64, 64 ))
  index_array = [3500, 4500]
  dx = config_dict["sim"]["L"] * np.pi / config_dict["sim"]["n"] *\
    config_dict["sim"]["r"]
  outputs_correction /= dx**2
  # _, model = utils.create_fine_coarse_simulator(config_dict)
  # what_filter = np.fft.rfftn(outputs_filter, axes=(1, 2))
  # what_correction = np.fft.rfftn(outputs_correction, axes=(1, 2))
  # what_correction - what_filter / (
  # 1 - model.dt / 2 * model.nu * model.laplacian[None, ..., None]
  # )

  for i in range(n_plots):
    im_array[0, i] = inputs_filter[index_array[i], ..., 0]
    im_array[1, i] = outputs_filter[index_array[i], ..., 0]
    im_array[2, i] = outputs_correction[index_array[i], ..., 0]
    ratio = outputs_filter[index_array[i], ..., 0].max() /\
      outputs_correction[index_array[i], ..., 0].max()
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
