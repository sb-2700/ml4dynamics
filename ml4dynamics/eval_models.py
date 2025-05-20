import argparse
from functools import partial

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import ml_collections
import numpy as np
import yaml
from box import Box
from matplotlib import pyplot as plt

from ml4dynamics.dataset_utils import dataset_utils
from ml4dynamics.utils import utils


def main(config_dict: ml_collections.ConfigDict):
  config = Box(config_dict)
  pde = config.case
  _global = (config.train.input == "global")
  if _global:
    batch_size = config.train.batch_size_global
    arch = "unet"
  else:
    batch_size = config.train.batch_size_local
    arch = "mlp"
  input_labels = config.train.input
  r = config.sim.r

  if pde == "ns_channel":
      model = utils.create_ns_channel_simulator(config_dict)
  else:
    model_fine, model_coarse = utils.create_fine_coarse_simulator(config_dict)
    res_fn, _ = dataset_utils.res_int_fn(config_dict)
  inputs, outputs, _, _, dataset = utils.load_data(config_dict, batch_size)
  one_traj_length = inputs.shape[0] // config.sim.case_num
  mode = "ols"
  if mode == "ae":
    ckpt_path = f"ckpts/{pde}/{dataset}_{mode}_{arch}.pkl"
  else:
    ckpt_path = f"ckpts/{pde}/{dataset}_{config.train.sgs}_{mode}_{arch}.pkl"
  dim = 2
  inputs_ = inputs
  outputs_ = outputs
  if pde == "ks":
    dim = 1
    if config.sim.BC == "Dirichlet-Neumann":
      inputs_ = inputs[:, :-1]
      outputs_ = outputs[:, :-1]

  train_state, _ = utils.prepare_unet_train_state(
    config_dict, ckpt_path, _global, False
  )
  if _global:
    @jax.jit
    def forward_fn(x):
      y_pred, _ = train_state.apply_fn_with_bn(
        {
          "params": train_state.params,
          "batch_stats": train_state.batch_stats
        },
        x,
        is_training=False
      )
      return y_pred
  else:
    @partial(jax.jit, static_argnums=(1, ))
    def _forward_fn(x, is_aug):
      """forward function for the local model

      The shape of the input and output is aligned with the
      global model for the a-posteriori simulation
      """
      tmp = []
      if not is_aug:
        """a-posteriori evaluation"""
        if "u" in input_labels:
          tmp.append(x[:, :-1])
        if "u_x" in input_labels:
          tmp.append(jnp.einsum("ij, ajk -> aik", model_coarse.L1, x[:, :-1]))
        if "u_xx" in input_labels:
          tmp.append(jnp.einsum("ij, ajk -> aik", model_coarse.L2, x[:, :-1]))
        if "u_xxxx" in input_labels:
          tmp.append(jnp.einsum("ij, ajk -> aik", model_coarse.L4, x[:, :-1]))
        x_ = jnp.concatenate(tmp, axis=-1)
        x_ = jnp.concatenate(
          [x_, jnp.zeros((x_.shape[0], 1, x_.shape[-1]))], axis=1
        )
        x_ = x_.reshape(-1, x_.shape[-1])
        return train_state.apply_fn(train_state.params, x_).reshape(x.shape)
      else:
        """a-priori evaluation"""
        x_ = x.reshape(-1, x.shape[-1])
        return train_state.apply_fn(train_state.params,
                                    x_).reshape(*(x.shape[:2]), -1)

  if mode == "ae":
    """visualizing the distribution shift"""

    def ae_loss_fn(x):
      x_pred = forward_fn(x)
      return np.sum(np.linalg.norm(x - x_pred, axis=-2))

    ds = np.zeros(one_traj_length)
    for i in range(one_traj_length):
      ds[i] = ae_loss_fn(inputs[i:i + 1])
    t = np.arange(0, inputs.shape[0]) * config.sim.dt
    print(ds.min(), ds.max())
    plt.plot(t, ds, label="distribution shift")
    plt.savefig(f"results/fig/{pde}_ds.png")
    plt.close()

  elif mode == "ols":
    if not _global: 
      forward_fn = partial(_forward_fn, is_aug=False)
    breakpoint()
    x0_fine = jnp.kron(inputs[0, ..., 0], jnp.ones((r, r)))
    model_fine.set_x_hist(np.fft.rfft2(x0_fine), model_fine.CN)
    model_coarse.set_x_hist(np.fft.rfft(inputs[0, ..., 0]), model_coarse.CN)

    x_hist = utils.eval_a_posteriori(
      config_dict=config_dict,
      forward_fn=forward_fn,
      inputs=inputs_[:one_traj_length],
      outputs=outputs_[:one_traj_length],
      dim=dim,
      beta=0.0,
      fig_name=None,
      _plot=False,
    )

    n_plot = 6
    index_array = np.arange(
      0, n_plot * step_num // n_plot - 1, step_num // n_plot
    )
    im_array = np.zeros((3, n_plot, *(outputs[0, ..., 0]).shape))
    for j in range(n_plot):
      im_array[0, j] = inputs[index_array[j], ..., 0]
      im_array[1, j] = x_hist[index_array[j], ..., 0]
      im_array[2, j] = (inputs - x_hist)[index_array[j], ..., 0]
    plot_with_horizontal_colorbar(
      im_array, (12, 6), None, f"results/fig/{fig_name}.png", 100
    )


    if False:
      """visualizing the sensitivity analysis of the map"""
      jacobian = jax.vmap(jax.jacfwd(forward_fn))
      batch_size = 200
      jac = np.zeros((inputs.shape[1], inputs.shape[1]))
      for i in range(0, inputs.shape[0], batch_size):
        jac += np.sum(jacobian(inputs[i:i + batch_size, ..., 0]), axis=0)
      jac /= inputs.shape[0]
      plt.imshow(jac)
      plt.colorbar()
      plt.savefig(f"results/fig/{pde}_jacobian.png")
      plt.close()


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "-c", "--config", default=None, help="Set the configuration file path."
  )
  args = parser.parse_args()
  with open(f"config/{args.config}.yaml", "r") as file:
    config_dict = yaml.safe_load(file)
  main(config_dict)
