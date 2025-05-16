import jax
jax.config.update("jax_enable_x64", True)
import ml_collections
import yaml
from box import Box

from ml4dynamics.utils import utils


def main(config_dict: ml_collections.ConfigDict):
  config = Box(config_dict)
  pde = config.case
  _global = (config.train.input == "global")
  if _global:
    batch_size = config.train.batch_size_global
    arch = "unet"
  else:
    batch_size = config.train.batch_size
    arch = "mlp"
  inputs, outputs, train_dataloader, test_dataloader, dataset = utils.load_data(
    config_dict, batch_size
  )
  one_traj_length = inputs.shape[0] // config.sim.case_num 
  mode = "ols"
  dim = 2
  inputs_ = inputs
  outputs_ = outputs
  if pde == "ks":
    dim = 1
    if config.sim.BC == "Dirichlet-Neumann":
      inputs_ = inputs[:, :-1]
      outputs_ = outputs[:, :-1]

  train_state, _ = utils.prepare_unet_train_state(
    config_dict, f"ckpts/{pde}/{dataset}_{mode}_{arch}.pkl", _global, False
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

    @jax.jit
    def forward_fn(x):
      """forward function for the local model

      The shape of the input and output is aligned with the
      global model for the a-posteriori simulation
      """
      x_ = x.reshape(-1, x.shape[-1])
      return train_state.apply_fn(train_state.params, x_).reshape(x.shape)

  if mode == "ae":
    utils.eval_a_priori(
      forward_fn, train_dataloader, test_dataloader, inputs[:one_traj_length],
      inputs[:one_traj_length], dim, f"reg_{fig_name}"
    )
    return
  utils.eval_a_priori(
    forward_fn, train_dataloader, test_dataloader, inputs[:one_traj_length],
    outputs[:one_traj_length], dim, f"reg_{fig_name}"
  )
  utils.eval_a_posteriori(
    config_dict, forward_fn, inputs_[:one_traj_length],
    outputs_[:one_traj_length], dim, f"sim_{fig_name}"
  )


if __name__ == "__main__":
  with open("config/simulation.yaml", "r") as file:
    config_dict = yaml.safe_load(file)
  main(config_dict)
