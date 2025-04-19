import pickle

import jax
import ml_collections
import optax
import yaml
from box import Box

from ml4dynamics import utils
from ml4dynamics.models.models_jax import CustomTrainState, UNet
from ml4dynamics.types import PRNGKey

jax.config.update("jax_enable_x64", True)


def main(config_dict: ml_collections.ConfigDict):
  config = Box(config_dict)

  with open("ckpts/react_diff/ols.pkl", 'rb') as f:
    dict = pickle.load(f)
  inputs, outputs, train_dataloader, test_dataloader = utils.load_data(
    config_dict, config.train.batch_size_unet, mode="jax"
  )
  unet = UNet(2)
  optimizer = optax.adam(0.001)
  train_state = CustomTrainState.create(
    apply_fn=unet.apply,
    params=dict["params"],
    tx=optimizer,
    batch_stats=dict["batch_stats"]
  )

  utils.eval_a_priori(
    train_state, train_dataloader, test_dataloader, inputs, outputs
  )
  utils.eval_a_posteriori(
    config_dict, train_state, inputs, outputs, "aposteriori"
  )


if __name__ == "__main__":
  with open("config/simulation.yaml", "r") as file:
    config_dict = yaml.safe_load(file)
  main(config_dict)
