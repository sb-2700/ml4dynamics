import pickle
from functools import partial
from time import time

import jax
import jax.numpy as jnp
import jax.random as random
import ml_collections
import yaml
from box import Box
from flax import traverse_util
from matplotlib import pyplot as plt
from tqdm import tqdm

from ml4dynamics import utils
from ml4dynamics.types import PRNGKey

jax.config.update("jax_enable_x64", True)


def main(config_dict: ml_collections.ConfigDict):

  def train(
    model_type: str,
    epochs: int,
    rng: PRNGKey,
  ):

    @partial(jax.jit, static_argnums=(3, ))
    def train_step(x, y, train_state, is_training=True):

      def loss_fn(params, batch_stats, is_training):
        y_pred, batch_stats = train_state.apply_fn_with_bn(
          {
            "params": params,
            "batch_stats": batch_stats
          },
          x,
          is_training=is_training
        )
        loss = jnp.mean((y - y_pred)**2)

        if model_type == "ae":
          loss = jnp.mean((x - y_pred)**2)
        elif model_type == "mols":
          squared_norms = jax.tree_map(lambda param: jnp.sum(param**2), params)
          loss += jax.tree_util.tree_reduce(
            lambda x, y: x + y, squared_norms
          ) * config.tr.lambda_mols
        elif model_type == "aols":
          pass
        elif model_type == "tr":
          pass

        return loss, batch_stats

      if is_training:
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, batch_stats
         ), grads = grad_fn(train_state.params, train_state.batch_stats, True)

        train_state = train_state.apply_gradients(grads=grads)
        train_state = train_state.update_batch_stats(batch_stats)
      else:
        loss, batch_stats = loss_fn(
          train_state.params, train_state.batch_stats, False
        )

      return loss, train_state

    train_state, schedule = utils.prepare_unet_train_state(config_dict)
    flat_params = traverse_util.flatten_dict(train_state.params)
    total_params = sum(jax.tree_util.tree_map(lambda x: x.size, flat_params).values())
    print(f"total parameters for {model_type}:", total_params)
    iters = tqdm(range(epochs))
    loss_hist = []
    for epoch in iters:
      total_loss = 0
      for batch_inputs, batch_outputs in train_dataloader:
        loss, train_state = train_step(
          jnp.array(batch_inputs), jnp.array(batch_outputs), train_state, True
        )
        total_loss += loss
      lr = schedule(train_state.step)
      desc_str = f"{lr=:.2e}|{total_loss=:.4e}"
      iters.set_description_str(desc_str)
      loss_hist.append(total_loss)

      if (epoch + 1) % config.train.save == 0:
        with open(f"ckpts/{pde_type}/{model_type}.pkl", "wb") as f:
          dict = {
            "params": train_state.params, "batch_stats": train_state.batch_stats
          }
          pickle.dump(dict, f)

    val_loss = 0
    for batch_inputs, batch_outputs in test_dataloader:
      loss, train_state = train_step(
        jnp.array(batch_inputs), jnp.array(batch_outputs), train_state, False
      )
      val_loss += loss
    print(f"val loss: {val_loss:.4e}")
    plt.plot(loss_hist)
    plt.savefig(f"results/fig/loss_hist_{model_type}.pdf")
    plt.clf()

    utils.eval_a_priori(
      train_state, train_dataloader, test_dataloader, inputs, outputs
    )
    utils.eval_a_posteriori(
      config_dict, train_state, inputs, outputs, f"tr_aposteriori_{model_type}"
    )

  config = Box(config_dict)
  pde_type = config.name
  epochs = config.train.epochs
  print("start loading data...")
  start = time()
  inputs, outputs, train_dataloader, test_dataloader = utils.load_data(
    config_dict, config.train.batch_size_unet, mode="jax"
  )
  print(f"finis loading data with {time() - start:.2f}s...")
  rng = jax.random.PRNGKey(config.sim.seed)
  # models_array = ["ae", "ols", "mols", "aols", "tr"]
  models_array = ["ols"]

  for _ in models_array:
    rng, key = random.split(rng)
    print(f"Training {_}...")
    train(_, epochs, key)


if __name__ == "__main__":

  with open("config/simulation.yaml", "r") as file:
    config_dict = yaml.safe_load(file)
  main(config_dict)
