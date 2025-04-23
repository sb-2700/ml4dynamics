import argparse
import pickle
from functools import partial
from time import time

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import yaml
from box import Box
from flax import traverse_util
from matplotlib import pyplot as plt
from tqdm import tqdm

from ml4dynamics import utils


def main():

  def train(
    model_type: str,
    epochs: int,
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
    total_params = sum(
      jax.tree_util.tree_map(lambda x: x.size, flat_params).values()
    )
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
            "params": train_state.params,
            "batch_stats": train_state.batch_stats
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
    plt.yscale("log")
    plt.savefig(f"results/fig/loss_hist_{model_type}.png")
    plt.clf()

    dim = 2
    inputs_ = inputs
    outputs_ = outputs
    if pde_type == "ks":
      dim = 1
      if config.sim.BC == "Dirichlet-Neumann":
        inputs_ = inputs[:, :-1]
        outputs_ = outputs[:, :-1]
    utils.eval_a_priori(
      train_state, train_dataloader, test_dataloader, inputs, outputs, dim,
      f"{args.config}_apriori_{model_type}"
    )
    utils.eval_a_posteriori(
      config_dict, train_state, inputs_, outputs_, dim,
      f"{args.config}_aposteriori_{model_type}"
    )

  parser = argparse.ArgumentParser()
  parser.add_argument(
    "-c", "--config", default=None, help="Set the configuration file path."
  )
  args = parser.parse_args()
  with open(f"config/{args.config}.yaml", "r") as file:
    config_dict = yaml.safe_load(file)

  config = Box(config_dict)
  pde_type = config.case
  epochs = config.train.epochs
  print("start loading data...")
  start = time()
  inputs, outputs, train_dataloader, test_dataloader = utils.load_data(
    config_dict, config.train.batch_size_unet, mode="jax"
  )
  print(f"finis loading data with {time() - start:.2f}s...")
  # models_array = ["ae", "ols", "mols", "aols", "tr"]
  models_array = ["ols"]

  for _ in models_array:
    print(f"Training {_}...")
    train(_, epochs)


if __name__ == "__main__":
  main()
