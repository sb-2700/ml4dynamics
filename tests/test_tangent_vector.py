"""
Test the implementation of the tangent space regularization, make sure the
normal vector constructed from AE is approximately orthogonal to the tangent
space.
"""
from functools import partial

import jax
import jax.numpy as jnp
import pytest
import yaml
from jax import random

from ml4dynamics import utils
from ml4dynamics.dynamics import KS


jax.config.update('jax_enable_x64', True)

# @pytest.mark.parametrize(
#     ("hw", "param_count"),
#     [
#         ((128, 128), 34_491_599),
#         # It's fully convolutional => same parameter number.
#         ((256, 256), 34_491_599),
#     ],
#   )

# def test_turing_pattern():
#     """Test whether the set of parameter will generate Turing pattern for RD equation
#     TODO
#     """
#     assert True


# @pytest.mark.skip
def test_ks_tangent_space():

  with open(f"config/ks.yaml", "r") as file:
    config_dict = yaml.safe_load(file)

  inputs, outputs, _, _, dataset = utils.load_data(
    config_dict, 10, mode="jax"
  )
  ae_train_state, _ = utils.prepare_unet_train_state(config_dict, f"ks/{dataset}_ae")
  ae_fn = partial(ae_train_state.apply_fn_with_bn, 
    {"params": ae_train_state.params,
    "batch_stats": ae_train_state.batch_stats}
  )
  def ae_loss_fn(x):
    x_pred, _ = ae_fn(x, is_training=False)
    # loss = jnp.linalg.norm(x - x_pred, axis=-1)
    loss = jnp.sum((x - x_pred)**2, axis=-1)
    return jnp.sum(loss)
  _, model_coarse = utils.create_fine_coarse_simulator(config_dict)
  def tangent_vector(x):

    return jnp.einsum("ij, ajb -> aib", model_coarse.L1, (x[:, :-1]**2)/2) +\
      jnp.einsum("ij, ajb -> aib", 2 * model_coarse.L, x[:, :-1])
  
  x = inputs[0:1]
  normal_vector = jax.grad(ae_loss_fn)(x)[:, :-1]
  loss = jnp.mean(jnp.abs(
    jnp.sum(normal_vector * tangent_vector(x), axis=(-2, -1))
  ))
  breakpoint()
          

if __name__ == "__main__":
  test_ks_tangent_space()
