"""
Test the implementation of the tangent space regularization, make sure the
normal vector constructed from AE is approximately orthogonal to the tangent
space.
"""
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import yaml
from matplotlib import pyplot as plt

from ml4dynamics.utils import utils


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

  inputs, outputs, _, _, dataset = utils.load_data(config_dict, 10)
  train_state, _ = utils.prepare_unet_train_state(
    config_dict, f"ckpts/ks/{dataset}_ae_unet.pkl", is_training=False
  )
  ae_fn = partial(train_state.apply_fn_with_bn, 
    {"params": train_state.params,
    "batch_stats": train_state.batch_stats}
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
  
  def calc_cos(i: int):
    x = inputs[i:i+1]
    normal_vector = jax.grad(ae_loss_fn)(x)[:, :-1]
    tangent_vector_ = tangent_vector(x)
    cos1 = jnp.abs(jnp.sum(normal_vector * tangent_vector_, axis=(-2, -1)))
    cos1 /= jnp.linalg.norm(normal_vector) * jnp.linalg.norm(tangent_vector_)
    tangent_vector_ = tangent_vector(x) + outputs[i:i+1, :-1]
    cos2 = jnp.abs(jnp.sum(normal_vector * tangent_vector_, axis=(-2, -1)))
    cos2 /= jnp.linalg.norm(normal_vector) * jnp.linalg.norm(tangent_vector_)
    return cos1, cos2

  x = jnp.array(inputs)
  normal_vector = jax.grad(ae_loss_fn)(x)[:, :-1]
  tangent_vector_ = tangent_vector(x)
  cos1 = jnp.abs(jnp.sum(normal_vector * tangent_vector_, axis=(-2, -1))) /\
    jnp.linalg.norm(normal_vector, axis=(-2, -1)) /\
    jnp.linalg.norm(tangent_vector_, axis=(-2, -1)) 
  tangent_vector_ = tangent_vector(x) + outputs[:, :-1]
  cos2 = jnp.abs(jnp.sum(normal_vector * tangent_vector_, axis=(-2, -1))) /\
    jnp.linalg.norm(normal_vector, axis=(-2, -1)) /\
    jnp.linalg.norm(tangent_vector_, axis=(-2, -1)) 
  t = np.arange(0, inputs.shape[0]) * 0.1
  plt.plot(t, cos1, label="cos1")
  plt.plot(t, cos1, label="cos2")
  plt.yscale("log")
  plt.savefig(f"results/fig/ks_cos.png")
  plt.close()
  print("cos1", cos1)
  print("cos2", cos2)
  breakpoint()

  for i in range(inputs.shape[0]):  
    cos1, cos2 = calc_cos(i)
    print(i, cos1, cos2)
          

if __name__ == "__main__":
  test_ks_tangent_space()
