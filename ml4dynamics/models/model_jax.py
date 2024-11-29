# Copyright 2024 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""cVAE model definitions."""

import jax.numpy as jnp
from flax import linen as nn
from jax import random


class Encoder(nn.Module):
  """cVAE Encoder."""

  latents: int

  @nn.compact
  def __call__(self, x, c):
    x = jnp.concatenate([x, c], axis=1)
    x = nn.Dense(self.latents, name='fc1')(x)
    x = nn.tanh(x)
    mean_x = nn.Dense(self.latents, name='fc2_mean')(x)
    logvar_x = nn.Dense(self.latents, name='fc2_logvar')(x)
    return mean_x, logvar_x


class Decoder(nn.Module):
  """cVAE Decoder."""

  latents: int
  features: int

  @nn.compact
  def __call__(self, z, c):
    z = jnp.concatenate([z, c], axis=1)
    z = nn.Dense(self.latents, name='fc1')(z)
    z = nn.tanh(z)
    z = nn.Dense(self.features, name='fc2')(z)
    return z


class cVAE(nn.Module):
  """Full cVAE model."""

  latents: int = 128
  features: int = 256

  # def __init__(self, latents, features):
  #   super(cVAE, self).__init__()
  #   self.latents = latents
  #   self.features = features
  #   self.encoder = None
  #   self.decoder = None

  def setup(self):
    self.encoder = Encoder(self.latents)
    self.decoder = Decoder(self.latents, self.features)

  def __call__(self, x, c, z_rng):
    mean, logvar = self.encoder(x, c)
    z = reparameterize(z_rng, mean, logvar)
    recon_x = self.decoder(z, c)
    return recon_x, mean, logvar

  def generate(self, z, c):
    return self.decoder(z, c)


def reparameterize(rng, mean, logvar):
  std = jnp.exp(0.5 * logvar)
  eps = random.normal(rng, logvar.shape)
  return mean + eps * std


def model(latents, features):
  return cVAE(latents=latents, features=features)
