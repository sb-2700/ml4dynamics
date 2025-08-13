# Copyright 2024 The Flax Authors.(https://flax.readthedocs.io/en/latest/api_reference/flax.errors.html#flax
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

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training.train_state import TrainState
from jax import random


class CustomTrainState(TrainState):
  batch_stats: dict

  def apply_fn_with_bn(self, *args, is_training, **nargs):
    output, mutated_vars = self.apply_fn(
      *args,
      **nargs,
      mutable=["batch_stats"],
      rngs={'dropout': jax.random.PRNGKey(2)}
    )
    new_batch_stats = mutated_vars["batch_stats"]
    return output, new_batch_stats

  def update_batch_stats(self, new_batch_stats):
    return self.replace(batch_stats=new_batch_stats)


class MLP(nn.Module):
  output_dim: int
  hidden_dim: int = 32
  dtype: str = jnp.float64

  @nn.compact
  def __call__(self, inputs):
    non_linear = nn.tanh
    x = inputs.reshape(inputs.shape[0], -1)
    x = nn.Dense(self.hidden_dim, param_dtype=self.dtype)(x)
    x = non_linear(x)
    x = nn.Dense(self.hidden_dim, param_dtype=self.dtype)(x)
    x = non_linear(x)
    x = nn.Dense(self.output_dim, param_dtype=self.dtype)(x)
    return x + nn.Dense(self.output_dim, param_dtype=self.dtype)(
      inputs.reshape(inputs.shape[0], -1)
    )


"""cVAE model definitions."""


class vae_Encoder(nn.Module):
  """cVAE Encoder."""

  latents: int
  dtype: str = jnp.float64

  @nn.compact
  def __call__(self, x, c):
    x = jnp.concatenate([x, c], axis=1)
    x = nn.Dense(self.latents, name='fc1', param_dtype=self.dtype)(x)
    x = nn.tanh(x)
    mean_x = nn.Dense(self.latents, name='fc2_mean', param_dtype=self.dtype)(x)
    logvar_x = nn.Dense(
      self.latents, name='fc2_logvar', param_dtype=self.dtype
    )(x)
    return mean_x, logvar_x


class vae_Decoder(nn.Module):
  """cVAE Decoder."""

  latents: int
  features: int
  dtype: str = jnp.float64

  @nn.compact
  def __call__(self, z, c):
    z = jnp.concatenate([z, c], axis=1)
    z = nn.Dense(self.latents, name='fc1', param_dtype=self.dtype)(z)
    z = nn.tanh(z)
    z = nn.Dense(self.features, name='fc2', param_dtype=self.dtype)(z)
    return z


class cVAE(nn.Module):
  """Full cVAE model."""

  latents: int = 128
  features: int = 256
  dtype: str = jnp.float64

  def setup(self):
    self.encoder = vae_Encoder(self.latents, self.dtype)
    self.decoder = vae_Decoder(self.latents, self.features, self.dtype)

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


"""UNet model definitions.
Implementation adapted from https://gitlab.com/1kaiser/jax-unet
"""


class Encoder1D(nn.Module):
  features: int = 2
  kernel_size: int = 3
  dtype: str = jnp.float64
  training: bool = True

  @nn.compact
  def __call__(self, x):
    # Block 1
    z1 = nn.Conv(
      self.features, kernel_size=(self.kernel_size, ), param_dtype=self.dtype
    )(x)
    z1 = nn.relu(z1)
    z1 = nn.Conv(
      self.features, kernel_size=(self.kernel_size, ), param_dtype=self.dtype
    )(z1)
    z1 = nn.BatchNorm(
      use_running_average=not self.training, param_dtype=self.dtype
    )(z1)
    z1 = nn.relu(z1)
    z1_pool = nn.max_pool(z1, window_shape=(2, ), strides=(2, ))

    # Block 2
    z2 = nn.Conv(
      self.features * 2,
      kernel_size=(self.kernel_size, ),
      param_dtype=self.dtype
    )(z1_pool)
    z2 = nn.relu(z2)
    z2 = nn.Conv(
      self.features * 2,
      kernel_size=(self.kernel_size, ),
      param_dtype=self.dtype
    )(z2)
    z2 = nn.BatchNorm(
      use_running_average=not self.training, param_dtype=self.dtype
    )(z2)
    z2 = nn.relu(z2)
    z2_pool = nn.max_pool(z2, window_shape=(2, ), strides=(2, ))

    # Block 3
    z3 = nn.Conv(
      self.features * 4,
      kernel_size=(self.kernel_size, ),
      param_dtype=self.dtype
    )(z2_pool)
    z3 = nn.relu(z3)
    z3 = nn.Conv(
      self.features * 4,
      kernel_size=(self.kernel_size, ),
      param_dtype=self.dtype
    )(z3)
    z3 = nn.BatchNorm(
      use_running_average=not self.training, param_dtype=self.dtype
    )(z3)
    z3 = nn.relu(z3)
    z3_pool = nn.max_pool(z3, window_shape=(2, ), strides=(2, ))

    # Block 4
    z4 = nn.Conv(
      self.features * 8,
      kernel_size=(self.kernel_size, ),
      param_dtype=self.dtype
    )(z3_pool)
    z4 = nn.relu(z4)
    z4 = nn.Conv(
      self.features * 8,
      kernel_size=(self.kernel_size, ),
      param_dtype=self.dtype
    )(z4)
    z4 = nn.BatchNorm(
      use_running_average=not self.training, param_dtype=self.dtype
    )(z4)
    z4 = nn.relu(z4)
    z4_dropout = nn.Dropout(0.5, deterministic=not self.training)(z4)
    z4_pool = nn.max_pool(z4_dropout, window_shape=(2, ), strides=(2, ))

    # Block 5 (bottleneck)
    z5 = nn.Conv(
      self.features * 16,
      kernel_size=(self.kernel_size, ),
      param_dtype=self.dtype
    )(z4_pool)
    z5 = nn.relu(z5)
    z5 = nn.Conv(
      self.features * 16,
      kernel_size=(self.kernel_size, ),
      param_dtype=self.dtype
    )(z5)
    z5 = nn.BatchNorm(
      use_running_average=not self.training, param_dtype=self.dtype
    )(z5)
    z5 = nn.relu(z5)
    z5_dropout = nn.Dropout(0.5, deterministic=not self.training)(z5)

    return z1, z2, z3, z4_dropout, z5_dropout


class Decoder1D(nn.Module):
  features: int = 2
  output_features: int = 2
  kernel_size: int = 3
  dtype: str = jnp.float64
  training: bool = True

  @nn.compact
  def __call__(self, z1, z2, z3, z4, z5):
    # Up Block 1
    z6_up = jax.image.resize(
      z5, shape=(z5.shape[0], z5.shape[1] * 2, z5.shape[2]), method='nearest'
    )
    z6 = nn.Conv(self.features * 8, kernel_size=(2, ),
                 param_dtype=self.dtype)(z6_up)
    z6 = nn.relu(z6)
    z6 = jnp.concatenate([z4, z6], axis=-1)
    z6 = nn.Conv(
      self.features * 8,
      kernel_size=(self.kernel_size, ),
      param_dtype=self.dtype
    )(z6)
    z6 = nn.relu(z6)
    z6 = nn.Conv(
      self.features * 8,
      kernel_size=(self.kernel_size, ),
      param_dtype=self.dtype
    )(z6)
    z6 = nn.BatchNorm(
      use_running_average=not self.training, param_dtype=self.dtype
    )(z6)
    z6 = nn.relu(z6)

    # Up Block 2
    z7_up = jax.image.resize(
      z6, shape=(z6.shape[0], z6.shape[1] * 2, z6.shape[2]), method='nearest'
    )
    z7 = nn.Conv(self.features * 4, kernel_size=(2, ),
                 param_dtype=self.dtype)(z7_up)
    z7 = nn.relu(z7)
    z7 = jnp.concatenate([z3, z7], axis=-1)
    z7 = nn.Conv(
      self.features * 4,
      kernel_size=(self.kernel_size, ),
      param_dtype=self.dtype
    )(z7)
    z7 = nn.relu(z7)
    z7 = nn.Conv(
      self.features * 4,
      kernel_size=(self.kernel_size, ),
      param_dtype=self.dtype
    )(z7)
    z7 = nn.BatchNorm(
      use_running_average=not self.training, param_dtype=self.dtype
    )(z7)
    z7 = nn.relu(z7)

    # Up Block 3
    z8_up = jax.image.resize(
      z7, shape=(z7.shape[0], z7.shape[1] * 2, z7.shape[2]), method='nearest'
    )
    z8 = nn.Conv(self.features * 2, kernel_size=(2, ),
                 param_dtype=self.dtype)(z8_up)
    z8 = nn.relu(z8)
    z8 = jnp.concatenate([z2, z8], axis=-1)
    z8 = nn.Conv(
      self.features * 2,
      kernel_size=(self.kernel_size, ),
      param_dtype=self.dtype
    )(z8)
    z8 = nn.relu(z8)
    z8 = nn.Conv(
      self.features * 2,
      kernel_size=(self.kernel_size, ),
      param_dtype=self.dtype
    )(z8)
    z8 = nn.BatchNorm(
      use_running_average=not self.training, param_dtype=self.dtype
    )(z8)
    z8 = nn.relu(z8)

    # Up Block 4
    z9_up = jax.image.resize(
      z8, shape=(z8.shape[0], z8.shape[1] * 2, z8.shape[2]), method='nearest'
    )
    z9 = nn.Conv(self.features, kernel_size=(2, ),
                 param_dtype=self.dtype)(z9_up)
    z9 = nn.relu(z9)
    z9 = jnp.concatenate([z1, z9], axis=-1)
    z9 = nn.Conv(
      self.features, kernel_size=(self.kernel_size, ), param_dtype=self.dtype
    )(z9)
    z9 = nn.relu(z9)
    z9 = nn.Conv(
      self.features, kernel_size=(self.kernel_size, ), param_dtype=self.dtype
    )(z9)
    z9 = nn.BatchNorm(
      use_running_average=not self.training, param_dtype=self.dtype
    )(z9)
    z9 = nn.relu(z9)

    # Final output
    y = nn.Conv(
      self.output_features, kernel_size=(1, ), param_dtype=self.dtype
    )(z9)
    return y


class Encoder2D(nn.Module):
  features: int = 2
  kernel_size: int = 3
  dtype: str = jnp.float64
  training: bool = True

  @nn.compact
  def __call__(self, x):
    z1 = nn.Conv(
      self.features,
      kernel_size=(self.kernel_size, self.kernel_size),
      param_dtype=self.dtype
    )(x)
    z1 = nn.relu(z1)
    z1 = nn.Conv(
      self.features,
      kernel_size=(self.kernel_size, self.kernel_size),
      param_dtype=self.dtype
    )(z1)
    z1 = nn.BatchNorm(
      use_running_average=not self.training, param_dtype=self.dtype
    )(z1)
    z1 = nn.relu(z1)
    z1_pool = nn.max_pool(z1, window_shape=(2, 2), strides=(2, 2))

    z2 = nn.Conv(
      self.features * 2,
      kernel_size=(self.kernel_size, self.kernel_size),
      param_dtype=self.dtype
    )(z1_pool)
    z2 = nn.relu(z2)
    z2 = nn.Conv(
      self.features * 2,
      kernel_size=(self.kernel_size, self.kernel_size),
      param_dtype=self.dtype
    )(z2)
    z2 = nn.BatchNorm(
      use_running_average=not self.training, param_dtype=self.dtype
    )(z2)
    z2 = nn.relu(z2)
    z2_pool = nn.max_pool(z2, window_shape=(2, 2), strides=(2, 2))

    z3 = nn.Conv(
      self.features * 4,
      kernel_size=(self.kernel_size, self.kernel_size),
      param_dtype=self.dtype
    )(z2_pool)
    z3 = nn.relu(z3)
    z3 = nn.Conv(
      self.features * 4,
      kernel_size=(self.kernel_size, self.kernel_size),
      param_dtype=self.dtype
    )(z3)
    z3 = nn.BatchNorm(
      use_running_average=not self.training, param_dtype=self.dtype
    )(z3)
    z3 = nn.relu(z3)
    z3_pool = nn.max_pool(z3, window_shape=(2, 2), strides=(2, 2))

    z4 = nn.Conv(
      self.features * 8,
      kernel_size=(self.kernel_size, self.kernel_size),
      param_dtype=self.dtype
    )(z3_pool)
    z4 = nn.relu(z4)
    z4 = nn.Conv(
      self.features * 8,
      kernel_size=(self.kernel_size, self.kernel_size),
      param_dtype=self.dtype
    )(z4)
    z4 = nn.BatchNorm(
      use_running_average=not self.training, param_dtype=self.dtype
    )(z4)
    z4 = nn.relu(z4)
    z4_dropout = nn.Dropout(0.5, deterministic=False)(z4)
    z4_pool = nn.max_pool(z4_dropout, window_shape=(2, 2), strides=(2, 2))

    z5 = nn.Conv(
      self.features * 16,
      kernel_size=(self.kernel_size, self.kernel_size),
      param_dtype=self.dtype
    )(z4_pool)
    z5 = nn.relu(z5)
    z5 = nn.Conv(
      self.features * 16,
      kernel_size=(self.kernel_size, self.kernel_size),
      param_dtype=self.dtype
    )(z5)
    z5 = nn.BatchNorm(
      use_running_average=not self.training, param_dtype=self.dtype
    )(z5)
    z5 = nn.relu(z5)
    z5_dropout = nn.Dropout(0.5, deterministic=False)(z5)

    return z1, z2, z3, z4_dropout, z5_dropout


class Decoder2D(nn.Module):
  features: int = 2
  output_features: int = 2
  kernel_size: int = 3
  dtype: str = jnp.float64
  training: bool = True

  @nn.compact
  def __call__(self, z1, z2, z3, z4_dropout, z5_dropout):
    z6_up = jax.image.resize(
      z5_dropout,
      shape=(
        z5_dropout.shape[0], z5_dropout.shape[1] * 2, z5_dropout.shape[2] * 2,
        z5_dropout.shape[3]
      ),
      method='nearest'
    )
    z6 = nn.Conv(self.features * 8, kernel_size=(2, 2),
                 param_dtype=self.dtype)(z6_up)
    z6 = nn.relu(z6)
    z6 = jnp.concatenate([z4_dropout, z6], axis=3)
    z6 = nn.Conv(
      self.features * 8,
      kernel_size=(self.kernel_size, self.kernel_size),
      param_dtype=self.dtype
    )(z6)
    z6 = nn.relu(z6)
    z6 = nn.Conv(
      self.features * 8,
      kernel_size=(self.kernel_size, self.kernel_size),
      param_dtype=self.dtype
    )(z6)
    z6 = nn.BatchNorm(
      use_running_average=not self.training, param_dtype=self.dtype
    )(z6)
    z6 = nn.relu(z6)

    z7_up = jax.image.resize(
      z6,
      shape=(z6.shape[0], z6.shape[1] * 2, z6.shape[2] * 2, z6.shape[3]),
      method='nearest'
    )
    z7 = nn.Conv(self.features * 4, kernel_size=(2, 2),
                 param_dtype=self.dtype)(z7_up)
    z7 = nn.relu(z7)
    z7 = jnp.concatenate([z3, z7], axis=3)
    z7 = nn.Conv(
      self.features * 4,
      kernel_size=(self.kernel_size, self.kernel_size),
      param_dtype=self.dtype
    )(z7)
    z7 = nn.relu(z7)
    z7 = nn.Conv(
      self.features * 4,
      kernel_size=(self.kernel_size, self.kernel_size),
      param_dtype=self.dtype
    )(z7)
    z7 = nn.BatchNorm(
      use_running_average=not self.training, param_dtype=self.dtype
    )(z7)
    z7 = nn.relu(z7)

    z8_up = jax.image.resize(
      z7,
      shape=(z7.shape[0], z7.shape[1] * 2, z7.shape[2] * 2, z7.shape[3]),
      method='nearest'
    )
    z8 = nn.Conv(self.features * 2, kernel_size=(2, 2),
                 param_dtype=self.dtype)(z8_up)
    z8 = nn.relu(z8)
    z8 = jnp.concatenate([z2, z8], axis=3)
    z8 = nn.Conv(
      self.features * 2,
      kernel_size=(self.kernel_size, self.kernel_size),
      param_dtype=self.dtype
    )(z8)
    z8 = nn.relu(z8)
    z8 = nn.Conv(
      self.features * 2,
      kernel_size=(self.kernel_size, self.kernel_size),
      param_dtype=self.dtype
    )(z8)
    z8 = nn.BatchNorm(
      use_running_average=not self.training, param_dtype=self.dtype
    )(z8)
    z8 = nn.relu(z8)

    z9_up = jax.image.resize(
      z8,
      shape=(z8.shape[0], z8.shape[1] * 2, z8.shape[2] * 2, z8.shape[3]),
      method='nearest'
    )
    z9 = nn.Conv(self.features, kernel_size=(2, 2),
                 param_dtype=self.dtype)(z9_up)
    z9 = nn.relu(z9)
    z9 = jnp.concatenate([z1, z9], axis=3)
    z9 = nn.Conv(
      self.features,
      kernel_size=(self.kernel_size, self.kernel_size),
      param_dtype=self.dtype
    )(z9)
    z9 = nn.relu(z9)
    z9 = nn.Conv(
      self.features,
      kernel_size=(self.kernel_size, self.kernel_size),
      param_dtype=self.dtype
    )(z9)
    z9 = nn.BatchNorm(
      use_running_average=not self.training, param_dtype=self.dtype
    )(z9)
    z9 = nn.relu(z9)

    y = nn.Conv(
      self.output_features, kernel_size=(1, 1), param_dtype=self.dtype
    )(z9)

    return y


class UNet(nn.Module):
  input_features: int = 2
  output_features: int = 2
  DIM: int = 2
  kernel_size: int = 3  # New parameter for kernel size
  dtype: str = jnp.float64
  training: bool = True

  @nn.compact
  def __call__(self, x):
    if self.DIM == 2:
      z1, z2, z3, z4_dropout, z5_dropout = Encoder2D(
        self.input_features * 4, self.kernel_size, self.dtype, self.training
      )(x)
      y = Decoder2D(
        self.input_features * 4, self.output_features, self.kernel_size,
        self.dtype, self.training
      )(z1, z2, z3, z4_dropout, z5_dropout)
    elif self.DIM == 1:
      z1, z2, z3, z4, z5 = Encoder1D(
        self.input_features * 8,
        kernel_size=self.kernel_size,
        dtype=self.dtype,
        training=self.training
      )(x)
      y = Decoder1D(
        self.input_features * 8,
        self.output_features,
        kernel_size=self.kernel_size,
        dtype=self.dtype,
        training=self.training
      )(z1, z2, z3, z4, z5)
      # y = nn.softplus(y)

    return y


"""Transformer model definitions."""
''' https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/JAX/tutorial6/Transformers_and_MHAttention.html '''

class Transformer1D(nn.Module):
  num_layers: int
  input_dim: int  # input feature dimension
  output_dim: int  # output feature dimension
  d_model: int  # dimension of the model
  num_heads: int
  dim_feedforward: int
  dropout_prob: float
  max_len: int = 2048 

  def setup(self):
    self.input_proj = nn.Dense(self.d_model)
    self.positional_encoding = PositionalEncoding(d_model=self.d_model, max_len=self.max_len)
    self.layers = [EncoderBlock(self.d_model, self.num_heads, self.dim_feedforward, self.dropout_prob) for _ in range(self.num_layers)]
    self.output_proj = nn.Dense(self.output_dim)

  def __call__(self, x, mask=None, train=True): #full self attention
    x = self.input_proj(x)  # Project input to model dimension
    x = self.positional_encoding(x)  # Add positional encoding
    for l in self.layers:
      x = l(x, mask=mask, train=train)
    x = self.output_proj(x)  # Project back to output dimension
    return x

#A single transformer block. Has MultiHeadAttention and a FeedForward layer (MLP)

#Feedforward: linear --> ReLU --> linear --> add --> LayerNorm
class EncoderBlock(nn.Module):
  input_dim: int  #input dimension equals output dimension because of residual connections
  num_heads: int
  dim_feedforward: int
  dropout_prob: float  # no longer used

  def setup(self):
    #Attention layer
    self.self_attn = MultiHeadAttention(embed_dim=self.input_dim, num_heads=self.num_heads)

    #2 Layer MLP
    self.linear1 = nn.Dense(self.dim_feedforward)
    self.linear2 = nn.Dense(self.input_dim)

    #Layers to apply in between the main layers
    self.norm1 = nn.LayerNorm()
    self.norm2 = nn.LayerNorm()

  def __call__(self, x, mask=None, train=True):
    #Attention part
    attn_out, _ = self.self_attn(x, mask=mask)
    x = x + attn_out
    x = self.norm1(x)

    #Feedforward block
    linear_out = nn.relu(self.linear1(x))
    linear_out = self.linear2(linear_out)
    x = x + linear_out
    x = self.norm2(x)

    return x

class MultiHeadAttention(nn.Module):
  embed_dim: int #Output dimension
  num_heads: int #number of parallel heads (h)

  def setup(self):
    #Stack all weight matrices 1...h and W^Q, W^K, W^V together for efficiency
    self.qkv_proj = nn.Dense(self.embed_dim * 3,
                            kernel_init=nn.initializers.xavier_uniform(), #Weights with Xavier uniform init
                            bias_init=nn.initializers.zeros
                            )  #Biases with zeros init
    self.o_proj = nn.Dense(self.embed_dim,
                            kernel_init=nn.initializers.xavier_uniform(),
                            bias_init=nn.initializers.zeros)
                    
  def __call__(self, x, mask=None):
    batch_size, seq_length, embed_dim = x.shape
    if mask is not None:
      mask = expand_mask(mask)
    qkv = self.qkv_proj(x)

    #Separate Q,K,V from linear output
    qkv = qkv.reshape(batch_size, seq_length, self.num_heads, -1)
    qkv = qkv.transpose(0, 2, 1, 3)  # (batch_size, num_heads, seq_length, dims)
    q, k, v = jnp.split(qkv, 3, axis=-1)

    #Determine value outputs
    values, attention = scaled_dot_product(q, k, v, mask=mask)
    values = values.transpose(0, 2, 1, 3)  # (batch_size, seq_length, num_heads, dims)
    values = values.reshape(batch_size, seq_length, embed_dim)
    o = self.o_proj(values)

    return o, attention

class PositionalEncoding(nn.Module):
  d_model: int #Hidden dimensionality of the input
  max_len: int = 2048 #Maximum length of the input sequence - needed???

  def setup(self):
    #Create a matrix of shape (max_len, d_model) with positional encodings
    pe = jnp.zeros((self.max_len, self.d_model))
    position = jnp.arange(0, self.max_len, dtype=jnp.float32)[:, None]
    div_term = jnp.exp(jnp.arange(0, self.d_model, 2) * (-jnp.log(10000.0) / self.d_model))
    pe = pe.at[:, 0::2].set(jnp.sin(position * div_term))
    pe = pe.at[:, 1::2].set(jnp.cos(position * div_term))
    pe = pe[None]
    self.pe = jax.device_put(pe)

  def __call__(self, x):
    x = x + self.pe[:, :x.shape[1]]
    return x

def scaled_dot_product(q, k, v, mask=None):
  d_k = q.shape[-1]
  attn_logits = jnp.matmul(q, jnp.swapaxes(k, -2, -1))
  attn_logits = attn_logits / jnp.sqrt(d_k)
  if mask is not None:
    attn_logits = jnp.where(mask == 0, -1e9, attn_logits)
  attention = nn.softmax(attn_logits, axis=-1)
  values = jnp.matmul(attention, v)
  return values, attention

