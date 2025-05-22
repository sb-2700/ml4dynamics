"""
UNet model (http://arxiv.org/abs/1505.04597).
Implementation taken from
https://github.com/google-research/scenic/blob/main/scenic/projects/baselines/unet.py

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import os
import sys
from pathlib import Path

ROOT_PATH = str(Path(__file__).parent.parent)
sys.path.append(ROOT_PATH)

import functools
from typing import Tuple

import flax.linen as nn
import jax.numpy as jnp
import ml_collections

import ml4dynamics.models.nn_ops as nn_ops

Conv3x3 = functools.partial(nn.Conv, kernel_size=(3, 3))


class DeConv3x3(nn.Module):
  """Deconvolution layer for upscaling.

  Attributes:
    features: Num convolutional features.
    padding: Type of padding: 'SAME' or 'VALID'.
    use_batch_norm: Whether to use batchnorm at the end or not.
  """

  features: int
  padding: str = "SAME"
  use_batch_norm: bool = True

  @nn.compact
  def __call__(self, x: jnp.ndarray, *, train: bool = True) -> jnp.ndarray:
    if self.padding == 'SAME':
      padding = ((1, 2), (1, 2))
    elif self.padding == 'VALID':
      padding = ((0, 1), (0, 1))
    elif self.padding == 'CIRCULAR':
      padding = 'CIRCULAR'
    else:
      raise ValueError(f'Unkonwn padding: {self.padding}')
    x = nn.Conv(
      features=self.features,
      kernel_size=(3, 3),
      input_dilation=(2, 2),
      padding=padding
    )(x)
    if self.use_batch_norm:
      x = nn.BatchNorm(use_running_average=not train)(x)
    return x


class ConvRelu2(nn.Module):
  """Two unpadded convolutions & relus.

  Attributes:
    features: Num convolutional features.
    padding: Type of padding: 'SAME' or 'VALID'.
    use_batch_norm: Whether to use batchnorm at the end or not.
  """

  features: int
  padding: str = "SAME"
  use_batch_norm: bool = True

  @nn.compact
  def __call__(self, x: jnp.ndarray, *, train: bool = True) -> jnp.ndarray:
    x = Conv3x3(features=self.features, name='conv1', padding=self.padding)(x)
    if self.use_batch_norm:
      x = nn.BatchNorm(use_running_average=not train)(x)
    x = nn.relu(x)
    x = Conv3x3(features=self.features, name='conv2', padding=self.padding)(x)
    if self.use_batch_norm:
      x = nn.BatchNorm(use_running_average=not train)(x)
    x = nn.relu(x)
    return x


class DownsampleBlock(nn.Module):
  """Two unpadded convolutions & downsample 2x.

  Attributes:
    features: Num convolutional features.
    padding: Type of padding: 'SAME' or 'VALID'.
    use_batch_norm: Whether to use batchnorm at the end or not.
  """

  features: int
  padding: str = "SAME"
  use_batch_norm: bool = True

  @nn.compact
  def __call__(self, x: jnp.ndarray, *, train: bool = True) -> jnp.ndarray:
    residual = x = ConvRelu2(
      features=self.features,
      padding=self.padding,
      use_batch_norm=self.use_batch_norm
    )(x, train=train)
    x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

    return x, residual


class BottleneckBlock(nn.Module):
  """Two unpadded convolutions, dropout & deconvolution.

  Attributes:
    features: Num convolutional features.
    padding: Type of padding: 'SAME' or 'VALID'.
    use_batch_norm: Whether to use batchnorm at the end or not.
  """

  features: int
  padding: str = "SAME"
  use_batch_norm: bool = True

  @nn.compact
  def __call__(self, x: jnp.ndarray, *, train: bool = True) -> jnp.ndarray:
    x = ConvRelu2(
      features=self.features,
      padding=self.padding,
      use_batch_norm=self.use_batch_norm
    )(x, train=train)
    x = DeConv3x3(
      features=self.features // 2,
      name='deconv',
      padding=self.padding,
      use_batch_norm=self.use_batch_norm
    )(x, train=train)

    return x


class UpsampleBlock(nn.Module):
  """Two unpadded convolutions and upsample.

  Attributes:
    features: Num convolutional features.
    padding: Type of padding: 'SAME' or 'VALID'.
    use_batch_norm: Whether to use batchnorm at the end or not.
  """

  features: int
  padding: str = "SAME"
  use_batch_norm: bool = True

  @nn.compact
  def __call__(
    self, x: jnp.ndarray, residual, *, train: bool = True
  ) -> jnp.ndarray:
    if residual is not None:
      #print(residual.shape)
      #print(x.shape)
      #print(nn_ops.central_crop(residual, x.shape).shape)
      x = jnp.concatenate([x, nn_ops.central_crop(residual, x.shape)], axis=-1)
    x = ConvRelu2(
      features=self.features,
      padding=self.padding,
      use_batch_norm=self.use_batch_norm
    )(x, train=train)
    x = DeConv3x3(
      features=self.features // 2,
      name='deconv',
      padding=self.padding,
      use_batch_norm=self.use_batch_norm
    )(x, train=train)

    return x


class OutputBlock(nn.Module):
  """Two unpadded convolutions followed by linear FC.


  Attributes:
    features: Num convolutional features.
    num_classes: Number of classes.
    padding: Type of padding: 'SAME' or 'VALID'.
    use_batch_norm: Whether to use batchnorm at the end or not.
  """

  features: int
  num_classes: int
  padding: str = 'SAME'
  use_batch_norm: bool = True

  @nn.compact
  def __call__(self, x: jnp.ndarray, *, train: bool) -> jnp.ndarray:
    x = ConvRelu2(
      self.features, padding=self.padding, use_batch_norm=self.use_batch_norm
    )(x, train=train)
    x = nn.Conv(features=self.num_classes, kernel_size=(1, 1),
                name='conv1x1')(x)
    if self.use_batch_norm:
      x = nn.BatchNorm(use_running_average=not train)(x)
    return x


class UNet(nn.Module):
  """U-Net from http://arxiv.org/abs/1505.04597.

  Based on:
  https://github.com/NVIDIA/DeepLearningExamples/blob/master/TensorFlow2/Segmentation/UNet_Medical/model/unet.py
  Note that the default configuration `config.padding="VALID"` does only work
  with images that have a certain minimum size (e.g. 128x128 is too small).

  Attributes:
    num_classes: Number of classes.
    block_size: Sequence of feature sizes used in UNet blocks.
    padding: Type of padding.
    use_batch_norm: Whether to use batchnorm or not.
  """

  num_classes: int
  block_size: Tuple[int, ...] = (64, 128, 256, 512)
  padding: str = "SAME"
  use_batch_norm: bool = True

  @nn.compact
  def __call__(
    self,
    x: jnp.ndarray,
    *,
    train: bool = True,
    debug: bool = False
  ) -> jnp.ndarray:
    """Applies the UNet model."""
    del debug
    skip_connections = []
    for i, features in enumerate(self.block_size):
      x, residual = DownsampleBlock(
        features=features,
        padding=self.padding,
        use_batch_norm=self.use_batch_norm,
        name=f'0_down_{i}'
      )(x, train=train)
      skip_connections.append(residual)
    x = BottleneckBlock(
      features=2 * self.block_size[-1],
      padding=self.padding,
      use_batch_norm=self.use_batch_norm,
      name='1_bottleneck'
    )(x, train=train)

    *upscaling_features, final_features = self.block_size[::-1]
    for i, features in enumerate(upscaling_features):
      x = UpsampleBlock(
        features=features,
        padding=self.padding,
        use_batch_norm=self.use_batch_norm,
        name=f'2_up_{i}'
      )(x, skip_connections.pop(), train=train)

    x = OutputBlock(
      features=final_features,
      num_classes=self.num_classes,
      padding=self.padding,
      use_batch_norm=self.use_batch_norm,
      name='output_projection'
    )(x, train=train)
    return x
