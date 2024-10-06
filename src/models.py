import jax
import jax.numpy as jnp
import numpy as np
import torch
import torch.nn.functional as F
from flax import linen as fnn
from torch import nn


##########################################################################
# jax implementation
##########################################################################
class Encoder(fnn.Module):
    features: int = 64
    training: bool = True

    @fnn.compact
    def __call__(self, x):
        z1 = fnn.Conv(self.features, kernel_size=(3, 3))(x)
        z1 = fnn.relu(z1)
        z1 = fnn.Conv(self.features, kernel_size=(3, 3))(z1)
        z1 = fnn.BatchNorm(use_running_average=not self.training)(z1)
        z1 = fnn.relu(z1)
        z1_pool = fnn.max_pool(z1, window_shape=(2, 2), strides=(2, 2))

        z2 = fnn.Conv(self.features * 2, kernel_size=(3, 3))(z1_pool)
        z2 = fnn.relu(z2)
        z2 = fnn.Conv(self.features * 2, kernel_size=(3, 3))(z2)
        z2 = fnn.BatchNorm(use_running_average=not self.training)(z2)
        z2 = fnn.relu(z2)
        z2_pool = fnn.max_pool(z2, window_shape=(2, 2), strides=(2, 2))

        z3 = fnn.Conv(self.features * 4, kernel_size=(3, 3))(z2_pool)
        z3 = fnn.relu(z3)
        z3 = fnn.Conv(self.features * 4, kernel_size=(3, 3))(z3)
        z3 = fnn.BatchNorm(use_running_average=not self.training)(z3)
        z3 = fnn.relu(z3)
        z3_pool = fnn.max_pool(z3, window_shape=(2, 2), strides=(2, 2))

        z4 = fnn.Conv(self.features * 8, kernel_size=(3, 3))(z3_pool)
        z4 = fnn.relu(z4)
        z4 = fnn.Conv(self.features * 8, kernel_size=(3, 3))(z4)
        z4 = fnn.BatchNorm(use_running_average=not self.training)(z4)
        z4 = fnn.relu(z4)
        z4_dropout = fnn.Dropout(0.5, deterministic=False)(z4)
        z4_pool = fnn.max_pool(z4_dropout, window_shape=(2, 2), strides=(2, 2))

        z5 = fnn.Conv(self.features * 16, kernel_size=(3, 3))(z4_pool)
        z5 = fnn.relu(z5)
        z5 = fnn.Conv(self.features * 16, kernel_size=(3, 3))(z5)
        z5 = fnn.BatchNorm(use_running_average=not self.training)(z5)
        z5 = fnn.relu(z5)
        z5_dropout = fnn.Dropout(0.5, deterministic=False)(z5)

        return z1, z2, z3, z4_dropout, z5_dropout


class Decoder(fnn.Module):
    features: int = 64
    training: bool = True

    @fnn.compact
    def __call__(self, z1, z2, z3, z4_dropout, z5_dropout):
        z6_up = jax.image.resize(z5_dropout, shape=(z5_dropout.shape[0], z5_dropout.shape[1] * 2, z5_dropout.shape[2] * 2, z5_dropout.shape[3]),
                                 method='nearest')
        z6 = fnn.Conv(self.features * 8, kernel_size=(2, 2))(z6_up)
        z6 = fnn.relu(z6)
        z6 = jnp.concatenate([z4_dropout, z6], axis=3)
        z6 = fnn.Conv(self.features * 8, kernel_size=(3, 3))(z6)
        z6 = fnn.relu(z6)
        z6 = fnn.Conv(self.features * 8, kernel_size=(3, 3))(z6)
        z6 = fnn.BatchNorm(use_running_average=not self.training)(z6)
        z6 = fnn.relu(z6)

        z7_up = jax.image.resize(z6, shape=(z6.shape[0], z6.shape[1] * 2, z6.shape[2] * 2, z6.shape[3]),
                                 method='nearest')
        z7 = fnn.Conv(self.features * 4, kernel_size=(2, 2))(z7_up)
        z7 = fnn.relu(z7)
        z7 = jnp.concatenate([z3, z7], axis=3)
        z7 = fnn.Conv(self.features * 4, kernel_size=(3, 3))(z7)
        z7 = fnn.relu(z7)
        z7 = fnn.Conv(self.features * 4, kernel_size=(3, 3))(z7)
        z7 = fnn.BatchNorm(use_running_average=not self.training)(z7)
        z7 = fnn.relu(z7)

        z8_up = jax.image.resize(z7, shape=(z7.shape[0], z7.shape[1] * 2, z7.shape[2] * 2, z7.shape[3]),
                                 method='nearest')
        z8 = fnn.Conv(self.features * 2, kernel_size=(2, 2))(z8_up)
        z8 = fnn.relu(z8)
        z8 = jnp.concatenate([z2, z8], axis=3)
        z8 = fnn.Conv(self.features * 2, kernel_size=(3, 3))(z8)
        z8 = fnn.relu(z8)
        z8 = fnn.Conv(self.features * 2, kernel_size=(3, 3))(z8)
        z8 = fnn.BatchNorm(use_running_average=not self.training)(z8)
        z8 = fnn.relu(z8)

        z9_up = jax.image.resize(z8, shape=(z8.shape[0], z8.shape[1] * 2, z8.shape[2] * 2, z8.shape[3]),
                                 method='nearest')
        z9 = fnn.Conv(self.features, kernel_size=(2, 2))(z9_up)
        z9 = fnn.relu(z9)
        z9 = jnp.concatenate([z1, z9], axis=3)
        z9 = fnn.Conv(self.features, kernel_size=(3, 3))(z9)
        z9 = fnn.relu(z9)
        z9 = fnn.Conv(self.features, kernel_size=(3, 3))(z9)
        z9 = fnn.BatchNorm(use_running_average=not self.training)(z9)
        z9 = fnn.relu(z9)

        y = fnn.Conv(1, kernel_size=(1, 1))(z9)
        y = fnn.sigmoid(y)

        return y


class UNet(fnn.Module):
    features: int = 64
    training: bool = True

    @fnn.compact
    def __call__(self, x):
        z1, z2, z3, z4_dropout, z5_dropout = Encoder(self.training)(x)
        y = Decoder(self.training)(z1, z2, z3, z4_dropout, z5_dropout)

        return y



##########################################################################
# torch implementation
##########################################################################


def weights_init(m):
  classname = m.__class__.__name__
  if classname.find('Conv') != -1:
    m.weight.data.normal_(0.0, 0.02)
  elif classname.find('BatchNorm') != -1:
    m.weight.data.normal_(1.0, 0.02)
    m.bias.data.fill_(0)


def blockUNet(
  in_c,
  out_c,
  name,
  transposed=False,
  bn=True,
  relu=True,
  size=4,
  pad=1,
  dropout=0.
):
  block = nn.Sequential()
  if relu:
    block.add_module('%s_relu' % name, nn.ReLU(inplace=True))
  else:
    block.add_module('%s_leakyrelu' % name, nn.LeakyReLU(0.2, inplace=True))
  if not transposed:
    block.add_module(
      '%s_conv' % name,
      nn.Conv2d(
        in_c, out_c, kernel_size=size, stride=2, padding=pad, bias=True
      )
    )
  else:
    block.add_module(
      '%s_upsam' % name,
      nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    )  # Note: old default was nearest neighbor
    # reduce kernel size by one for the upsampling (ie decoder part)
    block.add_module(
      '%s_tconv' % name,
      nn.Conv2d(
        in_c, out_c, kernel_size=(size - 1), stride=1, padding=pad, bias=True
      )
    )
  if bn:
    block.add_module('%s_bn' % name, nn.BatchNorm2d(out_c))
  if dropout > 0.:
    block.add_module('%s_dropout' % name, nn.Dropout2d(dropout, inplace=True))
  return block


class EDNet(nn.Module):

  def __init__(self, channel_array=[2, 4, 4, 8, 8, 16], dropout=0.):
    super(EDNet, self).__init__()

    self.layer1 = nn.Sequential()
    self.layer1.add_module(
      'layer1_conv',
      nn.Conv2d(channel_array[0], channel_array[1], 4, 2, 1, bias=True)
    )
    self.layer2 = blockUNet(
      channel_array[1],
      channel_array[2],
      'layer2',
      transposed=False,
      bn=True,
      relu=False,
      dropout=dropout
    )
    self.layer3 = blockUNet(
      channel_array[2],
      channel_array[3],
      'layer3',
      transposed=False,
      bn=True,
      relu=False,
      dropout=dropout
    )
    self.layer4 = blockUNet(
      channel_array[3],
      channel_array[4],
      'layer4',
      transposed=False,
      bn=True,
      relu=False,
      dropout=dropout,
      size=4
    )
    self.layer5 = blockUNet(
      channel_array[4],
      channel_array[5],
      'layer5',
      transposed=False,
      bn=True,
      relu=False,
      dropout=dropout,
      size=2,
      pad=0
    )
    self.layer6 = blockUNet(
      channel_array[5],
      channel_array[5],
      'layer6',
      transposed=False,
      bn=False,
      relu=False,
      dropout=dropout,
      size=2,
      pad=0
    )

    # note, kernel size is internally reduced by one now
    self.dlayer6 = blockUNet(
      channel_array[5],
      channel_array[5],
      'dlayer6',
      transposed=True,
      bn=True,
      relu=True,
      dropout=dropout,
      size=2,
      pad=0
    )
    self.dlayer5 = blockUNet(
      channel_array[5],
      channel_array[4],
      'dlayer5',
      transposed=True,
      bn=True,
      relu=True,
      dropout=dropout,
      size=2,
      pad=0
    )
    self.dlayer4 = blockUNet(
      channel_array[4],
      channel_array[3],
      'dlayer4',
      transposed=True,
      bn=True,
      relu=True,
      dropout=dropout
    )
    self.dlayer3 = blockUNet(
      channel_array[3],
      channel_array[2],
      'dlayer3',
      transposed=True,
      bn=True,
      relu=True,
      dropout=dropout
    )
    self.dlayer2 = blockUNet(
      channel_array[2],
      channel_array[1],
      'dlayer2',
      transposed=True,
      bn=True,
      relu=True,
      dropout=dropout
    )

    self.dlayer1 = nn.Sequential()
    self.dlayer1.add_module('dlayer1_relu', nn.ReLU(inplace=True))
    self.dlayer1.add_module(
      'dlayer1_tconv',
      nn.ConvTranspose2d(
        channel_array[1], channel_array[0], 4, 2, 1, bias=True
      )
    )

  def forward(self, x):
    out1 = self.layer1(x)
    out2 = self.layer2(out1)
    out3 = self.layer3(out2)
    out4 = self.layer4(out3)
    out5 = self.layer5(out4)
    # our current method is just decreasing the # of layer by 1
    #out6 = self.layer6(out5)
    #dout6 = self.dlayer6(out6)
    dout5 = self.dlayer5(out5)
    dout4 = self.dlayer4(dout5)
    dout3 = self.dlayer3(dout4)
    dout2 = self.dlayer2(dout3)
    dout1 = self.dlayer1(dout2)
    return dout1


class UNet(nn.Module):

  def __init__(self, channel_array=[2, 4, 8, 16, 32, 32, 2], dropout=0.):
    super(UNet, self).__init__()

    self.layer1 = nn.Sequential()
    self.layer1.add_module(
      'layer1_conv',
      nn.Conv2d(channel_array[0], channel_array[1], 4, 2, 1, bias=True)
    )
    self.layer2 = blockUNet(
      channel_array[1],
      channel_array[2],
      'layer2',
      transposed=False,
      bn=True,
      relu=False,
      dropout=dropout
    )
    self.layer3 = blockUNet(
      channel_array[2],
      channel_array[3],
      'layer3',
      transposed=False,
      bn=True,
      relu=False,
      dropout=dropout
    )
    self.layer4 = blockUNet(
      channel_array[3],
      channel_array[4],
      'layer4',
      transposed=False,
      bn=True,
      relu=False,
      dropout=dropout,
      size=4
    )
    self.layer5 = blockUNet(
      channel_array[4],
      channel_array[5],
      'layer5',
      transposed=False,
      bn=True,
      relu=False,
      dropout=dropout,
      size=2,
      pad=0
    )

    # horizontal layer, which serves as skip connnection
    self.hlayer4 = blockUNet(
      channel_array[4],
      channel_array[4],
      'hlayer5',
      transposed=True,
      bn=True,
      relu=True,
      dropout=dropout,
      size=2,
      pad=0
    )
    self.hlayer3 = blockUNet(
      channel_array[3],
      channel_array[3],
      'hlayer4',
      transposed=True,
      bn=True,
      relu=True,
      dropout=dropout
    )
    self.hlayer2 = blockUNet(
      channel_array[2],
      channel_array[2],
      'hlayer3',
      transposed=True,
      bn=True,
      relu=True,
      dropout=dropout
    )
    self.hlayer1 = blockUNet(
      channel_array[1],
      channel_array[1],
      'hlayer2',
      transposed=True,
      bn=True,
      relu=True,
      dropout=dropout
    )
    self.hlayer0 = blockUNet(
      channel_array[0],
      channel_array[0],
      'hlayer1',
      transposed=True,
      bn=True,
      relu=True,
      dropout=dropout
    )

    self.dlayer5 = blockUNet(
      channel_array[5],
      channel_array[4],
      'dlayer5',
      transposed=True,
      bn=True,
      relu=True,
      dropout=dropout,
      size=2,
      pad=0
    )
    self.dlayer4 = blockUNet(
      channel_array[4] * 2,
      channel_array[3],
      'dlayer4',
      transposed=True,
      bn=True,
      relu=True,
      dropout=dropout
    )
    self.dlayer3 = blockUNet(
      channel_array[3] * 2,
      channel_array[2],
      'dlayer3',
      transposed=True,
      bn=True,
      relu=True,
      dropout=dropout
    )
    self.dlayer2 = blockUNet(
      channel_array[2] * 2,
      channel_array[1],
      'dlayer2',
      transposed=True,
      bn=True,
      relu=True,
      dropout=dropout
    )
    self.dlayer1 = nn.Sequential()
    self.dlayer1.add_module('dlayer1_relu', nn.ReLU(inplace=True))
    self.dlayer1.add_module(
      'dlayer1_tconv',
      nn.ConvTranspose2d(
        channel_array[1] * 2, channel_array[6], 4, 2, 1, bias=True
      )
    )

  def forward(self, x):
    # normalization procedure, guaranteeing that our model is homogeneous
    scale = torch.max(torch.abs(x))
    x = x / scale
    out1 = self.layer1(x)
    out2 = self.layer2(out1)
    out3 = self.layer3(out2)
    out4 = self.layer4(out3)
    out5 = self.layer5(out4)
    dout5 = self.dlayer5(out5)
    out4_ = out4
    # currently we still use the skip connection, which is not reasonable
    # as the Poisson solver should not be the identity mapping
    #out4_ = self.hlayer4(out4)
    dout5_out4 = torch.cat([dout5, out4_], 1)
    dout4 = self.dlayer4(dout5_out4)
    out3_ = out3
    #out3_ = self.hlayer3(out3)
    dout4_out3 = torch.cat([dout4, out3_], 1)
    dout3 = self.dlayer3(dout4_out3)
    out2_ = out2
    #out2_ = self.hlayer2(out2)
    dout3_out2 = torch.cat([dout3, out2_], 1)
    dout2 = self.dlayer2(dout3_out2)
    out1_ = out1
    #out1_ = self.hlayer1(out1)
    dout2_out1 = torch.cat([dout2, out1_], 1)
    dout1 = self.dlayer1(dout2_out1)
    dout1 = dout1 * scale
    return dout1
