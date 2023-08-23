import torch
import numpy as np
import torch.nn.functional as F


from torch import nn


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

        
def blockUNet(in_c, out_c, name, transposed=False, bn=True, relu=True, size=4, pad=1, dropout=0.):
    block = nn.Sequential()
    if relu:
        block.add_module('%s_relu' % name, nn.ReLU(inplace=True))
    else:
        block.add_module('%s_leakyrelu' % name, nn.LeakyReLU(0.2, inplace=True))
    if not transposed:
        block.add_module('%s_conv' % name, nn.Conv2d(in_c, out_c, kernel_size=size, stride=2, padding=pad, bias=True))
    else:
        block.add_module('%s_upsam' % name, nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)) # Note: old default was nearest neighbor
        # reduce kernel size by one for the upsampling (ie decoder part)
        block.add_module('%s_tconv' % name, nn.Conv2d(in_c, out_c, kernel_size=(size-1), stride=1, padding=pad, bias=True))
    if bn:
        block.add_module('%s_bn' % name, nn.BatchNorm2d(out_c))
    if dropout>0.:
        block.add_module('%s_dropout' % name, nn.Dropout2d( dropout, inplace=True))
    return block
    
    
'''class RDNet(nn.Module):
    def __init__(self, channelExponent=2, dropout=0.):
        super(RDNet, self).__init__()
        channels = int(2 ** channelExponent + 0.5)

        self.layer1 = nn.Sequential()
        # layer1: 2*64*64    -->    4*32*32
        self.layer1.add_module('layer1_conv', nn.Conv2d(2, channels, 4, 2, 1, bias=True))

        # layer2: 4*32*32   -->    8*16*16  LeakyReLU + BatchNorm
        self.layer2 = blockUNet(channels  , channels*2, 'layer2', transposed=False, bn=True,  relu=False, dropout=dropout )
        #self.layer2b= blockUNet(channels*2, channels*2, 'layer2b',transposed=False, bn=True,  relu=False, dropout=dropout )
        # layer3: 8*16*16   -->    16*8*8   LeakyReLU + BatchNorm
        self.layer3 = blockUNet(channels*2, channels*4, 'layer3', transposed=False, bn=True,  relu=False, dropout=dropout )
        # layer4: 16*8*8    -->    32*4*4   LeakyReLU + BatchNorm
        self.layer4 = blockUNet(channels*4, channels*8, 'layer4', transposed=False, bn=True,  relu=False, dropout=dropout ,  size=4 ) 
        # layer5: 32*4*4   -->     32*2*2   LeakyReLU + BatchNorm
        self.layer5 = blockUNet(channels*8, channels*8, 'layer5', transposed=False, bn=True,  relu=False, dropout=dropout , size=2,pad=0)
        # layer5: 32*2*2   -->     32*1*1   LeakyReLU + BatchNorm
        self.layer6 = blockUNet(channels*8, channels*8, 'layer6', transposed=False, bn=False, relu=False, dropout=dropout , size=2,pad=0)
     
        # note, kernel size is internally reduced by one now
        self.dlayer6 = blockUNet(channels*8, channels*8, 'dlayer6', transposed=True, bn=True, relu=True, dropout=dropout , size=2,pad=0)
        self.dlayer5 = blockUNet(channels*16,channels*8, 'dlayer5', transposed=True, bn=True, relu=True, dropout=dropout , size=2,pad=0)
        self.dlayer4 = blockUNet(channels*16,channels*4, 'dlayer4', transposed=True, bn=True, relu=True, dropout=dropout ) 
        self.dlayer3 = blockUNet(channels*8, channels*2, 'dlayer3', transposed=True, bn=True, relu=True, dropout=dropout )
        #self.dlayer2b= blockUNet(channels*4, channels*2, 'dlayer2b',transposed=True, bn=True, relu=True, dropout=dropout )
        self.dlayer2 = blockUNet(channels*4, channels  , 'dlayer2', transposed=True, bn=True, relu=True, dropout=dropout )

        self.dlayer1 = nn.Sequential()
        self.dlayer1.add_module('dlayer1_relu', nn.ReLU(inplace=True))
        self.dlayer1.add_module('dlayer1_tconv', nn.ConvTranspose2d(channels*2, 2, 4, 2, 1, bias=True))

    def forward(self, x):
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        #out2b= self.layer2b(out2)
        #out3 = self.layer3(out2b)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)
        out6 = self.layer6(out5)
        dout6 = self.dlayer6(out6)
        dout6_out5 = torch.cat([dout6, out5], 1)
        dout5 = self.dlayer5(dout6_out5)
        dout5_out4 = torch.cat([dout5, out4], 1)
        dout4 = self.dlayer4(dout5_out4)
        dout4_out3 = torch.cat([dout4, out3], 1)
        dout3 = self.dlayer3(dout4_out3)
        #dout3_out2b = torch.cat([dout3, out2b], 1)
        #dout2b = self.dlayer2b(dout3_out2b)
        #dout2b_out2 = torch.cat([dout2b, out2], 1)
        #dout2 = self.dlayer2(dout2b_out2)
        dout3_out2 = torch.cat([dout3, out2], 1)
        dout2 = self.dlayer2(dout3_out2)
        dout2_out1 = torch.cat([dout2, out1], 1)
        dout1 = self.dlayer1(dout2_out1)
        return dout1'''


# class definition of encoder decoder network, it does not possess the skip connection
# but we can try to make it similar to the coding style of UNet
class EDNet(nn.Module):
    def __init__(self, channelExponent=2, dropout=0.):
        super(EDNet, self).__init__()
        channels = int(2 ** channelExponent + 0.5)

        self.layer1 = nn.Sequential()
        # layer1: 2*64*64    -->    4*32*32
        self.layer1.add_module('layer1_conv', nn.Conv2d(2, channels, 4, 2, 1, bias=True))
        # layer2: 4*32*32    -->    4*16*16  LeakyReLU + BatchNorm
        self.layer2 = blockUNet(channels  , channels, 'layer2', transposed=False, bn=True,  relu=False, dropout=dropout )
        #self.layer2b= blockUNet(channels*2, channels*2, 'layer2b',transposed=False, bn=True,  relu=False, dropout=dropout )
        # layer3: 4*16*16    -->    8*8*8   LeakyReLU + BatchNorm
        self.layer3 = blockUNet(channels, channels*2, 'layer3', transposed=False, bn=True,  relu=False, dropout=dropout )
        # layer4: 8*8*8      -->    8*4*4   LeakyReLU + BatchNorm
        self.layer4 = blockUNet(channels*2, channels*2, 'layer4', transposed=False, bn=True,  relu=False, dropout=dropout ,  size=4 ) 
        # layer5: 8*4*4      -->    16*2*2   LeakyReLU + BatchNorm
        self.layer5 = blockUNet(channels*2, channels*4, 'layer5', transposed=False, bn=True,  relu=False, dropout=dropout , size=2,pad=0)
        # layer5: 16*2*2   -->     16*1*1   LeakyReLU + BatchNorm
        self.layer6 = blockUNet(channels*4, channels*4, 'layer6', transposed=False, bn=False, relu=False, dropout=dropout , size=2,pad=0)
     
        # note, kernel size is internally reduced by one now
        self.dlayer6 = blockUNet(channels*4, channels*4, 'dlayer6', transposed=True, bn=True, relu=True, dropout=dropout , size=2,pad=0)
        self.dlayer5 = blockUNet(channels*4, channels*2, 'dlayer5', transposed=True, bn=True, relu=True, dropout=dropout , size=2,pad=0)
        self.dlayer4 = blockUNet(channels*2, channels*2, 'dlayer4', transposed=True, bn=True, relu=True, dropout=dropout ) 
        self.dlayer3 = blockUNet(channels*2, channels  , 'dlayer3', transposed=True, bn=True, relu=True, dropout=dropout )
        #self.dlayer2b= blockUNet(channels*4, channels*2, 'dlayer2b',transposed=True, bn=True, relu=True, dropout=dropout )
        self.dlayer2 = blockUNet(channels, channels  , 'dlayer2', transposed=True, bn=True, relu=True, dropout=dropout )

        self.dlayer1 = nn.Sequential()
        self.dlayer1.add_module('dlayer1_relu', nn.ReLU(inplace=True))
        self.dlayer1.add_module('dlayer1_tconv', nn.ConvTranspose2d(channels, 2, 4, 2, 1, bias=True))

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
    def __init__(self, channel_array=[2,4,8,16,32,64,2], dropout=0.):
        super(UNet, self).__init__()


        self.layer1 = nn.Sequential()
        # layer1: 2*64*64    -->    4*32*32
        #         2*128*32   -->    4*
        self.layer1.add_module('layer1_conv', nn.Conv2d(channel_array[0], channel_array[1], 4, 2, 1, bias=True))
        # layer2: 4*32*32   -->    8*16*16  LeakyReLU + BatchNorm
        self.layer2 = blockUNet(channel_array[1], channel_array[2], 'layer2', transposed=False, bn=True,  relu=False, dropout=dropout )
        #self.layer2b= blockUNet(channels*2, channels*2, 'layer2b',transposed=False, bn=True,  relu=False, dropout=dropout )
        # layer3: 8*16*16   -->    16*8*8   LeakyReLU + BatchNorm
        self.layer3 = blockUNet(channel_array[2], channel_array[3], 'layer3', transposed=False, bn=True,  relu=False, dropout=dropout )
        # layer4: 16*8*8    -->    32*4*4   LeakyReLU + BatchNorm
        self.layer4 = blockUNet(channel_array[3], channel_array[4], 'layer4', transposed=False, bn=True,  relu=False, dropout=dropout ,  size=4 ) 
        # layer5: 32*4*4   -->     32*2*2   LeakyReLU + BatchNorm
        self.layer5 = blockUNet(channel_array[4], channel_array[4], 'layer5', transposed=False, bn=True,  relu=False, dropout=dropout , size=2,pad=0)
        # layer6: 32*2*2   -->     32*1*1   LeakyReLU + BatchNorm
        #self.layer6 = blockUNet(channel_array[4], channel_array[4], 'layer6', transposed=False, bn=False, relu=False, dropout=dropout , size=2,pad=0)
     
        # horizontal layer, which serves as skip connnection
        self.hlayer4 = blockUNet(channel_array[4], channel_array[4], 'dlayer5', transposed=True, bn=True, relu=True, dropout=dropout , size=2,pad=0)
        self.hlayer3 = blockUNet(channel_array[5], channel_array[3], 'dlayer4', transposed=True, bn=True, relu=True, dropout=dropout ) 
        self.hlayer2 = blockUNet(channel_array[4], channel_array[2], 'dlayer3', transposed=True, bn=True, relu=True, dropout=dropout )
        self.hlayer1 = blockUNet(channel_array[4], channel_array[2], 'dlayer3', transposed=True, bn=True, relu=True, dropout=dropout )
        
        # note, kernel size is internally reduced by one now
        #self.dlayer6 = blockUNet(channel_array[4], channel_array[4], 'dlayer6', transposed=True, bn=True, relu=True, dropout=dropout , size=2,pad=0)
        self.dlayer5 = blockUNet(channel_array[4], channel_array[4], 'dlayer5', transposed=True, bn=True, relu=True, dropout=dropout , size=2,pad=0)
        self.dlayer4 = blockUNet(channel_array[5], channel_array[3], 'dlayer4', transposed=True, bn=True, relu=True, dropout=dropout ) 
        self.dlayer3 = blockUNet(channel_array[4], channel_array[2], 'dlayer3', transposed=True, bn=True, relu=True, dropout=dropout )
        #self.dlayer2b= blockUNet(channels*4, channels*2, 'dlayer2b',transposed=True, bn=True, relu=True, dropout=dropout )
        self.dlayer2 = blockUNet(channel_array[3], channel_array[1], 'dlayer2', transposed=True, bn=True, relu=True, dropout=dropout )
        self.dlayer1 = nn.Sequential()
        self.dlayer1.add_module('dlayer1_relu', nn.ReLU(inplace=True))
        self.dlayer1.add_module('dlayer1_tconv', nn.ConvTranspose2d(channel_array[2], channel_array[6], 4, 2, 1, bias=True))

    def forward(self, x):
        # normalization procedure, guaranteeing that our model is homogeneous
        scale = torch.max(torch.abs(x))
        x = x/scale
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