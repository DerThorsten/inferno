import torch.nn as nn

from . unet_simple_base import UNetSimpleBase
from ...layers.reshape import Concatenate
from ...layers.convolutional import ConvELUND, DeconvELUND 



class UNet(UNetSimpleBase):
    def __init__(self, dim, in_channels, out_channels, depth, initial_features, gain):

        super().__init__(dim=dim, in_channels=in_channels, 
                         out_channels=out_channels, depth=depth, 
                         initial_features=initial_features, gain=gain)


    def start_op_factory(self, in_channels, out_channels):
        return ConvELUND(in_channels, out_channels, 3, dim=self.dim), False

    def end_op_factory(self, in_channels, out_channels):
        return ConvELUND(in_channels, out_channels, 1, dim=self.dim), False

        

    def conv_down_op_factory(self, in_channels, out_channels, index):
        conv = nn.Sequential(ConvELUND(in_channels, out_channels, 3, dim=self.dim),
                             ConvELUND(out_channels, out_channels, 3, dim=self.dim))
        return conv, False

    def conv_bottom_op_factory(self, in_channels, out_channels):
        conv = nn.Sequential(ConvELUND(in_channels, out_channels, 3, dim=self.dim),
                             ConvELUND(out_channels, out_channels, 3, dim=self.dim))
        return conv, False
    def conv_up_op_factory(self, in_channels, out_channels, index):
        rconv = nn.Sequential(ConvELUND(in_channels, out_channels, 3, dim=self.dim),
                             ConvELUND(out_channels, out_channels, 3, dim=self.dim))
        return conv, False


    def downsample_op_factory(self, factor, in_channels, out_channels, index):
        return ConvELUND(in_channels, out_channels, 3, stride=factor, dim=self.dim), False

    def upsample_op_factory(self, factor, in_channels, out_channels, index):
        return DeconvELUND(in_channels=in_channels, out_channels=out_channels, 
                stride=factor, dim=self.dim), False

