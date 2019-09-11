import torch.nn as nn

from . unet_base import UNetBase
from ...layers.reshape import Concatenate
from ...layers.identity import Identity


class UNetSimpleBase(UNetBase):
    def __init__(self, dim, in_channels, out_channels, depth, initial_features, gain):

        self.dim = dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth
        self.initial_features = initial_features
        self.gain  = gain

        super().__init__(dim=dim, in_channels=in_channels, depth=depth)


    def get_num_channels(self, part, index):
        if part == 'start':
            return self.initial_features
        elif part == 'end':
            return self.out_channels
        elif part in ('conv_down', 'conv_up', 'bridge', 'ds'):
            return self.initial_features * self.gain**(index + 1)
        elif part == 'us':
            return self.initial_features * self.gain**(index + 2)
        elif part == 'bottom':
            return self.initial_features * self.gain**(self.depth + 1)
        elif part == 'combine':
            # concat us and brige
            us = self.initial_features * self.gain**(index + 2)
            bridge = self.initial_features * self.gain**(index + 1)
            return us + bridge
        else:
            raise RuntimeError()


    def get_downsample_factor(self, index):
        return 2

    def combine_op_factory(self, in_channels_horizonatal, in_channels_down, out_channels, index):
        return Concatenate(dim=1), False

    def bridge_op_factory(self, in_channels, out_channels, index):
        return Identity(),False