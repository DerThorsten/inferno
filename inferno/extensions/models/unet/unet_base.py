import torch
from torch import nn

import enum


class UnetParts(enum.Enum):
    START = 0
    CONV_DOWN = 1
    BRIDE = 2
    DOWNSAMPLE = 3
    CONV_BOTTOM =4
    UPSAMPLE = 5
    COMBINE = 6
    CONV_UP = 7
    END = 8

class BetterModuleDict(nn.ModuleDict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __setitem__(self, x, mod):
        if not isinstance(x, str):
            x = str(x)
        super().__setitem__(x, mod)

    def __getitem__(self, x):
        if not isinstance(x, str):
            x = str(x)
        return super().__getitem__(x)


R"""

THE UNET TOPOLOGY

[start]-[dconv0]----------- [bri0]-----------------[comb0]--[uconv0]-[end]  
         \                                           |
          \                                          |
          [ds0]                                     [us0]
            \                                        /
             \                                      /
             [dconv1]-------[bri1]--------[comb1]--[uconv]  
               \                              |
                \                             |
               [ds1]                         [us1]
                  \                           /   
                   \                         /
                  [dconv2]--[bri2]--[comb2]--[uconv2]  
                     \                  | 
                      \                 |         
                     [ds2]             [us2]
                        \              /            
                         \            /
                          \          /
                           \[bottom]/
"""
# parts: down bottom up bride
class UNetBase(nn.Module):
    
    parts = ['conv_down', 'conv_up', 
            'ds', 'us', 
            'bottom', 'bridge',
            'start', 'end']

    def __init__(self, dim, depth, in_channels):
        super().__init__()

        self.dim = dim
        self.depth = depth
        self.in_channels = in_channels

        # store modules
        self.ops = BetterModuleDict()

        # should be uses as side out
        self.output_keys_set = set()
        self.output_keys = []

        # help
        def add_op(key, op_and_is_side_out):
            op, is_side_out = op_and_is_side_out
            self.ops[key] = op
            if is_side_out or key is 'end':
              self.output_keys_set.add(key)
              self.output_keys.append(key)

        
        # start block
        key = 'start'
        out_channels = self.get_num_channels(key, None)
        add_op(key, self.start_op_factory(in_channels=in_channels,
                                     out_channels=out_channels))
        in_channels = out_channels

        # conv ops, bridge, and downsample ops
        in_channels = self.get_num_channels('start', index=None)
        for i in range(self.depth):

            # conv
            key = ('conv_down', i)
            out_channels = self.get_num_channels(*key)
            add_op(key, self.conv_down_op_factory(in_channels=in_channels,
                                         out_channels=out_channels,
                                         index=i))

            # bridge
            key = ('bridge', i)
            bridge_out_channels = self.get_num_channels(*key)
            add_op(key, self.bridge_op_factory(in_channels=in_channels,
                                         out_channels=bridge_out_channels,
                                         index=i))

            in_channels = out_channels

            # downsample
            key = ('ds', i)
            out_channels = self.get_num_channels(*key)
            factor = self.get_downsample_factor(index=i)
            add_op(key, self.downsample_op_factory(factor=factor,
                                             in_channels=in_channels,
                                             out_channels=out_channels,
                                             index=i))
            in_channels = out_channels

        # bottom
        key = 'bottom'
        out_channels = self.get_num_channels(key,None)
        add_op(key, self.conv_bottom_op_factory(in_channels=in_channels,
                                         out_channels=out_channels))
        in_channels = out_channels

        # upsample, combine and conv 
        for i in reversed(range(self.depth)):

            # upsample
            key = ('us', i)
            out_channels = self.get_num_channels(*key)
            factor = self.get_downsample_factor(index=i)
            add_op(key, self.upsample_op_factory(factor=factor,
                                             in_channels=in_channels,
                                             out_channels=out_channels,
                                             index=i))
            in_channels = out_channels

            # combine
            key = ('combine', i)
            bridge_key = ('bridge', i)
            out_channels = self.get_num_channels(*key)
            bridge_channels = self.get_num_channels(*bridge_key)
            add_op(key, self.combine_op_factory(in_channels_horizonatal=bridge_channels,
                                             in_channels_down=in_channels,
                                             out_channels=out_channels,
                                             index=i))
            in_channels = out_channels

            # conv
            key = ('conv_up', i)
            out_channels = self.get_num_channels(*key)
            add_op(key, self.conv_down_op_factory(in_channels=in_channels,
                                         out_channels=out_channels,
                                         index=i))
            in_channels = out_channels

        # end block
        key = 'end'
        out_channels = self.get_num_channels(key,None)
        add_op(key, self.end_op_factory(in_channels=in_channels,
                                     out_channels=out_channels))

        # reverse 
        self.output_keys =list(reversed(self.output_keys))
    
    def forward(self, x):

        # remember bridge results
        bridges = dict()

        # side outs
        outputs = []

        # helper to add to side outs
        def handle_side_out(out, key):
          if key in self.output_keys_set:
            outputs.append(out)


        # start
        x = self.ops['start'](x)

        for i in range(self.depth):

            # conv
            conv_res = self.ops['conv_down', i](x)
            handle_side_out(conv_res, ('conv_down', i))

            # bridge
            bridge_res = self.ops['bridge', i](conv_res) 
            handle_side_out(bridge_res, ('bridge', i))
            bridges['bridge', i] = bridge_res
            
            
            x = self.ops['ds', i](conv_res)
            handle_side_out(bridge_res, ('bridge', i))
            
            # move asserts
            for d in range(self.dim):
              assert x.size(2+d) * self.get_downsample_factor(i) == conv_res.size(2+d),"{} vs {}".format(x.size(), conv_res.size())

        # bottom
        x = self.ops['bottom'](x)
        handle_side_out(x, 'bottom')

        for i in reversed(range(self.depth)):

            # upsample
            x = self.ops['us', i](x)
            handle_side_out(x, ('us', i))

            # combine
            bridge = bridges['bridge', i]
            x = self.ops['combine', i](bridge, x)
            handle_side_out(x, ('combine', i))

            # conv
            x = self.ops['conv_up', i](x)
            handle_side_out(x, ('conv_up', i))

        # end 
        x = self.ops['end'](x)
        handle_side_out(x, 'end')

        # since 'end' is always added to side
        # outs this means we have not side out
        # just the regular out
        if len(outputs) == 1:
          return x
        else:
          assert len(outputs) == len(self.output_keys)
          return tuple(reversed(outputs))


        
    def get_num_channels(self, part, index):
        raise NotImplementedError()

    def get_downsample_factor(self, index):
        raise NotImplementedError()

    def start_op_factory(self, in_channels, out_channels):
        raise NotImplementedError()

    def end_op_factory(self, in_channels, out_channels):
        raise NotImplementedError()

    def combine_op_factory(self, in_channels_horizonatal, in_channels_down, out_channels, index):
        raise NotImplementedError()

    def downsample_op_factory(self, factor, in_channels, out_channels, index):
        raise NotImplementedError()

    def upsample_op_factory(self, factor, in_channels, out_channels, index):
        raise NotImplementedError()

    def conv_down_op_factory(self, in_channels, out_channels, index):
        raise NotImplementedError()

    def conv_bottom_op_factory(self, in_channels, out_channels):
        raise NotImplementedError()

    def bridge_op_factory(self, in_channels, out_channels, index):
        raise NotImplementedError()

    def conv_up_factory(self, in_channels, out_channels, index):
        raise NotImplementedError()
