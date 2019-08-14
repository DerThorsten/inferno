import torch.nn as nn
import sys
import functools
from ..initializers import (
    OrthogonalWeightsZeroBias,
    KaimingNormalWeightsZeroBias,
    SELUWeightsZeroBias,
)
from ..initializers import Initializer
from .activations import SELU
from ...utils.exceptions import assert_, ShapeError
from ...utils.partial_cls import register_partial_cls

__all__ = [
    "ConvActivation",
    "ConvELU2D",
    "ConvELU3D",
    "ConvSigmoid2D",
    "ConvSigmoid3D",
    "DeconvELU2D",
    "DeconvELU3D",
    "StridedConvELU2D",
    "StridedConvELU3D",
    "DilatedConvELU2D",
    "DilatedConvELU3D",
    "Conv2D",
    "Conv3D",
    "BNReLUConv2D",
    "BNReLUConv3D",
    "BNReLUDepthwiseConv2D",
    "ConvSELU2D",
    "ConvSELU3D",
    "ConvReLU2D",
    "ConvReLU3D",
    "BNReLUDilatedConv2D",
    "DilatedConv2D",
    "GlobalConv2D",
]
_all = __all__

register_partial_cls_here = functools.partial(register_partial_cls, module=__name__)


class ConvActivation(nn.Module):
    """Convolutional layer with 'SAME' padding by default followed by an activation."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        dim,
        activation,
        stride=1,
        dilation=1,
        groups=None,
        depthwise=False,
        bias=True,
        deconv=False,
        initialization=None,
        valid_conv=False,
    ):
        super(ConvActivation, self).__init__()
        # Validate dim
        assert_(
            dim in [1, 2, 3],
            "`dim` must be one of [1, 2, 3], got {}.".format(dim),
            ShapeError,
        )
        self.dim = dim
        # Check if depthwise
        if depthwise:
            assert_(
                in_channels == out_channels,
                "For depthwise convolutions, number of input channels (given: {}) "
                "must equal the number of output channels (given {}).".format(
                    in_channels, out_channels
                ),
                ValueError,
            )
            assert_(
                groups is None or groups == in_channels,
                "For depthwise convolutions, groups (given: {}) must "
                "equal the number of channels (given: {}).".format(groups, in_channels),
            )
            groups = in_channels
        else:
            groups = 1 if groups is None else groups
        self.depthwise = depthwise
        if valid_conv:
            self.conv = getattr(nn, "Conv{}d".format(self.dim))(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                groups=groups,
                bias=bias,
            )
        elif not deconv:
            # Get padding
            padding = self.get_padding(kernel_size, dilation)
            self.conv = getattr(nn, "Conv{}d".format(self.dim))(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding,
                stride=stride,
                dilation=dilation,
                groups=groups,
                bias=bias,
            )
        else:
            self.conv = getattr(nn, "ConvTranspose{}d".format(self.dim))(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                groups=groups,
                bias=bias,
            )
        if initialization is None:
            pass
        elif isinstance(initialization, Initializer):
            self.conv.apply(initialization)
        else:
            raise NotImplementedError

        if isinstance(activation, str):
            self.activation = getattr(nn, activation)()
        elif isinstance(activation, nn.Module):
            self.activation = activation
        elif activation is None:
            self.activation = None
        else:
            raise NotImplementedError

    def forward(self, input):
        conved = self.conv(input)
        if self.activation is not None:
            activated = self.activation(conved)
        else:
            # No activation
            activated = conved
        return activated

    def _pair_or_triplet(self, object_):
        if isinstance(object_, (list, tuple)):
            assert len(object_) == self.dim
            return object_
        else:
            object_ = [object_] * self.dim
            return object_

    def _get_padding(self, _kernel_size, _dilation):
        assert isinstance(_kernel_size, int)
        assert isinstance(_dilation, int)
        assert _kernel_size % 2 == 1
        return ((_kernel_size - 1) // 2) * _dilation

    def get_padding(self, kernel_size, dilation):
        kernel_size = self._pair_or_triplet(kernel_size)
        dilation = self._pair_or_triplet(dilation)
        padding = [
            self._get_padding(_kernel_size, _dilation)
            for _kernel_size, _dilation in zip(kernel_size, dilation)
        ]
        return tuple(padding)




def _register_conv_cls(prefix,  fix=None, default=None):
    if fix is None:
        fix = {}
    if default is None:
        default = {}

    # simple conv activation
    activations = ["ReLU", "ELU", "Sigmoid", "SELU", ""]
    init_map = {
        "ReLU": KaimingNormalWeightsZeroBias,
        "SELU": SELUWeightsZeroBias
    }
    for activation_str in activations:
        cls_name = cls_name = "{}{}".format(prefix,activation_str)
        initialization_cls = init_map.get(activation_str, OrthogonalWeightsZeroBias)
        if activation_str == "":
            activation = None
            _fix = {**fix}
            _default = {'activation':None}
        elif activation_str == "SELU":
            if hasattr(nn, "SELU"):
                # Pytorch 0.2: Use built in SELU
                activation = nn.SELU(inplace=True)
            else:
                # Pytorch < 0.1.12: Use handmade SELU
                activation = SELU()
            _fix={**fix, 'activation':activation}
            _default = {**default}
        else:
            activation = activation_str
            _fix={**fix, 'activation':activation}
            _default = {**default}

        register_partial_cls_here(ConvActivation, cls_name,
            fix=_fix,
            default={**_default, 'initialization':initialization_cls()}
        )
        for dim in [1, 2, 3]:
            cls_name = "{}{}{}D".format(prefix,activation_str, dim)
            register_partial_cls_here(ConvActivation, cls_name,
                fix={**_fix, 'dim':dim},
                default={**_default, 'initialization':initialization_cls()}
            )

_register_conv_cls("Conv")
_register_conv_cls("ValidConv",  fix=dict(valid_conv=True))
_register_conv_cls("Deconv", fix=dict(deconv=True), default=dict(kernel_size=2, stride=2))
_register_conv_cls("StridedConv",  fix=dict(stride=2))
_register_conv_cls("DilatedConv",  fix=dict(dilation=2))

del _register_conv_cls



# noinspection PyUnresolvedReferences
class _BNReLUSomeConv(object):
    def forward(self, input):
        normed = self.batchnorm(input)
        activated = self.activation(normed)
        conved = self.conv(activated)
        return conved


class BNReLUConv2D(_BNReLUSomeConv, ConvActivation):
    """
    2D BN-ReLU-Conv layer with 'SAME' padding and He weight initialization.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(BNReLUConv2D, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dim=2,
            stride=stride,
            activation=nn.ReLU(inplace=True),
            initialization=KaimingNormalWeightsZeroBias(0),
        )
        self.batchnorm = nn.BatchNorm2d(in_channels)


class BNReLUDilatedConv2D(_BNReLUSomeConv, ConvActivation):
    """
    2D dilated convolutional layer with 'SAME' padding, Batch norm,  Relu and He
    weight initialization.
    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation=2):
        super(BNReLUDilatedConv2D, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            dim=2,
            activation=nn.ReLU(inplace=True),
            initialization=KaimingNormalWeightsZeroBias(0),
        )
        self.batchnorm = nn.BatchNorm2d(in_channels)


class BNReLUConv3D(_BNReLUSomeConv, ConvActivation):
    """
    3D BN-ReLU-Conv layer with 'SAME' padding and He weight initialization.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(BNReLUConv3D, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dim=3,
            stride=stride,
            activation=nn.ReLU(inplace=True),
            initialization=KaimingNormalWeightsZeroBias(0),
        )
        self.batchnorm = nn.BatchNorm3d(in_channels)


class BNReLUDeconv2D(_BNReLUSomeConv, ConvActivation):
    """
    2D BN-ReLU-Deconv layer with He weight initialization and (default) stride 2.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=2):
        super(BNReLUDeconv2D, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dim=2,
            stride=stride,
            deconv=True,
            activation=nn.ReLU(inplace=True),
            initialization=KaimingNormalWeightsZeroBias(0),
        )
        self.batchnorm = nn.BatchNorm2d(in_channels)


class BNReLUDeconv3D(_BNReLUSomeConv, ConvActivation):
    """
    3D BN-ReLU-Deconv layer with He weight initialization and (default) stride 2.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=2):
        super(BNReLUDeconv3D, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dim=3,
            stride=stride,
            deconv=True,
            activation=nn.ReLU(inplace=True),
            initialization=KaimingNormalWeightsZeroBias(0),
        )
        self.batchnorm = nn.BatchNorm2d(in_channels)


class BNReLUDepthwiseConv2D(_BNReLUSomeConv, ConvActivation):
    """
    2D BN-ReLU-Conv layer with 'SAME' padding, He weight initialization and depthwise convolution.
    Note that depthwise convolutions require `in_channels == out_channels`.
    """

    def __init__(self, in_channels, out_channels, kernel_size):
        # We know that in_channels == out_channels, but we also want a consistent API.
        # As a compromise, we allow that out_channels be None or 'auto'.
        out_channels = in_channels if out_channels in [None, "auto"] else out_channels
        super(BNReLUDepthwiseConv2D, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dim=2,
            depthwise=True,
            activation=nn.ReLU(inplace=True),
            initialization=KaimingNormalWeightsZeroBias(0),
        )
        self.batchnorm = nn.BatchNorm2d(in_channels)



class GlobalConv2D(nn.Module):
    """From https://arxiv.org/pdf/1703.02719.pdf
    Main idea: we can have a bigger kernel size computationally acceptable
    if we separate 2D-conv in 2 1D-convs """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        local_conv_type,
        activation=None,
        use_BN=False,
        **kwargs
    ):
        super(GlobalConv2D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        assert isinstance(kernel_size, (int, list, tuple))
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 2
        self.kwargs = kwargs
        self.conv1a = local_conv_type(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=(kernel_size[0], 1),
            **kwargs
        )
        self.conv1b = local_conv_type(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            kernel_size=(1, kernel_size[1]),
            **kwargs
        )
        self.conv2a = local_conv_type(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=(1, kernel_size[1]),
            **kwargs
        )
        self.conv2b = local_conv_type(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            kernel_size=(kernel_size[0], 1),
            **kwargs
        )
        if use_BN:
            self.batchnorm = nn.BatchNorm2d(self.out_channels)
        else:
            self.batchnorm = None
        self.activation = activation

    def forward(self, input_):
        out1 = self.conv1a(input_)
        out1 = self.conv1b(out1)
        out2 = self.conv2a(input_)
        out2 = self.conv2b(out2)
        out = out1.add(1, out2)
        if self.activation is not None:
            out = self.activation(out)
        if self.batchnorm is not None:
            out = self.batchnorm(out)
        return out
