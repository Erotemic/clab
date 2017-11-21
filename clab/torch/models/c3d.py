import torch.nn as nn
from functools import partial
import torch  # NOQA
from .torch.models.output_shape_for import OutputShapeFor
from collections import OrderedDict


class Conv3DBlock(nn.Module):
    """
    >>> block = Conv3DBlock(in_channels=64, out_channels=128, n_conv=2)
    >>> block.output_shape_for([1, 3, 16, 112, 112])
    (1, 128, 8, 56, 56)

    >>> block = Conv3DBlock(in_channels=64, out_channels=128, conv_kernel=4, n_conv=1)
    >>> block.output_shape_for([1, 3, 16, 112, 112])
    (1, 128, 7, 55, 55)

    >>> block = Conv3DBlock(in_channels=64, out_channels=128, conv_kernel=4, n_conv=3)
    >>> block.output_shape_for([1, 3, 16, 112, 112])
    (1, 128, 6, 54, 54)

    """
    def __init__(self, in_channels, out_channels, n_conv=1,
                 conv_kernel=(3, 3, 3), conv_padding=1,
                 pool_kernel=(2, 2, 2), pool_stride=(2, 2, 2)):
        super(Conv3DBlock, self).__init__()
        nonlinearity = partial(nn.LeakyReLU, negative_slope=1e-2,
                               inplace=False)

        assert n_conv >= 1

        named_layers = []

        # First convolution uses input_channels
        named_layers += [
            ('conv0', nn.Conv3d(in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=conv_kernel,
                                padding=conv_padding)),
            ('nonlin0', nonlinearity()),
        ]

        # The remainder use output_channels
        for ix in range(1, n_conv):
            suff = str(ix)
            named_layers += [
                ('conv' + suff, nn.Conv3d(in_channels=out_channels,
                                          out_channels=out_channels,
                                          kernel_size=conv_kernel,
                                          padding=conv_padding)),
                ('nonlin' + suff, nonlinearity()),
            ]

        named_layers += [
            ('pool', nn.MaxPool3d(kernel_size=pool_kernel, stride=pool_stride)),
        ]

        self.sequence = nn.Sequential(OrderedDict(named_layers))

    def output_shape_for(self, input_shape):
        return OutputShapeFor(self.sequence)(input_shape)

    def forward(self, inputs):
        return self.sequence(inputs)


class FCBlock(nn.Module):
    def __init__(self, n_inputs, out_channels):
        super(FCBlock, self).__init__()
        nonlinearity = partial(nn.LeakyReLU, negative_slope=1e-2,
                               inplace=False)
        self.sequence = nn.Sequential(OrderedDict([
            ('fc', nn.Linear(n_inputs, out_channels)),
            ('nonlin', nonlinearity()),
        ]))

    def output_shape_for(self, input_shape):
        return OutputShapeFor(self.sequence)(input_shape)

    def forward(self, inputs):
        return self.sequence(inputs)


class C3D(nn.Module):
    """
    The C3D network as described in [1].

    References:
        [1] Tran, Du, et al. "Learning spatiotemporal features with 3d convolutional networks."
        Proceedings of the IEEE international conference on computer vision. 2015.

    Notes:
        * According to the findings in 2D ConvNet [37], small receptive fields
        of 3 × 3 convolution kernels with deeper architectures yield best
        results.  Hence, for our architecture search study we fix the spatial
        receptive field to 3 × 3 and vary only the temporal depth of the 3D
        convolution kernels

    References:
        https://github.com/DavideA/c3d-pytorch/blob/master/C3D_model.py

    Example:
        >>> B, C, D, H, W = 1, 3, 16, 112, 112
        >>> # default input shape is
        >>> input_shape = [B, C, D, H, W]
        >>> inputs = torch.autograd.Variable(torch.randn(input_shape)).cpu()
        >>> #input_shape = [None, C, D, H, W]
        >>> self = C3D(input_shape)
        >>> outputs = self(inputs)
    """

    def __init__(self, input_shape):
        """
        """
        super(C3D, self).__init__()
        # nonlinearity = partial(nn.ReLU)
        # kernels are specified in D, H, W

        feats = [64, 128, 256, 512, 512]
        conv_blocks = nn.Sequential(OrderedDict([
            ('block1', Conv3DBlock(in_channels=3, out_channels=feats[0], n_conv=1,
                                   pool_kernel=(1, 2, 2), pool_stride=(1, 2, 2))),
            ('block2', Conv3DBlock(in_channels=feats[0], out_channels=feats[1], n_conv=1)),
            ('block3', Conv3DBlock(in_channels=feats[1], out_channels=feats[2], n_conv=2)),
            ('block4', Conv3DBlock(in_channels=feats[2], out_channels=feats[3], n_conv=2)),
            ('block5', Conv3DBlock(in_channels=feats[3], out_channels=feats[4], n_conv=2)),
        ]))
        output_shape = OutputShapeFor(conv_blocks)(input_shape)
        print('output_shape = {!r}'.format(output_shape))
        import numpy as np

        self.input_shape = input_shape
        self.conv_blocks = conv_blocks
        self.n_conv_output = int(np.prod(output_shape[1:]))
        self.block6 = FCBlock(self.n_conv_output, 4096)
        self.block7 = FCBlock(4096, 4096)

        self.softmax = nn.Softmax(dim=1)

    def debug(self, inputs):
        c1, c2, c3, c4, c5 = self.conv_blocks.children()
        h1 = c1(inputs)
        assert h1.shape == c1.output_shape_for(inputs.shape)
        h2 = c2(h1)
        assert h2.shape == c2.output_shape_for(h1.shape)
        h3 = c3(h2)
        assert h3.shape == c3.output_shape_for(h2.shape)
        h4 = c4(h3)
        assert h4.shape == c4.output_shape_for(h3.shape)
        h5 = c5(h4)
        assert h5.shape == c5.output_shape_for(h4.shape)

    def forward(self, inputs):
        h = self.conv_blocks(inputs)

        h = h.view(-1, self.n_conv_output)
        h = self.block6(h)
        h = self.block7(h)

        probs = self.softmax(h)
        return probs
