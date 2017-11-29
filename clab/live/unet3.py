import torch
from clab.torch.models import unet
import ubelt as ub
import torchvision  # NOQA
import torch.nn as nn
import math
import torch  # NOQA
import torch.nn.functional as F
from clab import util
from clab.torch.models import mixin
from clab.torch.models.output_shape_for import OutputShapeFor
import numpy as np


def default_nonlinearity():
    # nonlinearity = functools.partial(nn.ReLU, inplace=False)
    return nn.LeakyReLU(inplace=True)


class DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate=0):
        util.super2(DenseLayer, self).__init__()
        self.bn_size = bn_size
        self.growth_rate = growth_rate
        self.n_feat_internal = bn_size * growth_rate
        self.n_feat_out = num_input_features + growth_rate

        self.add_module('norm.1', nn.BatchNorm2d(num_input_features)),
        self.add_module('noli.1', nn.ReLU(inplace=True)),
        self.add_module('conv.1', nn.Conv2d(num_input_features, bn_size *
                        growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm.2', nn.BatchNorm2d(self.n_feat_internal)),
        self.add_module('noli.2', nn.ReLU(inplace=True)),
        self.add_module('conv.2', nn.Conv2d(self.n_feat_internal, growth_rate,
                        kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = util.super2(DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)

    def output_shape_for(self, input_shape):
        N1, C1, W1, H1 = input_shape
        output_shape = (N1, self.n_feat_out, W1, H1)
        return output_shape


class DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size=4, growth_rate=32, drop_rate=0):
        util.super2(DenseBlock, self).__init__()
        self.num_layers = num_layers
        for i in range(num_layers):
            layer = DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)
        self.n_feat_out = layer.n_feat_out


class Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        util.super2(Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('noli', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class PadToAgree(nn.Module):
    def __init__(self):
        util.super2(PadToAgree, self).__init__()

    def padding(self, input_shape1, input_shape2):
        """
        Example:
            >>> self = PadToAgree()
            >>> input_shape1 = (1, 32, 37, 52)
            >>> input_shape2 = (1, 32, 28, 44)
            >>> self.padding(input1_shape, input2_shape)
            [-4, -4, -5, -4]
        """
        have_w, have_h = input_shape1[-2:]
        want_w, want_h = input_shape2[-2:]

        half_offw = (want_w - have_w) / 2
        half_offh = (want_h - have_h) / 2
        # padding = 2 * [offw // 2, offh // 2]

        padding = [
            # Padding starts from the final dimension and then move backwards.
            math.floor(half_offh),
            math.ceil(half_offh),
            math.floor(half_offw),
            math.ceil(half_offw),
        ]
        return padding

    def forward(self, inputs1, inputs2):
        input_shape1 = inputs1.size()
        input_shape2 = inputs2.size()
        padding = self.padding(input_shape1, input_shape2)

        if np.all(padding == 0):
            outputs1 = inputs1
        else:
            outputs1 = F.pad(inputs1, padding)
        return outputs1

    def output_shape_for(self, input_shape1, input_shape2):
        N1, C1, W1, H1 = input_shape1
        N2, C2, W2, H2 = input_shape2
        output_shape = (N1, C1, W2, H2)
        return output_shape


class DenseUNetUp(nn.Module):
    """
        Ignore:
            in_size = 64
            out_size = 32
            down = DenseBlock(3, 64, 4)
            self = DenseUNetUp(in_size1=64, in_size2=160)
            num_layers = 2
    """
    def __init__(self, in_size1, in_size2, compress=.5, num_layers=2,
                 growth_rate=4, is_deconv=True):
        util.super2(DenseUNetUp, self).__init__()
        out_size2 = int(compress * in_size2)
        in_size = in_size1 + out_size2
        bneck_size = int(in_size1 * compress)
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_size2, out_size2, kernel_size=2,
                                         stride=2, bias=False)
        else:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.pad = PadToAgree()
        self.conv = DenseBlock(num_layers, in_size, growth_rate=growth_rate,
                                bn_size=4)
        self.bottleneck = nn.Conv2d(self.conv.n_feat_out, bneck_size,
                                    kernel_size=1, stride=1, bias=False)
        self.n_feat_out = bneck_size

    def output_shape_for(self, input1_shape, input2_shape):
        """

        Example:
            >>> self = DenseUNetUp(128, 256)
            >>> input1_shape = [4, 128, 24, 24]
            >>> input2_shape = [4, 256, 8, 8]
            >>> output_shape = self.output_shape_for(input1_shape, input2_shape)
            (4, 64, 24, 24)
            >>> inputs1 = torch.autograd.Variable(torch.rand(input1_shape))
            >>> inputs2 = torch.autograd.Variable(torch.rand(input2_shape))
            >>> assert self.forward(inputs1, inputs2).shape == output_shape
        """
        output2_shape = OutputShapeFor(self.up)(input2_shape)
        output2_shape = OutputShapeFor(self.pad)(output2_shape, input1_shape)

        # Taking the easy way out and padding the upsampled layer instead of
        # cropping the down layer

        # output1_shape = OutputShapeFor(self.pad)(input1_shape, output2_shape)
        # cat_shape     = OutputShapeFor(torch.cat)([output1_shape, output2_shape], 1)

        cat_shape     = OutputShapeFor(torch.cat)([input1_shape, output2_shape], 1)
        conv_shape  = OutputShapeFor(self.conv)(cat_shape)
        output_shape  = OutputShapeFor(self.bottleneck)(conv_shape)
        return output_shape

    def forward(self, inputs1, inputs2):
        """
        inputs1 = (37 x 52)
        inputs2 = (14 x 22) -> up -> (28 x 44)
        self.up = self.up_concat4.up

        want_w, want_h = (28, 44)  # outputs2
        have_w, have_h = (37, 52)  # inputs1

        offw = -9
        offh = -8

        padding [-5, -4, -4, -4]
        """
        outputs2 = self.up(inputs2)
        # Oposite padding to ensure output size = input size
        # (might be better to accept loss of border resolution)
        outputs2 = self.pad(outputs2, inputs1)
        # outputs1 = self.pad(inputs1, outputs2)
        # outputs_cat = torch.cat([outputs1, outputs2], 1)

        outputs_cat = torch.cat([inputs1, outputs2], 1)
        return self.bottleneck(self.conv(outputs_cat))


class DenseUNet(nn.Module, mixin.NetMixin):
    """
    Note input shapes should be a power of 2.

    In this case there will be a ~188 pixel difference between input and output
    dims, so the input should be mirrored with

    Example:
        >>> from clab.live.unet3 import *
        >>> from torch.autograd import Variable
        >>> B, C, W, H = (4, 3, 480, 360)
        >>> input_shape = (B, C, W, H)

        >>> self = DenseUNet(in_channels=C)
        >>> print('output shapes')
        >>> print(ub.repr2(self.output_shapes(input_shape), nl=1))
        >>> self.number_of_parameters()
        >>> inputs = Variable(torch.rand(B, C, W, H), volatile=True)
        >>> outputs = self(inputs)

        >>> from clab.torch.models import unet
        >>> model_unet = unet.UNet()
        >>> self.number_of_parameters()

        >>> B, C, W, H = (4, 5, 480, 360)
        >>> input_shape = (B, C, W, H)
        >>> n_classes = 11
        >>> inputs = Variable(torch.rand(B, C, W, H))
        >>> labels = Variable((torch.rand(B, W, H) * n_classes).long())
        >>> self = DenseUNet(in_channels=C, n_classes=n_classes)
        >>> outputs = self.forward(inputs)
        >>> print('inputs.size() = {!r}'.format(inputs.size()))
        >>> print('outputs.size() = {!r}'.format(outputs.size()))
        >>> print(np.array(inputs.size()) - np.array(outputs.size()))

    """
    def __init__(self, n_classes=21, n_alt_classes=3, in_channels=3):
        util.super2(DenseUNet, self).__init__()
        self.in_channels = in_channels

        n_feat0 = 64
        from torch import nn
        features = nn.Sequential(ub.odict([
            ('conv0', nn.Conv2d(in_channels, n_feat0, kernel_size=7, stride=1,
                                padding=3,
                                bias=False)),
            ('norm0', nn.BatchNorm2d(n_feat0)),
            ('noli0', default_nonlinearity()),
            # ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        block_config = [3, 4, 4, 4, 4]
        bn_size = 4
        compress = .5
        growth_rate = 32

        n_feat = n_feat0

        # downsampling
        down = []
        densekw = dict(bn_size=bn_size, growth_rate=growth_rate)

        for i, num_layers in enumerate(block_config[0:-1]):
            down.append(('denseblock%d' % (i + 1), DenseBlock(num_layers=num_layers, num_input_features=n_feat, **densekw)))
            n_feat = n_feat + num_layers * growth_rate
            n_feat_compress = int(n_feat * compress)
            down.append(('transition%d' % (i + 1), Transition(num_input_features=n_feat, num_output_features=n_feat_compress)))
            n_feat = n_feat_compress

        # for key, value in down:
        #     self.add_module(key, value)
        self.denseblock1 = down[0][1]
        self.transition1 = down[1][1]
        self.denseblock2 = down[2][1]
        self.transition2 = down[3][1]
        self.denseblock3 = down[4][1]
        self.transition3 = down[5][1]
        self.denseblock4 = down[6][1]
        self.transition4 = down[7][1]

        num_layers = block_config[-1]
        center5 = DenseBlock(num_layers=num_layers, num_input_features=n_feat, **densekw)
        n_feat = n_feat + num_layers * growth_rate

        center5.n_feat_out

        up_concat4 = DenseUNetUp(self.denseblock4.n_feat_out, center5.n_feat_out)
        up_concat3 = DenseUNetUp(self.denseblock3.n_feat_out, up_concat4.n_feat_out)
        up_concat2 = DenseUNetUp(self.denseblock2.n_feat_out, up_concat3.n_feat_out)
        up_concat1 = DenseUNetUp(self.denseblock1.n_feat_out, up_concat2.n_feat_out)

        self.features = features
        self.center5 = center5

        self.up_concat4 = up_concat4
        self.up_concat3 = up_concat3
        self.up_concat2 = up_concat2
        self.up_concat1 = up_concat1

        # final conv (without any concat)
        self.final1 = nn.Conv2d(up_concat1.n_feat_out, n_classes, 1)
        self.final2 = nn.Conv2d(up_concat1.n_feat_out, n_alt_classes, 1)
        self._cache = {}

        self.main_path = [
            'features',
            'denseblock1',
            'transition1',
            'denseblock2',
            'transition2',
            'denseblock3',
            'transition3',
            'denseblock4',
            'transition4',
            'center5',
            'up_concat4',
            'up_concat3',
            'up_concat2',
            'up_concat1',
        ]
        # When a node accepts multiple inputs, we need to specify which order
        # they appear in the signature
        self.other_edges = [
            ('denseblock4', 'up_concat4', {'argx': 0}),
            ('denseblock3', 'up_concat3', {'argx': 0}),
            ('denseblock2', 'up_concat2', {'argx': 0}),
            ('denseblock1', 'up_concat1', {'argx': 0}),
            ('up_concat1', 'final1', {'argx': 0}),
            ('up_concat1', 'final2', {'argx': 0}),
        ]

        # down_net = nn.Sequential(ub.odict(down))
        # self.down_net = down_net

    def output_shapes(self, input_shape):
        graph = self.connectivity()
        output_shapes = ub.odict()
        # prev = None
        import networkx as nx
        for node in list(nx.topological_sort(graph)):
            preds = graph.pred[node]
            if not preds:
                in_shapes = [input_shape]
            else:
                in_shapes = []
                argxs = []
                for k in preds:
                    argxs.append(graph.edges[(k, node)].get('argx', None))
                    in_shapes.append(output_shapes[k])
                def rectify_argxs(argxs):
                    """
                    Ensure the arguments are given in the correct order
                    """
                    given = [a for a in argxs if a is not None]
                    mask = np.array(ub.boolmask(given, len(argxs)))
                    values = np.where(~mask)[0]
                    if len(values) > 0:
                        assert len(values) <= 1
                        missingx = argxs.index(None)
                        argxs[missingx] = values[0]
                    return argxs
                argxs = rectify_argxs(argxs)
                in_shapes = list(ub.take(in_shapes, argxs))

            out_shapes = OutputShapeFor(getattr(self, node))(*in_shapes)
            output_shapes[node] = out_shapes
        return output_shapes

        #     print('* {}'.format(node))
        #     implicit_input_shape = len(preds) == 1 and prev in preds
        #     if not implicit_input_shape:
        #         print('   * in_shapes  = {!r}'.format(in_shapes))
        #     print('   * out_shapes = {!r}'.format(out_shapes))
        #     prev = node
        # print('output_shapes = {}'.format(ub.repr2(output_shapes, nl=1)))

    def connectivity(self):
        import networkx as nx
        graph = nx.DiGraph()
        # Main FCN path
        nx.add_path(graph, self.main_path, )
        graph.add_edges_from(self.other_edges)
        return graph

    def output_shape_for(self, input_shape):
        output_shapes = self.output_shapes(input_shape)
        return (output_shapes['final1'], output_shapes['final2'])

    def forward(self, inputs):
        if isinstance(inputs, (list, tuple)):
            if len(inputs) != 1:
                raise ValueError('Only accepts one input')
            inputs = inputs[0]

        features = self.features(inputs)

        conv1 = self.denseblock1(features)
        maxpool1 = self.transition1(conv1)

        conv2 = self.denseblock2(maxpool1)
        maxpool2 = self.transition2(conv2)

        conv3 = self.denseblock3(maxpool2)
        maxpool3 = self.transition3(conv3)

        conv4 = self.denseblock4(maxpool3)
        maxpool4 = self.transition4(conv4)

        center = self.center5(maxpool4)

        up4 = self.up_concat4(conv4, center)
        up3 = self.up_concat3(conv3, up4)
        up2 = self.up_concat2(conv2, up3)
        up1 = self.up_concat1(conv1, up2)

        final1 = self.final1(up1)
        final2 = self.final2(up1)
        return final1, final2
