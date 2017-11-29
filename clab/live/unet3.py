import torch
from clab.torch.models import unet
import torchvision  # NOQA
import torch.nn as nn
import functools
import math
import torch  # NOQA
import torch.nn.functional as F
from clab.torch import nninit
from clab.torch.models.output_shape_for import OutputShapeFor

__all__ = ['DenseUNet']


def default_nonlinearity():
    # nonlinearity = functools.partial(nn.ReLU, inplace=False)
    return nn.LeakyReLU(inplace=True)


class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate=0):
        super(_DenseLayer, self).__init__()
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
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)

    def output_shape_for(self, input_shape):
        N1, C1, W1, H1 = input_shape
        output_shape = (N1, self.n_feat_out, W1, H1)
        return output_shape


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size=4, growth_rate=32, drop_rate=0):
        super(_DenseBlock, self).__init__()
        self.num_layers = num_layers
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)
        self.n_feat_out = layer.n_feat_out


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('noli', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class DenseUNetConv(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm, nonlinearity='relu'):
        super(DenseUNetConv, self).__init__()

        if nonlinearity == 'relu':
            nonlinearity = functools.partial(nn.ReLU, inplace=False)
        elif nonlinearity == 'leaky_relu':
            nonlinearity = functools.partial(nn.LeakyReLU, inplace=False)

        conv2d_1 = nn.Conv2d(in_size, out_size, kernel_size=3, stride=1,
                             padding=0)
        conv2d_2 = nn.Conv2d(out_size, out_size, kernel_size=3, stride=1,
                             padding=0)

        if is_batchnorm:
            self.conv1 = nn.Sequential(conv2d_1, nn.BatchNorm2d(out_size),
                                       nonlinearity(),)
            self.conv2 = nn.Sequential(conv2d_2, nn.BatchNorm2d(out_size),
                                       nonlinearity(),)
        else:
            self.conv1 = nn.Sequential(conv2d_1, nonlinearity(),)
            self.conv2 = nn.Sequential(conv2d_2, nonlinearity(),)
    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs

    def output_shape_for(self, input_shape, math=math):
        shape = OutputShapeFor(self.conv1[0])(input_shape)
        shape = OutputShapeFor(self.conv2[0])(shape)
        output_shape = shape
        return output_shape


class PadToAgree(nn.Module):
    def __init__(self):
        super(PadToAgree, self).__init__()

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
            down = _DenseBlock(3, 64, 4)
            self = DenseUNetUp(in_size1=64, in_size2=160)
            num_layers = 2
    """
    def __init__(self, in_size1, in_size2, compress=.5, num_layers=2,
                 growth_rate=4, is_deconv=True):
        super(DenseUNetUp, self).__init__()
        out_size2 = int(compress * in_size2)
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_size2, out_size2, kernel_size=2, stride=2)
        else:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.pad = PadToAgree()
        in_size = in_size1 + out_size2
        self.conv = _DenseBlock(num_layers, in_size, growth_rate=growth_rate,
                                bn_size=4)
        bneck_size = int(in_size1 * compress)
        self.bottleneck = nn.Conv2d(self.conv.n_feat_out, bneck_size,
                                    kernel_size=1, stride=1, bias=False)
        self.n_feat_out = bneck_size

        # self.conv = UNetConv2(in_size, num_layers, is_batchnorm=False,
        #                       nonlinearity=nonlinearity)

    def output_shape_for(self, input1_shape, input2_shape):
        """

        Example:
            >>> self = UNetUp(256, 128)
            >>> input1_shape = [4, 128, 24, 24]
            >>> input2_shape = [4, 256, 8, 8]
            >>> output_shape = self.output_shape_for(input1_shape, input2_shape)
            >>> output_shape
            (4, 128, 12, 12)
            >>> inputs1 = torch.autograd.Variable(torch.rand(input1_shape))
            >>> inputs2 = torch.autograd.Variable(torch.rand(input2_shape))
            >>> assert self.forward(inputs1, inputs2).shape == output_shape
        """
        output2_shape = OutputShapeFor(self.up)(input2_shape)
        output1_shape = OutputShapeFor(self.pad)(input1_shape, output2_shape)
        cat_shape     = OutputShapeFor(torch.cat)([output1_shape, output2_shape], 1)
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
        outputs1 = self.pad(inputs1, outputs2)
        outputs_cat = torch.cat([outputs1, outputs2], 1)
        return self.bottleneck(self.conv(outputs_cat))


class DenseUNet(nn.Module):
    """
    Note input shapes should be a power of 2.

    In this case there will be a ~188 pixel difference between input and output
    dims, so the input should be mirrored with

    Example:
        >>> from clab.torch.models.unet import *  # NOQA
        >>> from torch.autograd import Variable
        >>> B, C, W, H = (4, 3, 256, 256)
        >>> B, C, W, H = (4, 3, 572, 572)
        >>> n_classes = 11
        >>> inputs = Variable(torch.rand(B, C, W, H))
        >>> labels = Variable((torch.rand(B, W, H) * n_classes).long())
        >>> self = UNet(in_channels=C, n_classes=n_classes)
        >>> outputs = self.forward(inputs)
        >>> print('inputs.size() = {!r}'.format(inputs.size()))
        >>> print('outputs.size() = {!r}'.format(outputs.size()))
        >>> print(np.array(inputs.size()) - np.array(outputs.size()))

    Example:
        >>> from clab.torch.models.unet import *  # NOQA
        >>> from torch.autograd import Variable
        >>> input_shape = (4, 6, 480, 360)
        >>> B, C, W, H = (4, 5, 480, 360)
        >>> n_classes = 11
        >>> in_channels = 3
        >>> inputs = Variable(torch.rand(B, C, W, H))
        >>> labels = Variable((torch.rand(B, W, H) * n_classes).long())
        >>> self = UNet(in_channels=C, n_classes=n_classes)
        >>> outputs = self.forward(inputs)
        >>> print('inputs.size() = {!r}'.format(inputs.size()))
        >>> print('outputs.size() = {!r}'.format(outputs.size()))
        >>> print(np.array(inputs.size()) - np.array(outputs.size()))
    """
    def __init__(self, n_classes=21, is_deconv=True, in_channels=3):
        super(DenseUNet, self).__init__()
        self.in_channels = in_channels

        import ubelt as ub

        n_feat0 = 64
        from torch import nn
        features = nn.Sequential(ub.odict([
            ('conv0', nn.Conv2d(in_channels, n_feat0, kernel_size=7, stride=1, padding=3, bias=False)),
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
            down.append(('denseblock%d' % (i + 1), _DenseBlock(num_layers=num_layers, num_input_features=n_feat, **densekw)))
            n_feat = n_feat + num_layers * growth_rate
            n_feat_compress = int(n_feat * compress)
            down.append(('transition%d' % (i + 1), _Transition(num_input_features=n_feat, num_output_features=n_feat_compress)))
            n_feat = n_feat_compress

        down_net = nn.Sequential(ub.odict(down))

        num_layers = block_config[-1]
        center5 = _DenseBlock(num_layers=num_layers, num_input_features=n_feat, **densekw)
        n_feat = n_feat + num_layers * growth_rate

        center5.n_feat_out

        up_concat4 = DenseUNetUp(down_net.denseblock4.n_feat_out, center5.n_feat_out)
        up_concat3 = DenseUNetUp(down_net.denseblock3.n_feat_out, up_concat4.n_feat_out)
        up_concat2 = DenseUNetUp(down_net.denseblock2.n_feat_out, up_concat3.n_feat_out)
        up_concat1 = DenseUNetUp(down_net.denseblock1.n_feat_out, up_concat2.n_feat_out)

        self.up_concat4 = up_concat4
        self.up_concat3 = up_concat3
        self.up_concat2 = up_concat2
        self.up_concat1 = up_concat1

        feat_shape = OutputShapeFor(features)(input_shape)
        dense_shape = OutputShapeFor(down_net)(feat_shape)

        # downsampling
        # self.conv1 = UNetConv2(self.in_channels, filters[0], self.is_batchnorm)
        # self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        # self.conv2 = UNetConv2(filters[0], filters[1], self.is_batchnorm, self.nonlinearity)
        # self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        # self.conv3 = UNetConv2(filters[1], filters[2], self.is_batchnorm, self.nonlinearity)
        # self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        # self.conv4 = UNetConv2(filters[2], filters[3], self.is_batchnorm, self.nonlinearity)
        # self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        # self.center = UNetConv2(filters[3], filters[4], self.is_batchnorm, self.nonlinearity)

        # upsampling
        # self.up_concat4 = DenseUNetUp(filters[4], filters[3], self.is_deconv)
        # self.up_concat3 = DenseUNetUp(filters[3], filters[2], self.is_deconv)
        # self.up_concat2 = DenseUNetUp(filters[2], filters[1], self.is_deconv)
        # self.up_concat1 = DenseUNetUp(filters[1], filters[0], self.is_deconv)

        # final conv (without any concat)
        self.final = nn.Conv2d(filters[0], n_classes, 1)
        self._cache = {}

    def output_shape_for(self, input_shape):
        # N1, C1, W1, H1 = input_shape
        # output_shape = (N1, self.n_classes, W1, H1)
        shape = input_shape
        shape = conv1 = OutputShapeFor(self.conv1)(shape)
        shape = OutputShapeFor(self.maxpool1)(shape)

        shape = conv2 = OutputShapeFor(self.conv2)(shape)
        shape = OutputShapeFor(self.maxpool2)(shape)

        shape = conv3 = OutputShapeFor(self.conv3)(shape)
        shape = OutputShapeFor(self.maxpool3)(shape)

        shape = conv4 = OutputShapeFor(self.conv4)(shape)
        shape = OutputShapeFor(self.maxpool4)(shape)

        shape = OutputShapeFor(self.center)(shape)

        shape = OutputShapeFor(self.up_concat4)(conv4, shape)
        shape = OutputShapeFor(self.up_concat3)(conv3, shape)
        shape = OutputShapeFor(self.up_concat2)(conv2, shape)
        shape = OutputShapeFor(self.up_concat1)(conv1, shape)

        shape = OutputShapeFor(self.final)(shape)
        output_shape = shape
        return output_shape

    def raw_output_shape_for(self, input_shape):
        # output shape without fancy prepad mirrors and post crops
        shape = conv1 = OutputShapeFor(self.conv1)(input_shape)
        shape = OutputShapeFor(self.maxpool1)(shape)

        shape = conv2 = OutputShapeFor(self.conv2)(shape)
        shape = OutputShapeFor(self.maxpool2)(shape)

        shape = conv3 = OutputShapeFor(self.conv3)(shape)
        shape = OutputShapeFor(self.maxpool3)(shape)

        shape = conv4 = OutputShapeFor(self.conv4)(shape)
        shape = OutputShapeFor(self.maxpool4)(shape)

        shape = OutputShapeFor(self.center)(shape)

        shape = OutputShapeFor(self.up_concat4)(conv4, shape)
        shape = OutputShapeFor(self.up_concat3)(conv3, shape)
        shape = OutputShapeFor(self.up_concat2)(conv2, shape)
        shape = OutputShapeFor(self.up_concat1)(conv1, shape)

        shape = OutputShapeFor(self.final)(shape)
        output_shape = shape
        return output_shape

    def find_padding_and_crop_for(self, input_shape):
        """
        Example:
            >>> from clab.torch.models.unet import *  # NOQA
            >>> from clab.torch.models.unet import nn, math, torch, F, OutputShapeFor
            >>> from torch.autograd import Variable
            >>> B, C, W, H = (4, 3, 572, 572)
            >>> B, C, W, H = (4, 3, 372, 400)
            >>> n_classes = 11
            >>> input_shape = (B, C, W, H)
            >>> self = UNet(in_channels=C, n_classes=n_classes)
            >>> self.raw_output_shape_for(input_shape)
            >>> prepad, postcrop = self.find_padding_and_crop_for(input_shape)
        """
        input_shape = tuple(input_shape)
        if input_shape in self._cache:
            return self._cache[input_shape]

        import sympy as sym
        shape = input_shape

        raw_output = self.raw_output_shape_for(input_shape)
        assert raw_output[2] > 0, 'input is too small'
        assert raw_output[3] > 0, 'input is too small'

        input_shape_ = sym.symbols('N, C, W, H', integer=True, positive=True)
        orig = OutputShapeFor.math
        OutputShapeFor.math = sym
        shape = input_shape_
        # hack OutputShapeFor with sympy to do some symbolic math
        output_shape = self.raw_output_shape_for(shape)
        OutputShapeFor.math = orig

        W1, H1 = input_shape_[-2:]
        W2_raw = output_shape[2]
        H2_raw = output_shape[3]

        padw, padh = sym.symbols('padw, padh', integer=True, positive=True)

        def find_padding(D_in, D_out, pad_in, want):
            """
            Find a padding where
            want = numeric in dimension
            out_dimension = forward(in_dimension + padding)
            """
            D_out_pad = D_out.subs({D_in: D_in + pad_in})

            expr = D_out_pad
            target = want
            fixed = {D_in: want}
            solve_for = pad_in

            fixed_expr = expr.subs(fixed).simplify()
            def func(a1):
                expr_value = float(fixed_expr.subs({solve_for: a1}).evalf())
                return expr_value - target

            def integer_step_linear_zero(func):
                value = 0
                hi, lo = 10000, 0
                while lo <= hi:
                    guess = (lo + hi) // 2
                    result = func(guess)
                    if result < value:
                        lo = guess + 1
                    elif result > value:
                        hi = guess - 1
                    else:
                        break

                # force a positive padding
                while result < value:
                    guess += 1
                    result = func(guess)

                # always choose the lowest level value of guess
                i = guess
                i -= 1
                result = func(i)
                while result == value:
                    i -= 1
                    result = func(i)
                low_level = i + 1

                # always choose the lowest level value of guess
                j = guess
                j += 1
                result = func(j)
                while result == value:
                    j += 1
                    result = func(j)
                high_level = j - 1

                return low_level, high_level

            # The correct pad is non-unique when it exists
            low_level, high_level = integer_step_linear_zero(func)
            got = low_level
            deltaf = fixed_expr.subs({solve_for: got}).evalf() - target
            delta = math.ceil(deltaf)

            # We return how much you need to pad and how much you need to crop
            # in order to get an output-size = input-size. pads and crops ard
            # in total, so do the propper floor / ceiling.
            crop = delta
            pad = got
            return pad, crop

        want_w, want_h = input_shape[2:4]
        prepad_w, postcrop_w = find_padding(W1, W2_raw, padw, want_w)
        prepad_h, postcrop_h = find_padding(H1, H2_raw, padh, want_h)

        prepad = (prepad_w, prepad_h)
        postcrop = (postcrop_w, postcrop_h)

        self._cache[input_shape] = (prepad, postcrop)
        print('prepad = {!r}'.format(prepad))
        print('postcrop = {!r}'.format(postcrop))
        return prepad, postcrop

    def prepad(self, inputs):
        # do appropriate mirroring so final.size()[-2:] >= input.size()[:-2]
        pad_wh, crop_wh = self.find_padding_and_crop_for(inputs.size())
        padw, padh = pad_wh
        halfw, halfh = padw / 2, padh / 2
        padding = [
            # Padding starts from the final dimension and then move backwards.
            math.floor(halfh),
            math.ceil(halfh),
            math.floor(halfw),
            math.ceil(halfw),
        ]
        mirrored = F.pad(inputs, padding, mode='reflect')
        return mirrored, crop_wh

    def postcrop(self, final, crop_wh):
        # do appropriate mirroring so final.size()[-2:] >= input.size()[:-2]
        w, h = crop_wh

        halfw, halfh = w / 2, h / 2
        # Padding starts from the final dimension and then move backwards.
        y1 = math.floor(halfh)
        y2 = final.size()[-1] - math.ceil(halfh)
        x1 = math.floor(halfw)
        x2 = final.size()[-2] - math.ceil(halfw)

        cropped = final[:, :, x1:x2, y1:y2]
        return cropped

    def raw_forward(self, inputs):
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)

        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)

        center = self.center(maxpool4)

        up4 = self.up_concat4(conv4, center)
        up3 = self.up_concat3(conv3, up4)
        up2 = self.up_concat2(conv2, up3)
        up1 = self.up_concat1(conv1, up2)

        final = self.final(up1)
        return final

    def forward(self, inputs):
        """
            >>> from clab.torch.models.unet import *  # NOQA
            >>> import torch
            >>> from torch.autograd import Variable
            >>> B, C, W, H = (4, 5, 256, 256)
            >>> n_classes = 11
            >>> inputs = Variable(torch.rand(B, C, W, H))
            >>> labels = Variable((torch.rand(B, W, H) * n_classes).long())
            >>> self = UNet(in_channels=C, n_classes=n_classes)
            >>> outputs = self.forward(inputs)
        """
        # Is there a way to miror so that we have enough input pixels?
        # so we can crop off extras after?
        if isinstance(inputs, tuple):
            inputs = inputs[0]

        mirrored = inputs
        mirrored, crop_wh = self.prepad(inputs)

        final = self.raw_forward(mirrored)

        cropped = self.postcrop(final, crop_wh)
        return cropped

    def trainable_layers(self):
        queue = [self]
        while queue:
            item = queue.pop(0)
            # TODO: need to put all trainable layer types here
            if isinstance(item, nn.Conv2d):
                yield item
            for child in item.children():
                queue.append(child)

    def init_he_normal(self):
        # down_blocks = [self.down1, self.down2, self.down3, self.down4, self.down5]
        # up_blocks = [self.up5, self.up4, self.up3, self.up2, self.up1]
        for layer in self.trainable_layers():
            nninit.he_normal(layer.weight)
            layer.bias.data.fill_(0)

    def shock_outward(self):
        for layer in self.trainable_layers():
            nninit.shock_outward(layer.weight)
            # shock inward
            layer.bias.data *= .1

    def load_partial_state(model, model_state_dict, shock_partial=True):
        """
        Example:
            >>> from clab.torch.models.unet import *  # NOQA
            >>> self1 = UNet(in_channels=5, n_classes=3)
            >>> self2 = UNet(in_channels=6, n_classes=4)
            >>> model_state_dict = self1.state_dict()
            >>> self2.load_partial_state(model_state_dict)

            >>> key = 'conv1.conv1.0.weight'
            >>> model = self2
            >>> other_value = model_state_dict[key]
        """
        self_state = model.state_dict()
        unused_keys = set(self_state.keys())

        for key, other_value in model_state_dict.items():
            if key in self_state:
                self_value = self_state[key]
                if other_value.size() == self_value.size():
                    self_state[key] = other_value
                    unused_keys.remove(key)
                elif len(other_value.size()) == len(self_value.size()):
                    if key.endswith('bias'):
                        print('Skipping {} due to incompatable size'.format(key))
                    else:
                        import numpy as np
                        print('Partially add {} with incompatable size'.format(key))
                        # Initialize all weights in case any are unspecified
                        nninit.he_normal(self_state[key])

                        # Transfer as much as possible
                        min_size = np.minimum(self_state[key].shape, other_value.shape)
                        sl = tuple([slice(0, s) for s in min_size])
                        self_state[key][sl] = other_value[sl]

                        if shock_partial:
                            # Shock weights because we are doing something weird
                            # might help the network recover in case this is
                            # not a good idea
                            nninit.shock_he(self_state[key], gain=1e-5)
                        unused_keys.remove(key)
                else:
                    print('Skipping {} due to incompatable size'.format(key))
            else:
                print('Skipping {} because it does not exist'.format(key))

        print('Initializing unused keys {} using he normal'.format(unused_keys))
        for key in unused_keys:
            if key.endswith('.bias'):
                self_state[key].fill_(0)
            else:
                nninit.he_normal(self_state[key])
        model.load_state_dict(self_state)


class UNet2(unet.UNet):
    """
    Note input shapes should be a power of 2.

    In this case there will be a ~188 pixel difference between input and output
    dims, so the input should be mirrored with

    Example:
        >>> from clab.live.unet2 import *  # NOQA
        >>> from torch.autograd import Variable
        >>> B, C, W, H = (4, 3, 256, 256)
        >>> B, C, W, H = (4, 3, 572, 572)
        >>> n_classes = 11
        >>> inputs = Variable(torch.rand(B, C, W, H))
        >>> labels = Variable((torch.rand(B, W, H) * n_classes).long())
        >>> self = UNet2(in_channels=C, n_classes=n_classes)
        >>> outputs = self.forward(inputs)
        >>> print('inputs.shape = {!r}'.format(inputs.shape))
        >>> print('outputs.shape = {!r}'.format(outputs.shape))
        >>> print(np.array(inputs.shape) - np.array(outputs.shape))

    Example:
        >>> from clab.torch.models.unet import *  # NOQA
        >>> from torch.autograd import Variable
        >>> B, C, W, H = (4, 5, 480, 360)
        >>> n_classes = 11
        >>> inputs = Variable(torch.rand(B, C, W, H))
        >>> labels = Variable((torch.rand(B, W, H) * n_classes).long())
        >>> self = UNet(in_channels=C, n_classes=n_classes)
        >>> outputs = self.forward(inputs)
        >>> print('inputs.shape = {!r}'.format(inputs.shape))
        >>> print('outputs.shape = {!r}'.format(outputs.shape))
        >>> print(np.array(inputs.shape) - np.array(outputs.shape))
    """
    def __init__(self, n_alt_classes=2, **kwargs):
        super(UNet2, self).__init__(**kwargs)

        self.final
        self.final2 = torch.nn.Conv2d(self.final.in_channels, n_alt_classes, 1)

    def raw_forward2(self, inputs):
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)

        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)

        center = self.center(maxpool4)

        up4 = self.up_concat4(conv4, center)
        up3 = self.up_concat3(conv3, up4)
        up2 = self.up_concat2(conv2, up3)
        up1 = self.up_concat1(conv1, up2)

        final = self.final(up1)
        final2 = self.final2(up1)
        return final, final2

    def forward(self, inputs):
        """
            >>> from clab.torch.models.unet import *  # NOQA
            >>> import torch
            >>> from torch.autograd import Variable
            >>> B, C, W, H = (4, 5, 256, 256)
            >>> n_classes = 11
            >>> inputs = Variable(torch.rand(B, C, W, H))
            >>> labels = Variable((torch.rand(B, W, H) * n_classes).long())
            >>> self = UNet(in_channels=C, n_classes=n_classes)
            >>> outputs = self.forward(inputs)
        """
        # Is there a way to miror so that we have enough input pixels?
        # so we can crop off extras after?
        if isinstance(inputs, (list, tuple)):
            assert len(inputs) == 1, 'unet2 only takes single branch input'
            inputs = inputs[0]

        mirrored = inputs
        mirrored, crop_wh = self.prepad(inputs)

        final1, final2 = self.raw_forward2(mirrored)

        cropped1 = self.postcrop(final1, crop_wh)
        cropped2 = self.postcrop(final2, crop_wh)
        return cropped1, cropped2
