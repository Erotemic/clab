import math
import torch
import torch.nn as nn
import torchvision

REGISTERED_OUTPUT_SHAPE_TYPES = []


def compute_type(type):
    def _wrap(func):
        REGISTERED_OUTPUT_SHAPE_TYPES.append((type, func))
        return func
    return _wrap


class OutputShapeFor(object):

    math = math  # for hacking in sympy

    def __init__(self, module):
        self.module = module
        self._func = getattr(module, 'output_shape_for', None)
        if self._func is None:
            # Lookup shape func if we can't find it
            for type, _func in REGISTERED_OUTPUT_SHAPE_TYPES:
                try:
                    if module is type or isinstance(module, type):
                        self._func = _func
                except TypeError:
                    pass
            if not self._func:
                raise TypeError('Unknown module type {}'.format(module))

    def __call__(self, *args, **kwargs):
        if isinstance(self.module, nn.Module):
            # bound methods dont need module
            is_bound  = hasattr(self._func, '__func__') and getattr(self._func, '__func__', None) is not None
            is_bound |= hasattr(self._func, 'im_func') and getattr(self._func, 'im_func', None) is not None
            if is_bound:
                output_shape = self._func(*args, **kwargs)
            else:
                # nn.Module with state
                output_shape = self._func(self.module, *args, **kwargs)
        else:
            # a simple pytorch func
            output_shape = self._func(*args, **kwargs)
        return output_shape

    @staticmethod
    @compute_type(nn.UpsamplingBilinear2d)
    def UpsamplingBilinear2d(module, input_shape):
        """
        - Input: :math:`(N, C, H_{in}, W_{in})`
        - Output: :math:`(N, C, H_{out}, W_{out})` where
            :math:`H_{out} = floor(H_{in} * scale\_factor)`
            :math:`W_{out} = floor(W_{in}  * scale\_factor)`

        Example:
            >>> from clab.torch.models.output_shape_for import *
            >>> input_shape = (1, 3, 256, 256)
            >>> module = nn.UpsamplingBilinear2d(scale_factor=2)
            >>> output_shape = OutputShapeFor(module)(input_shape)
            >>> print('output_shape = {!r}'.format(output_shape))
            output_shape = (1, 3, 512, 512)
        """
        math = OutputShapeFor.math
        (N, C, H_in, W_in) = input_shape
        H_out = math.floor(H_in * module.scale_factor)
        W_out = math.floor(W_in  * module.scale_factor)
        output_shape = (N, C, H_out, W_out)
        return output_shape

    @staticmethod
    @compute_type(nn.ConvTranspose2d)
    def conv2dT(module, input_shape):
        """
          - Input: :math:`(N, C_{in}, H_{in}, W_{in})`
          - Output: :math:`(N, C_{out}, H_{out}, W_{out})` where
            :math:`H_{out} = (H_{in} - 1) * stride[0] - 2 * padding[0] + kernel\_size[0] + output\_padding[0]`
            :math:`W_{out} = (W_{in} - 1) * stride[1] - 2 * padding[1] + kernel\_size[1] + output\_padding[1]`

        Example:
            >>> from clab.torch.models.output_shape_for import *
            >>> input_shape = (1, 3, 256, 256)
            >>> module = nn.ConvTranspose2d(input_shape[1], 11, kernel_size=2, stride=2)
            >>> output_shape = OutputShapeFor(module)(input_shape)
            >>> print('output_shape = {!r}'.format(output_shape))
            output_shape = (1, 11, 512, 512)
        """
        (N, C_in, H_in, W_in) = input_shape
        C_out = module.out_channels
        stride = module.stride
        kernel_size = module.kernel_size
        output_padding = module.output_padding
        padding = module.padding
        H_out = (H_in - 1) * stride[0] - 2 * padding[0] + kernel_size[0] + output_padding[0]
        W_out = (W_in - 1) * stride[1] - 2 * padding[1] + kernel_size[1] + output_padding[1]
        output_shape = (N, C_out, H_out, W_out)
        return output_shape

    @staticmethod
    @compute_type(nn.Conv2d)
    def conv2d(module, input_shape):
        """
        Notes:
            - Input: :math:`(N, C_{in}, H_{in}, W_{in})`
            - Output: :math:`(N, C_{out}, H_{out}, W_{out})` where
                :math:`H_{out} = floor((H_{in}  + 2 * padding[0] - dilation[0] * (kernel\_size[0] - 1) - 1) / stride[0] + 1)`
                :math:`W_{out} = floor((W_{in}  + 2 * padding[1] - dilation[1] * (kernel\_size[1] - 1) - 1) / stride[1] + 1)`

        Example:
            >>> from clab.torch.models.output_shape_for import *
            >>> input_shape = (1, 3, 256, 256)
            >>> module = nn.Conv2d(input_shape[1], 11, 3, 1, 0)
            >>> output_shape = OutputShapeFor(module)(input_shape)
            >>> print('output_shape = {!r}'.format(output_shape))
            output_shape = (1, 11, 254, 254)
        """
        math = OutputShapeFor.math
        N, C_in, H_in, W_in = input_shape
        C_out = module.out_channels
        padding = module.padding
        stride = module.stride
        dilation = module.dilation
        kernel_size = module.kernel_size
        H_out = math.floor((H_in  + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1)
        W_out = math.floor((W_in  + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1] + 1)
        output_shape = (N, C_out, H_out, W_out)
        return output_shape

    @staticmethod
    @compute_type(nn.Conv3d)
    def conv3d(module, input_shape):
        """
        Notes:
           - Input: :math:`(N, C_{in}, D_{in}, H_{in}, W_{in})`
           - Output: :math:`(N, C_{out}, D_{out}, H_{out}, W_{out})` where
             :math:`D_{out} = floor((D_{in}  + 2 * padding[0] - dilation[0] * (kernel\_size[0] - 1) - 1) / stride[0] + 1)`
             :math:`H_{out} = floor((H_{in}  + 2 * padding[1] - dilation[1] * (kernel\_size[1] - 1) - 1) / stride[1] + 1)`
             :math:`W_{out} = floor((W_{in}  + 2 * padding[2] - dilation[2] * (kernel\_size[2] - 1) - 1) / stride[2] + 1)`

        Example:
            >>> from clab.torch.models.output_shape_for import *
            >>> input_shape = (1, 3, 25, 32, 32)
            >>> module = nn.Conv3d(in_channels=input_shape[1], out_channels=11,
            >>>                    kernel_size=(3, 3, 3), stride=1, padding=0,
            >>>                    dilation=1, groups=1, bias=True)
            >>> output_shape = OutputShapeFor(module)(input_shape)
            >>> print('output_shape = {!r}'.format(output_shape))
            output_shape = (1, 11, 23, 30, 30)
        """
        math = OutputShapeFor.math
        N, C_in, D_in, H_in, W_in = input_shape
        C_out = module.out_channels
        padding = module.padding
        stride = module.stride
        dilation = module.dilation
        kernel_size = module.kernel_size

        D_out = math.floor((D_in  + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1)
        H_out = math.floor((H_in  + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1] + 1)
        W_out = math.floor((W_in  + 2 * padding[2] - dilation[2] * (kernel_size[2] - 1) - 1) / stride[2] + 1)

        output_shape = (N, C_out, D_out, H_out, W_out)
        return output_shape

    @staticmethod
    @compute_type(nn.MaxPool3d)
    def max_pool3d(module, input_shape):
        """
          - Input: :math:`(N, C, D_{in}, H_{in}, W_{in})`
          - Output: :math:`(N, C, D_{out}, H_{out}, W_{out})` where
            :math:`D_{out} = floor((D_{in}  + 2 * padding[0] - dilation[0] * (kernel\_size[0] - 1) - 1) / stride[0] + 1)`
            :math:`H_{out} = floor((H_{in}  + 2 * padding[1] - dilation[1] * (kernel\_size[1] - 1) - 1) / stride[1] + 1)`
            :math:`W_{out} = floor((W_{in}  + 2 * padding[2] - dilation[2] * (kernel\_size[2] - 1) - 1) / stride[2] + 1)`
        """

        math = OutputShapeFor.math
        N, C_in, D_in, H_in, W_in = input_shape

        def ensure_iterable2(scalar):
            try:
                iter(scalar)
            except TypeError:
                return [scalar] * 3
            return scalar

        padding = ensure_iterable2(module.padding)
        stride = ensure_iterable2(module.stride)
        dilation = ensure_iterable2(module.dilation)
        kernel_size = ensure_iterable2(module.kernel_size)

        D_out = math.floor((D_in  + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1)
        H_out = math.floor((H_in  + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1] + 1)
        W_out = math.floor((W_in  + 2 * padding[2] - dilation[2] * (kernel_size[2] - 1) - 1) / stride[2] + 1)

        output_shape = (N, C_in, D_out, H_out, W_out)
        return output_shape

    @staticmethod
    @compute_type(torch.cat)
    def cat(input_shapes, dim=0):
        """

        Example:
            >>> from clab.torch.models.output_shape_for import *
            >>> input_shape1 = (1, 3, 256, 256)
            >>> input_shape2 = (1, 4, 256, 256)
            >>> input_shapes = [input_shape1, input_shape2]
            >>> output_shape = OutputShapeFor(torch.cat)(input_shapes, dim=1)
            >>> print('output_shape = {!r}'.format(output_shape))
            output_shape = [1, 7, 256, 256]
        """
        n_dims = max(map(len, input_shapes))
        assert n_dims == min(map(len, input_shapes))
        output_shape = [None] * n_dims
        for shape in input_shapes:
            for i, v in enumerate(shape):
                if output_shape[i] is None:
                    output_shape[i] = v
                else:
                    if i == dim:
                        output_shape[i] += v
                    else:
                        assert output_shape[i] == v, 'inconsistent dims'
        return output_shape

    @staticmethod
    @compute_type(nn.MaxPool2d)
    def maxpool2(module, input_shape):
        """

        Example:
            >>> from clab.torch.models.output_shape_for import *
            >>> input_shape = (1, 3, 256, 256)
            >>> module = nn.MaxPool2d(kernel_size=2)
            >>> output_shape = OutputShapeFor(module)(input_shape)
            >>> print('output_shape = {!r}'.format(output_shape))
            output_shape = [1, 7, 256, 256]

        Shape:
            Same as conv2 forumla except C2 = C1
            - Input: :math:`(N, C, H_{in}, W_{in})`
            - Output: :math:`(N, C, H_{out}, W_{out})` where
            :math:`H_{out} = floor((H_{in}  + 2 * padding[0] - dilation[0] * (kernel\_size[0] - 1) - 1) / stride[0] + 1)`
            :math:`W_{out} = floor((W_{in}  + 2 * padding[1] - dilation[1] * (kernel\_size[1] - 1) - 1) / stride[1] + 1)`
        """
        math = OutputShapeFor.math
        N, C, H_in, W_in = input_shape

        def ensure_iterable2(scalar):
            try:
                iter(scalar)
            except TypeError:
                return [scalar] * 2
            return scalar

        padding = ensure_iterable2(module.padding)
        stride = ensure_iterable2(module.stride)
        dilation = ensure_iterable2(module.dilation)
        kernel_size = ensure_iterable2(module.kernel_size)

        H_out = math.floor((H_in  + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1)
        W_out = math.floor((W_in  + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1] + 1)
        output_shape = (N, C, H_out, W_out)
        return output_shape

    @staticmethod
    @compute_type(nn.AvgPool2d)
    def avepool2d(module, input_shape):
        """
          Shape:
              - Input: :math:`(N, C, H_{in}, W_{in})`
              - Output: :math:`(N, C, H_{out}, W_{out})` where
                :math:`H_{out} = floor((H_{in}  + 2 * padding[0] - kernel\_size[0]) / stride[0] + 1)`
                :math:`W_{out} = floor((W_{in}  + 2 * padding[1] - kernel\_size[1]) / stride[1] + 1)`
        """
        math = OutputShapeFor.math
        N, C, H_in, W_in = input_shape

        def ensure_iterable2(scalar):
            try:
                iter(scalar)
            except TypeError:
                return [scalar] * 2
            return scalar

        padding = ensure_iterable2(module.padding)
        stride = ensure_iterable2(module.stride)
        kernel_size = ensure_iterable2(module.kernel_size)

        H_out = math.floor((H_in + 2 * padding[0] - kernel_size[0]) / stride[0] + 1)
        W_out = math.floor((W_in + 2 * padding[1] - kernel_size[1]) / stride[1] + 1)
        output_shape = (N, C, H_out, W_out)
        return output_shape

    @staticmethod
    @compute_type(nn.Linear)
    def linear(module, input_shape):
        """
           Shape:
               - Input: :math:`(N, *, in\_features)` where `*` means any number of
                 additional dimensions
               - Output: :math:`(N, *, out\_features)` where all but the last dimension
                 are the same shape as the input.
        """
        N, *other, in_feat = input_shape
        output_shape = [N] + other + [module.out_features]
        return output_shape

    @staticmethod
    @compute_type(nn.BatchNorm2d)
    def batchnorm(module, input_shape):
        return input_shape

    @staticmethod
    @compute_type(nn.ReLU)
    def relu(module, input_shape):
        return input_shape

    @staticmethod
    @compute_type(nn.LeakyReLU)
    def leaky_relu(module, input_shape):
        return input_shape

    @staticmethod
    @compute_type(nn.Sequential)
    def sequential(module, input_shape):
        shape = input_shape
        for child in module._modules.values():
            shape = OutputShapeFor(child)(shape)
        return shape

    @staticmethod
    @compute_type(torchvision.models.resnet.BasicBlock)
    def resent_basic_block(module, input_shape):
        residual_shape = input_shape
        shape = input_shape

        shape = OutputShapeFor(module.conv1)(shape)
        shape = OutputShapeFor(module.bn1)(shape)
        shape = OutputShapeFor(module.relu)(shape)

        shape = OutputShapeFor(module.conv2)(shape)
        shape = OutputShapeFor(module.bn2)(shape)
        shape = OutputShapeFor(module.relu)(shape)

        if module.downsample is not None:
            residual_shape = OutputShapeFor(module.downsample)(residual_shape)

        # assert residual_shape[-2:] == shape[-2:], 'cannot add residual {} {}'.format(residual_shape, shape)
        # out += residual
        shape = OutputShapeFor(module.relu)(shape)
        # print('BASIC residual_shape = {!r}'.format(residual_shape[-2:]))
        # print('BASIC shape          = {!r}'.format(shape[-2:]))
        # print('---')
        return shape

    @staticmethod
    @compute_type(torchvision.models.resnet.Bottleneck)
    def resent_bottleneck(module, input_shape):
        residual_shape = input_shape
        shape = input_shape

        shape = OutputShapeFor(module.conv1)(shape)
        shape = OutputShapeFor(module.bn1)(shape)
        shape = OutputShapeFor(module.relu)(shape)

        shape = OutputShapeFor(module.conv2)(shape)
        shape = OutputShapeFor(module.bn2)(shape)
        shape = OutputShapeFor(module.relu)(shape)

        shape = OutputShapeFor(module.conv3)(shape)
        shape = OutputShapeFor(module.bn3)(shape)

        if module.downsample is not None:
            residual_shape = OutputShapeFor(module.downsample)(input_shape)

        assert residual_shape[-2:] == shape[-2:], 'cannot add residual {} {}'.format(residual_shape, shape)
        # out += residual
        shape = OutputShapeFor(module.relu)(shape)
        # print('bottle downsample     = {!r}'.format(module.downsample))
        # print('bottle input_shape    = {!r}'.format(input_shape[-2:]))
        # print('bottle residual_shape = {!r}'.format(residual_shape[-2:]))
        # print('bottle shape          = {!r}'.format(shape[-2:]))
        # print('---')
        return shape

    @staticmethod
    @compute_type(torchvision.models.resnet.ResNet)
    def resnet_model(module, input_shape):
        shape = input_shape
        shape = OutputShapeFor(module.conv1)(shape)
        shape = OutputShapeFor(module.bn1)(shape)
        shape = OutputShapeFor(module.relu)(shape)
        shape = OutputShapeFor(module.maxpool)(shape)

        shape = OutputShapeFor(module.layer1)(shape)
        shape = OutputShapeFor(module.layer2)(shape)
        shape = OutputShapeFor(module.layer3)(shape)
        shape = OutputShapeFor(module.layer4)(shape)

        shape = OutputShapeFor(module.avgpool)(shape)
        print('pre-flatten-shape = {!r}'.format(shape))

        def prod(args):
            result = args[0]
            for arg in args[1:]:
                result = result * arg
            return result
        shape = (shape[0], prod(shape[1:]))
        # shape = shape.view(shape.size(0), -1)

        shape = OutputShapeFor(module.fc)(shape)

    @staticmethod
    def resnet_conv_part(module, input_shape):
        shape = input_shape
        shape = OutputShapeFor(module.conv1)(shape)
        shape = OutputShapeFor(module.bn1)(shape)
        shape = OutputShapeFor(module.relu)(shape)
        shape = OutputShapeFor(module.maxpool)(shape)

        shape = OutputShapeFor(module.layer1)(shape)
        shape = OutputShapeFor(module.layer2)(shape)
        shape = OutputShapeFor(module.layer3)(shape)
        shape = OutputShapeFor(module.layer4)(shape)

        shape = OutputShapeFor(module.avgpool)(shape)
        # print('pre-flatten-shape = {!r}'.format(shape))

        def prod(args):
            result = args[0]
            for arg in args[1:]:
                result = result * arg
            return result
        shape = (shape[0], prod(shape[1:]))
        # shape = shape.view(shape.size(0), -1)
        return shape
