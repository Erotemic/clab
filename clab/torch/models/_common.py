"""
References:
    https://github.com/meetshah1995/pytorch-semseg/blob/master/ptsemseg/models/utils.py
"""
import torch.nn as nn
from clab.torch.models.output_shape_for import OutputShapeFor


class Conv2DBatchNorm(nn.Module):
    def __init__(self, in_channels, n_filters, k_size,  stride, padding, bias=True):
        super(Conv2DBatchNorm, self).__init__()

        self.cb_unit = nn.Sequential(nn.Conv2d(int(in_channels), int(n_filters), kernel_size=k_size,
                                               padding=padding, stride=stride, bias=bias),
                                     nn.BatchNorm2d(int(n_filters)),)

    def forward(self, inputs):
        outputs = self.cb_unit(inputs)
        return outputs

    def output_shape_for(self, input_shape):
        return OutputShapeFor(self.cb_unit)(input_shape)


class Conv2DBatchNormRelu(nn.Module):
    def __init__(self, in_channels, n_filters, k_size,  stride, padding, bias=True):
        super(Conv2DBatchNormRelu, self).__init__()

        self.cbr_unit = nn.Sequential(nn.Conv2d(int(in_channels), int(n_filters), kernel_size=k_size,
                                                padding=padding, stride=stride, bias=bias),
                                      nn.BatchNorm2d(int(n_filters)),
                                      nn.ReLU(inplace=True),)

    def forward(self, inputs):
        outputs = self.cbr_unit(inputs)
        return outputs

    def output_shape_for(self, input_shape):
        return OutputShapeFor(self.cbr_unit)(input_shape)


class ResidualBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, n_filters, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()

        self.convbnrelu1 = Conv2DBatchNormRelu(in_channels, n_filters, 3,  stride, 1, bias=False)
        self.convbn2 = Conv2DBatchNorm(n_filters, n_filters, 3, 1, 1, bias=False)
        self.downsample = downsample
        self.stride = stride
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x

        out = self.convbnrelu1(x)
        out = self.convbn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out

    def output_shape_for(self, input_shape):
        residual = input_shape
        out = OutputShapeFor(self.convbnrelu1)(input_shape)
        out = OutputShapeFor(self.convbn2)(out)
        if self.downsample is not None:
            residual = OutputShapeFor(self.downsample)(input_shape)

        if residual[:-3] != out[:-3]:
            print('disagree:')
            print('out      = {!r}'.format(out))
            print('residual = {!r}'.format(residual))
        out = OutputShapeFor(self.relu)(out)
        return out


class Deconv2DBatchNorm(nn.Module):
    def __init__(self, in_channels, n_filters, k_size,  stride, padding, bias=True):
        super(Deconv2DBatchNorm, self).__init__()

        self.dcb_unit = nn.Sequential(nn.ConvTranspose2d(int(in_channels), int(n_filters), kernel_size=k_size,
                                                         padding=padding, stride=stride, bias=bias),
                                      nn.BatchNorm2d(int(n_filters)),)

    def forward(self, inputs):
        outputs = self.dcb_unit(inputs)
        return outputs

    def output_shape_for(self, input_shape):
        return OutputShapeFor(self.dcb_unit)(input_shape)


class Deconv2DBatchNormRelu(nn.Module):
    def __init__(self, in_channels, n_filters, k_size, stride, padding, bias=True):
        super(Deconv2DBatchNormRelu, self).__init__()

        self.dcbr_unit = nn.Sequential(nn.ConvTranspose2d(int(in_channels), int(n_filters), kernel_size=k_size,
                                                          padding=padding, stride=stride, bias=bias),
                                       nn.BatchNorm2d(int(n_filters)),
                                       nn.ReLU(inplace=True),)

    def forward(self, inputs):
        outputs = self.dcbr_unit(inputs)
        return outputs

    def output_shape_for(self, input_shape):
        return OutputShapeFor(self.dcbr_unit)(input_shape)
