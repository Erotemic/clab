"""
Adapated from:
    https://github.com/meetshah1995/pytorch-semseg
"""
import torch.nn as nn
from ._common import Conv2DBatchNormRelu

__all__ = ['SegNet']


class SegnetDown2(nn.Module):
    def __init__(self, in_size, out_size):
        super(SegnetDown2, self).__init__()
        self.conv1 = Conv2DBatchNormRelu(in_size, out_size, 3, 1, 1)
        self.conv2 = Conv2DBatchNormRelu(out_size, out_size, 3, 1, 1)
        self.maxpool_with_argmax = nn.MaxPool2d(2, 2, return_indices=True)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        unpooled_shape = outputs.size()
        outputs, indices = self.maxpool_with_argmax(outputs)
        return outputs, indices, unpooled_shape


class SegnetDown3(nn.Module):
    def __init__(self, in_size, out_size):
        super(SegnetDown3, self).__init__()
        self.conv1 = Conv2DBatchNormRelu(in_size, out_size, 3, 1, 1)
        self.conv2 = Conv2DBatchNormRelu(out_size, out_size, 3, 1, 1)
        self.conv3 = Conv2DBatchNormRelu(out_size, out_size, 3, 1, 1)
        self.maxpool_with_argmax = nn.MaxPool2d(2, 2, return_indices=True)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        outputs = self.conv3(outputs)
        unpooled_shape = outputs.size()
        outputs, indices = self.maxpool_with_argmax(outputs)
        return outputs, indices, unpooled_shape


class SegnetUp2(nn.Module):
    def __init__(self, in_size, out_size):
        super(SegnetUp2, self).__init__()
        self.unpool = nn.MaxUnpool2d(2, 2)
        self.conv1 = Conv2DBatchNormRelu(in_size, out_size, 3, 1, 1)
        self.conv2 = Conv2DBatchNormRelu(out_size, out_size, 3, 1, 1)

    def forward(self, inputs, indices, output_shape):
        outputs = self.unpool(input=inputs, indices=indices, output_size=output_shape)
        outputs = self.conv1(outputs)
        outputs = self.conv2(outputs)
        return outputs


class SegnetUp3(nn.Module):
    def __init__(self, in_size, out_size):
        super(SegnetUp3, self).__init__()
        self.unpool = nn.MaxUnpool2d(2, 2)
        self.conv1 = Conv2DBatchNormRelu(in_size, out_size, 3, 1, 1)
        self.conv2 = Conv2DBatchNormRelu(out_size, out_size, 3, 1, 1)
        self.conv3 = Conv2DBatchNormRelu(out_size, out_size, 3, 1, 1)

    def forward(self, inputs, indices, output_shape):
        outputs = self.unpool(input=inputs, indices=indices,
                              output_size=output_shape)
        outputs = self.conv1(outputs)
        outputs = self.conv2(outputs)
        outputs = self.conv3(outputs)
        return outputs


class SegNet(nn.Module):
    """
    >>> import sys
    >>> from clab.torch.models.segnet import *
    >>> n_classes = 12
    >>> in_channels = 5
    >>> self = SegNet(n_classes, in_channels)
    """
    def __init__(self, n_classes, in_channels=3, is_unpooling=True):
        super(SegNet, self).__init__()

        self.in_channels = in_channels
        self.is_unpooling = is_unpooling

        self.down1 = SegnetDown2(self.in_channels, 64)
        self.down2 = SegnetDown2(64, 128)
        self.down3 = SegnetDown3(128, 256)
        self.down4 = SegnetDown3(256, 512)
        self.down5 = SegnetDown3(512, 512)

        self.up5 = SegnetUp3(512, 512)
        self.up4 = SegnetUp3(512, 256)
        self.up3 = SegnetUp3(256, 128)
        self.up2 = SegnetUp2(128, 64)
        self.up1 = SegnetUp2(64, n_classes)

    def trainable_layers(self):
        queue = [self]
        while queue:
            item = queue.pop(0)
            # TODO: need to put all trainable layer types here
            if isinstance(item, nn.Conv2d):
                yield item
            for child in item.children():
                queue.append(child)

    def forward(self, inputs):
        """
            >>> from clab.torch.models.segnet import *  # NOQA
            >>> from torch.autograd import Variable
            >>> B, C, W, H = (4, 5, 256, 256)
            >>> n_classes = 11
            >>> inputs = Variable(torch.rand(B, C, W, H))
            >>> labels = Variable((torch.rand(B, W, H) * n_classes).long())
            >>> self = SegNet(in_channels=C, n_classes=n_classes)
            >>> outputs = self.forward(inputs)

        """

        down1, indices_1, unpool_shape1 = self.down1(inputs)
        down2, indices_2, unpool_shape2 = self.down2(down1)
        down3, indices_3, unpool_shape3 = self.down3(down2)
        down4, indices_4, unpool_shape4 = self.down4(down3)
        down5, indices_5, unpool_shape5 = self.down5(down4)

        up5 = self.up5(down5, indices_5, unpool_shape5)
        up4 = self.up4(up5, indices_4, unpool_shape4)
        up3 = self.up3(up4, indices_3, unpool_shape3)
        up2 = self.up2(up3, indices_2, unpool_shape2)
        up1 = self.up1(up2, indices_1, unpool_shape1)

        return up1

    def init_he_normal(self):
        # down_blocks = [self.down1, self.down2, self.down3, self.down4, self.down5]
        # up_blocks = [self.up5, self.up4, self.up3, self.up2, self.up1]
        for layer in self.trainable_layers():
            from clab.torch  import nninit
            nninit.he_normal(layer.weight)
            layer.bias.data.fill_(0)

    def init_vgg16_params(self):
        import torchvision
        print('initializing using VGG params')
        vgg16 = torchvision.models.vgg16(pretrained=True)

        # ranges = [[0, 4], [5, 9], [10, 16], [17, 23], [24, 29]]
        vgg_layers = [_layer for _layer in vgg16.features.children()
                      if isinstance(_layer, nn.Conv2d)]

        blocks = [self.down1,
                  self.down2,
                  self.down3,
                  self.down4,
                  self.down5]

        merged_layers = []
        for idx, conv_block in enumerate(blocks):
            if idx < 2:
                units = [conv_block.conv1.cbr_unit,
                         conv_block.conv2.cbr_unit]
            else:
                units = [conv_block.conv1.cbr_unit,
                         conv_block.conv2.cbr_unit,
                         conv_block.conv3.cbr_unit]
            for _unit in units:
                for _layer in _unit:
                    if isinstance(_layer, nn.Conv2d):
                        merged_layers.append(_layer)

        assert len(vgg_layers) == len(merged_layers)

        for internal, other in zip(merged_layers, vgg_layers):
            if isinstance(other, nn.Conv2d) and isinstance(internal, nn.Conv2d):
                assert other.bias.size() == internal.bias.size()
                ob, oc, ow, oh = other.weight.size()
                ib, ic, iw, ih = internal.weight.size()
                assert ob == ib and ow == iw and oh == ih
                assert oc <= ic
                # hack, when inputs have more channels try pulling in only the
                # first parts
                if oc == ic:
                    internal.weight.data = other.weight.data
                else:
                    internal.weight.data[:, 0:oc, :, :] = other.weight.data
                internal.bias.data = other.bias.data
