"""
https://github.com/meetshah1995/pytorch-semseg/blob/master/ptsemseg/models/linknet.py
"""
import torch.nn as nn
from clab.torch.models._common import OutputShapeFor
from clab.torch.models._common import Conv2DBatchNormRelu, ResidualBlock, Deconv2DBatchNormRelu

__all__ = ['LinkNet']


class LinkNetUp(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(LinkNetUp, self).__init__()

        # B, 2C, H, W -> B, C/2, H, W
        self.convbnrelu1 = Conv2DBatchNormRelu(
            in_channels, n_filters / 2, k_size=1, stride=1, padding=1)

        # B, C/2, H, W -> B, C/2, H, W
        self.deconvbnrelu2 = Deconv2DBatchNormRelu(
            n_filters / 2, n_filters / 2, k_size=3,  stride=2, padding=0,)

        # B, C/2, H, W -> B, C, H, W
        self.convbnrelu3 = Conv2DBatchNormRelu(
            n_filters / 2, n_filters, k_size=1, stride=1, padding=1)

        self.unit = nn.Sequential(
            self.convbnrelu1,
            self.deconvbnrelu2,
            self.convbnrelu3,
        )

    def forward(self, x):
        x = self.unit(x)
        return x

    def output_shape_for(self, input_shape):
        return OutputShapeFor(self.unit)(input_shape)


class LinkNet(nn.Module):
    """
    References:
        https://arxiv.org/pdf/1707.03718.pdf
    """

    def __init__(self, feature_scale=4, n_classes=21, is_deconv=True,
                 in_channels=3, is_batchnorm=True):
        super(LinkNet, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        filters = [64, 128, 256, 512]
        filters = [int(x / self.feature_scale) for x in filters]

        self.inplanes = filters[0]

        # Encoder
        self.convbnrelu1 = Conv2DBatchNormRelu(
            in_channels=self.in_channels, k_size=7, n_filters=64, padding=3,
            stride=2, bias=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.n_blocks = [2, 2, 2, 2]  # Currently hardcoded for ResNet-18
        self.encoder1 = self._make_layer(filters[0], self.n_blocks[0])
        self.encoder2 = self._make_layer(filters[1], self.n_blocks[1], stride=2)
        self.encoder3 = self._make_layer(filters[2], self.n_blocks[2], stride=2)
        self.encoder4 = self._make_layer(filters[3], self.n_blocks[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)

        # Decoder
        self.decoder4 = LinkNetUp(filters[3], filters[2])
        self.decoder4 = LinkNetUp(filters[2], filters[1])
        self.decoder4 = LinkNetUp(filters[1], filters[0])
        self.decoder4 = LinkNetUp(filters[0], filters[0])

        # Final Classifier
        self.finaldeconvbnrelu1 = nn.Sequential(
            nn.ConvTranspose2d(filters[0], int(32 / feature_scale), 3, 2, 1),
            nn.BatchNorm2d(int(32 / feature_scale)),
            nn.ReLU(inplace=True),
        )
        self.finalconvbnrelu2 = Conv2DBatchNormRelu(
            in_channels=32 / feature_scale, k_size=3, n_filters=32 / feature_scale, padding=1, stride=1)
        self.finalconv3 = nn.Conv2d(int(32 / feature_scale), n_classes, 2, 2, 0)

    def _make_layer(self, planes, blocks, stride=1):
        downsample = None
        block = ResidualBlock
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(int(self.inplanes), int(planes * block.expansion),
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(int(planes * block.expansion)),
            )
        layers = [
            block(self.inplanes, planes, stride, downsample)
        ]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        # Encoder
        x = self.convbnrelu1(x)
        x = self.maxpool(x)

        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Decoder with Skip Connections
        d4 = self.decoder4(e4)
        d4 += e3
        d3 = self.decoder3(d4)
        d3 += e2
        d2 = self.decoder2(d3)
        d2 += e1
        d1 = self.decoder1(d2)

        # Final Classification
        f1 = self.finaldeconvbnrelu1(d1)
        f2 = self.finalconvbnrelu2(f1)
        f3 = self.finalconv3(f2)

        return f3
