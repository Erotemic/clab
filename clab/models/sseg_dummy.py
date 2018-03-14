import torch.nn as nn
import torch # NOQA

__all__ = ['SSegDummy']


class SSegDummy(nn.Module):
    """
    >>> import sys
    >>> from clab.models.segnet import *
    >>> n_classes = 12
    >>> in_channels = 5
    >>> self = SegNet(n_classes, in_channels)
    """
    def __init__(self, n_classes, in_channels=3):
        super(SSegDummy, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, n_classes, (1, 1))

        # self.conv1 = nn.Conv2d(3, 1, (3, 3), stride=100)
        # self.maxpool1 = nn.MaxPool2d(2, 2)
        # nn.Conv2d(in_channels,

    def forward(self, inputs):
        """
        Example:
            >>> from clab.models.sseg_dummy import *
            >>> from torch.autograd import Variable
            >>> B, C, W, H = (4, 5, 256, 256)
            >>> n_classes = 11
            >>> inputs = Variable(torch.rand(B, C, W, H))
            >>> labels = Variable((torch.rand(B, W, H) * n_classes).long())
            >>> self = SSegDummy(in_channels=C, n_classes=n_classes)
            >>> outputs = self.forward(inputs)
        """
        return self.conv1(inputs)
