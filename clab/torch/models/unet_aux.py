import torch.nn as nn
import torch
from .torch.models import unet
# import math
# import torch.nn.functional as F
# from ._common import OutputShapeFor

__all__ = ['InputAux2']


class InputAux2(nn.Module):
    def __init__(self):
        super(InputAux2, self).__init__()
        self._make_layer()
        self.aux_norm1 = self._make_layer()
        self.aux_norm2 = self._make_layer()

    def _make_layer(self):
        return nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1,
                      stride=1, padding=0),
            nn.BatchNorm2d(1),
        )

    def forward(self, inputs, aux1, aux2):
        """
            >>> from clab.torch.models.unet_aux import *
            >>> from clab.torch.models.unet import *
            >>> from torch.autograd import Variable
            >>> B, W, H = (4, 256, 256)
            >>> inputs = Variable(torch.rand(B, 3, W, H))
            >>> aux1 = Variable(torch.rand(B, 1, W, H))
            >>> aux2 = Variable(torch.rand(B, 1, W, H))
            >>> labels = Variable((torch.rand(B, W, H) * n_classes).long())
            >>> self = InputAux2()
            >>> outputs = self.forward(inputs, aux1, aux2)
        """
        aux1_ = self.aux_norm1(aux1)
        aux2_ = self.aux_norm2(aux2)
        combined_inputs = torch.cat([inputs, aux1_, aux2_], dim=1)
        return combined_inputs


class UnetAux2(nn.Module):
    def __init__(self, *args, **kwargs):
        self.model = unet.model(*args, **kwargs)

    def forward(self, inputs, aux1, aux2):
        combined_inputs = self.forward(inputs, aux1, aux2)
        output = self.model.forward(combined_inputs)
        return output
