import torch
import numpy as np


class Flatten(torch.nn.Module):
    """
    Example:
        >>> input = torch.rand(4, 3, 8, 8)
        >>> self = Flatten()
        >>> output = self(input)
        >>> output_shape = self.output_shape_for(input.shape)
        >>> list(output.shape) == output_shape
    """
    def forward(self, input):
        return input.view(input.size(0), -1)

    def output_shape_for(self, input_shape):
        bsize, *dims = input_shape
        return [bsize, np.prod(dims)]

    def activations_for(self, input_shape):
        return []
