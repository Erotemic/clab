# import torch
import torch.nn as nn
from clab.torch import nninit
import numpy as np


class NetMixin(object):
    def trainable_layers(self):
        queue = [self]
        while queue:
            item = queue.pop(0)
            # TODO: need to put all trainable layer types here
            if isinstance(item, nn.Conv2d):
                yield item
            for child in item.children():
                queue.append(child)

    def number_of_parameters(self):
        return sum([np.prod(p.size()) for p in self.parameters()])

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
