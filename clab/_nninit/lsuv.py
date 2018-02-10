"""
Modified from

https://github.com/ducha-aiki/LSUV-pytorch
"""
import numpy as np
import tqdm
import torch
import torch.nn.init
import torch.nn as nn
import ubelt as ub
from clab import util


def svd_orthonormal(shape, rng=None):
    """
    References:
        Orthonorm init code is taked from Lasagne
        https://github.com/Lasagne/Lasagne/blob/master/lasagne/init.py
    """
    rng = util.ensure_rng(rng)

    if len(shape) < 2:
        raise RuntimeError("Only shapes of length 2 or more are supported.")
    flat_shape = (shape[0], np.prod(shape[1:]))

    # rand_sequence = rng.randint(0, 2 ** 16)
    # depends = [shape, rand_sequence]
    depends = [shape]

    # this process can be expensive, cache it
    cacher = ub.Cacher('svd_orthonormal', appname='clab',
                        cfgstr=ub.hash_data(depends))
    q = cacher.tryload()
    if q is None:
        a = rng.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v
        # print(shape, flat_shape)
        q = q.reshape(shape)
        q = q.astype(np.float32)
        cacher.save(q)
    return q


class LSUV(object):
    """

    CommandLine:
        python -m clab._nninit.lsuv LSUV

    Example:
        >>> from clab._nninit.lsuv import *
        >>> import torchvision
        >>> import torch
        >>> #model = torchvision.models.AlexNet()
        >>> model = torchvision.models.SqueezeNet()
        >>> initer = LSUV()
        >>> data = torch.autograd.Variable(torch.randn(4, 3, 224, 224))
        >>> initer.forward(model, data)

    Example:
        >>> # DISABLE_DOCTEST
        >>> from clab._nninit.lsuv import *
        >>> import torchvision
        >>> import torch
        >>> #model = torchvision.models.AlexNet()
        >>> model = torchvision.models.SqueezeNet()
        >>> initer = LSUV()
        >>> data = torch.autograd.Variable(torch.randn(4, 3, 224, 224))
        >>> initer.forward(model, data)

    """
    def __init__(self, needed_std=1.0, std_tol=0.1, max_attempts=10,
                 do_orthonorm=True, rng=0):

        self.rng = util.ensure_rng(rng)

        self.do_orthonorm = do_orthonorm
        self.needed_std = needed_std
        self.std_tol = std_tol
        self.max_attempts = max_attempts
        self.gg = {}
        self.gg['hook_position'] = 0
        self.gg['total_fc_conv_layers'] = 0
        self.gg['done_counter'] = -1
        self.gg['hook'] = None
        self.gg['act_dict'] = {}
        self.gg['counter_to_apply_correction'] = 0
        self.gg['correction_needed'] = False
        self.gg['current_coef'] = 1.0

    def apply_weights_correction(self, m):
        if self.gg['hook'] is None:
            return
        if not self.gg['correction_needed']:
            return
        if (isinstance(m, nn.Conv2d)) or (isinstance(m, nn.Linear)):
            if self.gg['counter_to_apply_correction'] < self.gg['hook_position']:
                self.gg['counter_to_apply_correction'] += 1
            else:
                if hasattr(m, 'weight_g'):
                    m.weight_g.data *= float(self.gg['current_coef'])
                    #print(m.weight_g.data)
                    #print(m.weight_v.data)
                    #print('weights norm after = ', m.weight.data.norm())
                    self.gg['correction_needed'] = False
                else:
                    #print('weights norm before = ', m.weight.data.norm())
                    m.weight.data *= self.gg['current_coef']
                    #print('weights norm after = ', m.weight.data.norm())
                    self.gg['correction_needed'] = False
                return
        return

    def store_activations(self, module, input, output):
        self.gg['act_dict'] = output.data.cpu().numpy()
        return

    def add_current_hook(self, m):
        if self.gg['hook'] is not None:
            return
        if (isinstance(m, nn.Conv2d)) or (isinstance(m, nn.Linear)):
            #print('trying to hook to', m, self.gg['hook_position'], self.gg['done_counter'])
            if self.gg['hook_position'] > self.gg['done_counter']:
                self.gg['hook'] = m.register_forward_hook(self.store_activations)
                #print(' hooking layer = ', self.gg['hook_position'], m)
            else:
                #print(m, 'already done, skipping')
                self.gg['hook_position'] += 1
        return

    def count_conv_fc_layers(self, m):
        if (isinstance(m, nn.Conv2d)) or (isinstance(m, nn.Linear)):
            self.gg['total_fc_conv_layers'] += 1
        return

    def remove_hooks(self, hooks):
        for h in hooks:
            h.remove()
        return

    def orthogonal_weights_init(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            if hasattr(m, 'weight_v'):
                w_ortho = svd_orthonormal(m.weight_v.data.cpu().numpy().shape, self.rng)
                m.weight_v.data = torch.from_numpy(w_ortho)
                try:
                    nn.init.constant(m.bias, 0)
                except Exception:
                    pass
            else:
                #nn.init.orthogonal(m.weight)
                w_ortho = svd_orthonormal(m.weight.data.cpu().numpy().shape, self.rng)
                #print(w_ortho)
                #m.weight.data.copy_(torch.from_numpy(w_ortho))
                m.weight.data = torch.from_numpy(w_ortho)
                try:
                    nn.init.constant(m.bias, 0)
                except Exception:
                    pass
        return

    def forward(self, model, data):

        model.train(False)

        print('Starting LSUV')
        model.apply(self.count_conv_fc_layers)

        print('Total layers to process:', self.gg['total_fc_conv_layers'])
        if self.do_orthonorm:
            print('Applying orthogonal weights')
            model.apply(self.orthogonal_weights_init)
            print('Orthonorm done')
            # if cuda:
            #     model = model.cuda()

        for layer_idx in tqdm.trange(self.gg['total_fc_conv_layers'], desc='init layer', leave=True):
            # print(layer_idx)
            model.apply(self.add_current_hook)
            out = model(data)  # NOQA
            current_std = self.gg['act_dict'].std()
            tqdm.tqdm.write('layer {}: std={:.4f}'.format(layer_idx, current_std))
            #print  self.gg['act_dict'].shape
            attempts = 0
            for attempts in tqdm.trange(self.max_attempts, desc='iterate'):
                if not (np.abs(current_std - self.needed_std) > self.std_tol):
                    break
                self.gg['current_coef'] =  self.needed_std / (current_std  + 1e-8)
                self.gg['correction_needed'] = True
                model.apply(self.apply_weights_correction)

                # if cuda:
                #     model = model.cuda()

                out = model(data)  # NOQA

                current_std = self.gg['act_dict'].std()
                tqdm.tqdm.write('layer {}: std={:.4f}, mean={:.4f}'.format(
                        layer_idx, current_std, self.gg['act_dict'].mean()))
            if attempts >= self.max_attempts:
                tqdm.tqdm.write('Cannot converge in {} iterations'.format(self.max_attempts))
            if self.gg['hook'] is not None:
                self.gg['hook'].remove()
            self.gg['done_counter'] += 1
            self.gg['counter_to_apply_correction'] = 0
            self.gg['hook_position'] = 0
            self.gg['hook']  = None
            # print('finish at layer', layer_idx)
        print('LSUV init done!')

        # if not cuda:
        #     model = model.cpu()
        return model

if __name__ == '__main__':
    r"""
    CommandLine:
        python -m clab._nninit.lsuv
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
