"""
References:
    https://github.com/alykhantejani/nninit
"""
from os.path import dirname
import torch.nn as nn
import ubelt as ub
from os.path import exists
from os.path import join
import numpy as np
from clab import util
from torch.autograd import Variable
import math
import torch


class _BaseInitializer(object):
    """
    """
    def __call__(self, model):
        self.forward(model)

    def forward(self, model):
        """
        Abstract function that does the initailization
        """
        raise NotImplementedError('implement me')

    def history(self):
        """
        Initializer methods have histories which are short for algorithms and
        can be quite long for pretrained models
        """
        return None

    def get_initkw(self):
        """
        Initializer methods have histories which are short for algorithms and
        can be quite long for pretrained models
        """
        initkw = self.__dict__.copy()
        # info = {}
        # info['__name__'] = self.__class__.__name__
        # info['__module__'] = self.__class__.__module__
        # info['__initkw__'] = initkw
        return initkw


class Pretrained(_BaseInitializer):
    """
    Attributes:
        fpath (str): location of the pretrained weights file
        initializer (_BaseInitializer): backup initializer if the weights can
            only be partially applied
    """
    def __init__(self, fpath, initializer='HeNormal', shock_partial=False):
        self.fpath = fpath

        if isinstance(initializer, str):
            from clab.torch import nninit
            initializer = getattr(nninit, initializer)()

        self.initializer = initializer
        self.shock_partial = shock_partial

    def forward(self, model):
        model_state_dict = torch.load(self.fpath)
        if 'model_state_dict' in model_state_dict:
            model_state_dict = model_state_dict['model_state_dict']
        load_partial_state(model, model_state_dict,
                           initializer=self.initializer,
                           shock_partial=self.shock_partial)

    def history(self):
        """
        if available return the history of the model as well
        """
        info_dpath = dirname(dirname(ub.truepath(self.fpath)))
        info_fpath = join(info_dpath, 'train_info.json')
        if exists(info_fpath):
            return util.read_json(info_fpath)
        else:
            return '__UNKNOWN__'


class NoOp(_BaseInitializer):
    """
    Example:
        >>> from clab.torch.nninit import *
        >>> self = NoOp()
        >>> #info = self.history()
        >>> #assert info['__name__'] == 'NoOp'
    """
    def forward(self, model):
        return


class HeNormal(_BaseInitializer):
    """
    Example:
        >>> from clab.torch.nninit import *
        >>> self = HeNormal()
        >>> #info = self.history()
        >>> #assert info['__name__'] == 'HeNormal'
    """
    def __init__(self, gain=.01):
        self.gain = gain

    def forward(self, model):
        apply_initializer(model, he_normal, self.__dict__)


def kaiming_normal(tensor, nonlinearity='leaky_relu', param=0, mode='fan_in'):
    """
    similar to pytorch version 0.4, but exposes different params
    """
    if isinstance(tensor, Variable):
        kaiming_normal(tensor.data, nonlinearity=nonlinearity, param=param, mode=mode)
        return tensor

    fan = nn.init._calculate_correct_fan(tensor, mode)
    gain = nn.init.calculate_gain(nonlinearity=nonlinearity, param=param)
    std = gain / math.sqrt(fan)
    return tensor.normal_(0, std)


def kaiming_uniform(tensor, nonlinearity='leaky_relu', param=0, mode='fan_in'):
    if isinstance(tensor, Variable):
        kaiming_uniform(tensor, nonlinearity, param, mode)
        return tensor

    fan = nn.init._calculate_correct_fan(tensor, mode)
    gain = nn.init.calculate_gain(nonlinearity, param)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    return tensor.uniform_(-bound, bound)


class KaimingUniform(_BaseInitializer):
    """
    Same as HeNormal, but uses pytorch implementation

    Example:
        >>> from clab.torch.nninit import *
        >>> self = KaimingNormal()
    """
    def __init__(self, nonlinearity='leaky_relu', param=0, mode='fan_in'):
        self.nonlinearity = nonlinearity
        self.param = param
        self.mode = mode

    def forward(self, model):
        apply_initializer(model, kaiming_uniform, self.__dict__)


class KaimingNormal(_BaseInitializer):
    """
    Same as HeNormal, but uses pytorch implementation

    Example:
        >>> from clab.torch.nninit import *
        >>> self = KaimingNormal()
    """
    def __init__(self, nonlinearity='leaky_relu', param=0, mode='fan_in'):
        self.nonlinearity = nonlinearity
        self.param = param
        self.mode = mode

    def forward(self, model):
        apply_initializer(model, kaiming_normal, self.__dict__)


class Orthogonal(_BaseInitializer):
    """
    Same as HeNormal, but uses pytorch implementation

    Example:
        >>> from clab.torch.nninit import *
        >>> self = Orthogonal()
    """
    def __init__(self, gain=1):
        self.gain = gain

    def forward(self, model):
        apply_initializer(model, nn.init.orthogonal, self.__dict__)


def apply_initializer(input, func, funckw):
    """
    Args:
        input: can be a model, layer, or tensor
    """
    if getattr(input, 'bias', None) is not None:
        # zero all biases
        input.bias.data.zero_()

    if isinstance(input, (Variable, torch.Tensor)):
        assert False, ('input is tensor? does this make sense?')
        # data = input
    elif isinstance(input, (torch.nn.modules.conv._ConvNd)):
        func(input.weight, **funckw)
    elif isinstance(input, torch.nn.modules.batchnorm._BatchNorm):
        input.reset_parameters()
    # elif isinstance(input, torch.nn.modules.Linear):
    #     input.reset_parameters()
    elif hasattr(input, 'reset_parameters'):
        input.reset_parameters()
    else:
        # input is a torch module
        model = input
        for item in trainable_layers(model):
            apply_initializer(item, func, funckw)


def uniform(tensor, a=0, b=1):
    """Fills the input Tensor or Variable with values drawn from a uniform U(a,b)

    Args:
        tensor: a n-dimension torch.Tensor
        a: the lower bound of the uniform distribution
        b: the upper bound of the uniform distribution

    Examples:
        >>> w = torch.Tensor(3, 5)
        >>> nninit.uniform(w)
    """
    if isinstance(tensor, Variable):
        uniform(tensor.data, a=a, b=b)
        return tensor
    else:
        return tensor.uniform_(a, b)


def normal(tensor, mean=0, std=1):
    """Fills the input Tensor or Variable with values drawn from a normal distribution with the given mean and std

    Args:
        tensor: a n-dimension torch.Tensor
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution

    Examples:
        >>> w = torch.Tensor(3, 5)
        >>> nninit.normal(w)
    """
    if isinstance(tensor, Variable):
        normal(tensor.data, mean=mean, std=std)
        return tensor
    else:
        return tensor.normal_(mean, std)


def constant(tensor, val):
    """Fills the input Tensor or Variable with the value `val`

    Args:
        tensor: a n-dimension torch.Tensor
        val: the value to fill the tensor with

    Examples:
        >>> w = torch.Tensor(3, 5)
        >>> nninit.constant(w)
    """
    if isinstance(tensor, Variable):
        constant(tensor.data, val)
        return tensor
    else:
        return tensor.fill_(val)


def _calculate_fan_in_and_fan_out(tensor):
    if tensor.ndimension() < 2:
        raise ValueError("fan in and fan out can not be computed for tensor of size ", tensor.size())

    if tensor.ndimension() == 2:  # Linear
        fan_in = tensor.size(1)
        fan_out = tensor.size(0)
    else:
        num_input_fmaps = tensor.size(1)
        num_output_fmaps = tensor.size(0)
        receptive_field_size = np.prod(tensor.numpy().shape[2:])
        fan_in = num_input_fmaps * receptive_field_size
        fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out


def xavier_uniform(tensor, gain=1):
    """
    Fills the input Tensor or Variable with values according to the method
    described in "Understanding the difficulty of training deep feedforward
    neural networks" - Glorot, X. and Bengio, Y., using a uniform distribution.

    The resulting tensor will have values sampled from U(-a, a) where a = gain * sqrt(2/(fan_in + fan_out))

    Args:
        tensor: a n-dimension torch.Tensor
        gain: an optional scaling factor to be applied

    Examples:
        >>> w = torch.Tensor(3, 5)
        >>> nninit.xavier_uniform(w, gain=np.sqrt(2.0))
    """
    if isinstance(tensor, Variable):
        xavier_uniform(tensor.data, gain=gain)
        return tensor
    else:
        fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
        std = gain * np.sqrt(2.0 / (fan_in + fan_out))
        a = np.sqrt(3.0) * std
        return tensor.uniform_(-a, a)


def xavier_normal(tensor, gain=1):
    """Fills the input Tensor or Variable with values according to the method described in "Understanding the difficulty of training
       deep feedforward neural networks" - Glorot, X. and Bengio, Y., using a normal distribution. 2010

       The resulting tensor will have values sampled from normal distribution with mean=0 and
       std = gain * sqrt(2/(fan_in + fan_out))

    Args:
        tensor: a n-dimension torch.Tensor
        gain: an optional scaling factor to be applied

    Examples:
        >>> w = torch.Tensor(3, 5)
        >>> nninit.xavier_normal(w, gain=np.sqrt(2.0))
    """
    if isinstance(tensor, Variable):
        xavier_normal(tensor.data, gain=gain)
        return tensor
    else:
        fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
        std = gain * np.sqrt(2.0 / (fan_in + fan_out))
        return tensor.normal_(0, std)


def he_uniform(tensor, gain=1):
    """
    Fills the input Tensor or Variable with values according to the method
    described in "Delving deep into rectifiers: Surpassing human-level
    performance on ImageNet classification" - He, K. et al using a uniform
    distribution. 2015

    The resulting tensor will have values sampled from U(-a, a) where a = gain * sqrt(1/(fan_in))

    Args:
        tensor: a n-dimension torch.Tensor
        gain: an optional scaling factor to be applied

    Examples:
        >>> w = torch.Tensor(3, 5)
        >>> nninit.he_uniform(w, gain=np.sqrt(2.0))
    """

    if isinstance(tensor, Variable):
        he_uniform(tensor.data, gain=gain)
        return tensor
    else:
        fan_in, _ = _calculate_fan_in_and_fan_out(tensor)
        std = gain * np.sqrt(1.0 / fan_in)
        a = np.sqrt(3.0) * std
        return tensor.uniform_(-a, a)


def he_normal(tensor, gain=1):
    """
    Fills the input Tensor or Variable with values according to the method
    described in "Delving deep into rectifiers: Surpassing human-level
    performance on ImageNet classification" - He, K. et al using a normal
    distribution. 2015

   The resulting tensor will have values sampled from normal distribution with
   mean=0 and std = gain * sqrt(1/(fan_in))

    Args:
        tensor: a n-dimension torch.Tensor
        gain: an optional scaling factor to be applied

    Examples:
        >>> w = torch.Tensor(3, 5)
        >>> he_normal(w, gain=np.sqrt(2.0))
    """
    if isinstance(tensor, Variable):
        he_normal(tensor.data, gain=gain)
        return tensor
    else:
        fan_in, _ = _calculate_fan_in_and_fan_out(tensor)
        std = gain * np.sqrt(1.0 / fan_in)
        return tensor.normal_(0, std)


def orthogonal(tensor, gain=1):
    """Fills the input Tensor or Variable with a (semi) orthogonal matrix. The input tensor must have at least 2 dimensions,
       and for tensors with more than 2 dimensions the trailing dimensions are flattened. viewed as 2D representation with
       rows equal to the first dimension and columns equal to the product of  as a sparse matrix, where the non-zero elements
       will be drawn from a normal distribution with mean=0 and std=`std`.
       Reference: "Exact solutions to the nonlinear dynamics of learning in deep linear neural networks" - Saxe, A. et al.

    Args:
        tensor: a n-dimension torch.Tensor, where n >= 2
        gain: optional gain to be applied

    Examples:
        >>> w = torch.Tensor(3, 5)
        >>> nninit.orthogonal(w)
    """
    if isinstance(tensor, Variable):
        orthogonal(tensor.data, gain=gain)
        return tensor
    else:
        if tensor.ndimension() < 2:
            raise ValueError("Only tensors with 2 or more dimensions are supported.")

        flattened_shape = (tensor.size(0), int(np.prod(tensor.numpy().shape[1:])))
        flattened = torch.Tensor(flattened_shape[0], flattened_shape[1]).normal_(0, 1)

        u, s, v = np.linalg.svd(flattened.numpy(), full_matrices=False)
        if u.shape == flattened.numpy().shape:
            tensor.view_as(flattened).copy_(torch.from_numpy(u))
        else:
            tensor.view_as(flattened).copy_(torch.from_numpy(v))

        tensor.mul_(gain)
        return tensor


def sparse(tensor, sparsity, std=0.01):
    """Fills the 2D input Tensor or Variable as a sparse matrix, where the non-zero elements will be drawn from a
       normal distribution with mean=0 and std=`std`.

    Args:
        tensor: a n-dimension torch.Tensor
        sparsity: The fraction of elements in each column to be set to zero
        std: the standard deviation of the normal distribution used to generate the non-zero values

    Examples:
        >>> w = torch.Tensor(3, 5)
        >>> nninit.sparse(w, sparsity=0.1)
    """
    if isinstance(tensor, Variable):
        sparse(tensor.data, sparsity, std=std)
        return tensor
    else:
        if tensor.ndimension() != 2:
            raise ValueError("Sparse initialization only supported for 2D inputs")
        tensor.normal_(0, std)
        rows, cols = tensor.size(0), tensor.size(1)
        num_zeros = int(np.ceil(cols * sparsity))

        for col_idx in range(tensor.size(1)):
            row_indices = np.arange(rows)
            np.random.shuffle(row_indices)
            zero_indices = row_indices[:num_zeros]
            tensor.numpy()[zero_indices, col_idx] = 0

        return tensor


def shock_he(tensor):
    """
    Adds a very small he initial values to current tensor state.
    Helps tensor achieve full rank in case it lost it.

    DEPRICATE IN FAVOR OF ABSTRACT SHOCK

    Example:
        >>> tensor = torch.eye(3, 3)
        >>> tensor[0, 0] = 0
        >>> np.linalg.matrix_rank(tensor.numpy())
        2
        >>> shock_he(tensor)
        >>> np.linalg.matrix_rank(tensor.numpy())
        3
    """
    if isinstance(tensor, Variable):
        shock_he(tensor.data)
        return tensor
    else:
        # prb = tensor.copy()
        # he_normal(prb, gain)
        # tensor += prb
        # return tensor
        shock(tensor, he_normal, funckw={})
        # fan_in, _ = _calculate_fan_in_and_fan_out(tensor)
        # std = gain * np.sqrt(1.0 / fan_in)
        # prb = torch.randn(tensor.shape) * std
        # tensor += prb
        return tensor


def shock(tensor, func, scale=.0001, funckw={}):
    if isinstance(tensor, Variable):
        shock(tensor.data, func, scale, funckw)
        return tensor
    else:
        perterb = tensor.copy()
        # Init the perterbation matrix with the desired method and down scale
        func(perterb, **funckw)
        perterb *= scale
        # Shock the tensor by perterbing it
        tensor += perterb
        return tensor


# def shock_outward(tensor, scale=.1, a_min=.01):
#     """
#     send weights away from zero
#     """
#     if isinstance(tensor, Variable):
#         shock_outward(tensor.data, scale)
#         return tensor
#     else:
#         std = max(torch.abs(tensor).max(), a_min) * scale
#         # perterb outward
#         offset = np.abs(torch.randn(tensor.shape) * std) * torch.sign(tensor)
#         tensor += offset
#         return tensor

# TRAINABLE_LAYER_TYPES = [
#     # Any module with a reset_parameters()
#     torch.nn.modules.conv._ConvNd,
#     torch.nn.modules.batchnorm._BatchNorm,
#     torch.nn.modules.Linear,
#     torch.nn.modules.Embedding,
#     torch.nn.modules.EmbeddingBag,
#     torch.nn.modules.GRUCell,

# ]

# for key, value in vars(torch.nn.modules).items():
#     if hasattr(value, 'reset_parameters'):
#         print(key)


def trainable_layers(model):
    queue = [model]
    while queue:
        item = queue.pop(0)
        # TODO: need to put all trainable layer types here
        # (I think this is just everything with reset_parameters)
        if isinstance(item, torch.nn.modules.conv._ConvNd):
            yield item
        elif isinstance(item, torch.nn.modules.batchnorm._BatchNorm):
            yield item
        elif hasattr(item, 'reset_parameters'):
            yield item
        # if isinstance(input, torch.nn.modules.Linear):
        #     yield item
        # if isinstance(input, torch.nn.modules.Bilinear):
        #     yield item
        # if isinstance(input, torch.nn.modules.Embedding):
        #     yield item
        # if isinstance(input, torch.nn.modules.EmbeddingBag):
        #     yield item
        for child in item.children():
            queue.append(child)


def init_he_normal(model):
    for item in trainable_layers(model):
        if isinstance(item, torch.nn.modules.conv._ConvNd):
            he_normal(item.weight)
        if getattr(item, 'bias', None) is not None:
            item.bias.data.fill_(0)


def load_partial_state(model, model_state_dict, initializer=None, shock_partial=True):
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
    if initializer is None:
        initializer = he_normal

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
                    print('Partially add {} with incompatable size'.format(key))
                    # Initialize all weights in case any are unspecified
                    initializer(self_state[key])

                    # Transfer as much as possible
                    min_size = np.minimum(self_state[key].shape, other_value.shape)
                    sl = tuple([slice(0, s) for s in min_size])
                    self_state[key][sl] = other_value[sl]

                    if shock_partial:
                        # Shock weights because we are doing something weird
                        # might help the network recover in case this is
                        # not a good idea
                        shock(self_state[key], func=initializer)
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
            initializer(self_state[key])
    model.load_state_dict(self_state)
