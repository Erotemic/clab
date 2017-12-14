# -*- coding: utf-8 -*-
"""
Torch version of hyperparams
"""
import numpy as np
import ubelt as ub
import torch
import six
from clab import util
from clab.torch import criterions
from clab.torch import nninit
# from clab.torch import lr_schedule


def make_short_idstr(params):
    """
    Make id-string where they keys are shortened
    """
    if len(params) == 0:
        return ''
    short_keys = util.shortest_unique_prefixes(list(params.keys()),
                                               allow_simple=False,
                                               allow_end=True,
                                               min_length=1)
    def shortval(v):
        if isinstance(v, bool):
            return int(v)
        return v
    d = dict(zip(short_keys, map(shortval, params.values())))
    def make_idstr(d):
        return ub.repr2(d, itemsep='', nobr=True, explicit=True, nl=0,
                        si=True, sort=True)
    short_idstr = make_idstr(d)
    return short_idstr


def make_idstr(d):
    """
    Make full-length-key id-string
    """
    if len(d) == 0:
        return ''
    if not isinstance(d, ub.odict):
        d = ub.odict(sorted(d.items()))
    return ub.repr2(d, itemsep='', nobr=True, explicit=True, nl=0,
                    si=True)


def _rectify_class(lookup, arg, kw):
    if arg is None:
        return None, {}

    if isinstance(arg, tuple):
        cls = lookup(arg[0])
        kw2 = arg[1]
    else:
        cls = lookup(arg)
        kw2 = {}

    cls_kw = _class_default_params(cls).copy()
    cls_kw.update(kw2)
    for key in cls_kw:
        if key in kw:
            cls_kw[key] = kw.pop(key)
    return cls, cls_kw


def _class_default_params(cls):
    """
    cls = torch.optim.Adam
    cls = lr_schedule.Exponential
    """
    import inspect
    sig = inspect.signature(cls)
    default_params = {
        k: p.default
        for k, p in sig.parameters.items()
        if p.default is not p.empty
    }
    return default_params


def _rectify_criterion(arg, kw):
    if arg is None:
        # arg = 'CrossEntropyLoss'
        return None, None

    def _lookup(arg):
        if isinstance(arg, six.string_types):
            options = [
                criterions.CrossEntropyLoss2D,
                criterions.ContrastiveLoss,
                torch.nn.CrossEntropyLoss,
            ]
            cls = {c.__name__: c for c in options}[arg]
        else:
            cls = arg
        return cls

    cls, kw2 = _rectify_class(_lookup, arg, kw)
    return cls, kw2


def _rectify_optimizer(arg, kw):
    if arg is None:
        arg = 'SGD'

    def _lookup(arg):
        if isinstance(arg, six.string_types):
            options = [
                torch.optim.Adam,
                torch.optim.SGD,
            ]
            cls = {c.__name__.lower(): c for c in options}[arg.lower()]
        else:
            cls = arg
        return cls

    cls, kw2 = _rectify_class(_lookup, arg, kw)

    from torch.optim import optimizer
    for k, v in kw2.items():
        if v is optimizer.required:
            raise ValueError('Must specify {} for {}'.format(k, cls))

    return cls, kw2


def _rectify_lr_scheduler(arg, kw):
    if arg is None:
        return None, None
        # arg = 'Constant'

    def _lookup(arg):
        if isinstance(arg, six.string_types):
            options = [
                torch.optim.lr_scheduler.LambdaLR,
                torch.optim.lr_scheduler.StepLR,
                torch.optim.lr_scheduler.MultiStepLR,
                torch.optim.lr_scheduler.ExponentialLR,
                torch.optim.lr_scheduler.ReduceLROnPlateau,
                # lr_schedule.Constant,
                # lr_schedule.Exponential,
            ]
            cls = {c.__name__: c for c in options}[arg]
        else:
            cls = arg
        return cls

    cls, kw2 = _rectify_class(_lookup, arg, kw)
    return cls, kw2


def _rectify_initializer(arg, kw):
    if arg is None:
        # arg = 'CrossEntropyLoss'
        return None, None

    def _lookup(arg):
        if isinstance(arg, six.string_types):
            options = [
                nninit.HeNormal,
            ]
            cls = {c.__name__: c for c in options}[arg]
        else:
            cls = arg
        return cls

    cls, kw2 = _rectify_class(_lookup, arg, kw)
    return cls, kw2


def _rectify_model(arg, kw):
    if arg is None:
        return None, None

    def _lookup_model(arg):
        import torchvision
        if isinstance(arg, six.string_types):
            options = [
                torchvision.models.AlexNet,
                torchvision.models.DenseNet,
            ]
            cls = {c.__name__: c for c in options}[arg]
        else:
            cls = arg
        return cls

    cls, kw2 = _rectify_class(_lookup_model, arg, kw)
    return cls, kw2


class HyperParams(object):
    """
    Holds hyper relavent to training strategy

    CommandLine:
        python -m clab.torch.hyperparams HyperParams

    Example:
        >>> from clab.torch.hyperparams import *
        >>> hyper = HyperParams(
        >>>     criterion=('CrossEntropyLoss2D', {
        >>>         'weight': [0, 2, 1],
        >>>     }),
        >>>     optimizer=(torch.optim.SGD, {
        >>>         'nesterov': True, 'weight_decay': .0005,
        >>>         'momentum': 0.9, lr=.001,
        >>>     }),
        >>>     scheduler=('ReduceLROnPlateau', {}),
        >>> )
        >>> print(hyper.hyper_id())
    """

    def __init__(hyper, criterion=None, optimizer=None, scheduler=None,
                 model=None, other=None, initializer=None, **kwargs):

        cls, kw = _rectify_model(model, kwargs)
        hyper.model_cls = cls
        hyper.model_params = kw

        cls, kw = _rectify_optimizer(optimizer, kwargs)
        hyper.optimizer_cls = cls
        hyper.optimizer_params = kw
        # hyper.optimizer_params.pop('lr', None)  # hack

        cls, kw = _rectify_lr_scheduler(scheduler, kwargs)
        hyper.scheduler_cls = cls
        hyper.scheduler_params = kw

        # What if multiple criterions are used?
        cls, kw = _rectify_criterion(criterion, kwargs)
        hyper.criterion_cls = cls
        hyper.criterion_params = kw

        cls, kw = _rectify_initializer(initializer, kwargs)
        hyper.initializer_cls = cls
        hyper.initializer_params = kw

        if len(kwargs) > 0:
            raise ValueError('Unused kwargs {}'.format(list(kwargs.keys())))

        hyper.other = other
    # def _normalize(hyper):
    #     """
    #     normalize for hashid generation
    #     """
    #     weight = hyper.criterion_params.get('weight', None)
    #     if weight is not None:
    #         weight = list(map(float, weight))
    #         hyper.criterion_params['weight'] = weight

    def make_model(hyper):
        """ Instanciate the model defined by the hyperparams """
        model = hyper.model_cls(**hyper.model_params)
        return model

    def make_optimizer(hyper, parameters):
        """ Instanciate the optimizer defined by the hyperparams """
        optimizer = hyper.optimizer_cls(parameters, **hyper.optimizer_params)
        return optimizer

    def make_scheduler(hyper, optimizer):
        """ Instanciate the lr scheduler defined by the hyperparams """
        scheduler = hyper.scheduler_cls(optimizer, **hyper.scheduler_params)
        return scheduler

    def make_initializer(hyper):
        """ Instanciate the initializer defined by the hyperparams """
        initializer = hyper.initializer_cls(**hyper.initializer_params)
        return initializer

    def model_id(hyper, brief=False):
        """
        CommandLine:
            python -m clab.torch.hyperparams HyperParams.model_id

        Example:
            >>> from clab.torch.hyperparams import *
            >>> hyper = HyperParams(model='DenseNet', optimizer=('SGD', dict(lr=.001)))
            >>> print(hyper.model_id())
            >>> hyper = HyperParams(model='AlexNet', optimizer=('SGD', dict(lr=.001)))
            >>> print(hyper.model_id())
            >>> print(hyper.hyper_id())
            >>> hyper = HyperParams(model='AlexNet', optimizer=('SGD', dict(lr=.001)), scheduler='ReduceLROnPlateau')
            >>> print(hyper.hyper_id())
        """
        arch = hyper.model_cls.__name__
        # TODO: add model as a hyperparam specification
        # archkw = _class_default_params(hyper.model_cls)
        # archkw.update(hyper.model_params)
        archkw = hyper.model_params
        if brief:
            arch_id = arch + ',' + util.hash_data(make_short_idstr(archkw))[0:8]
        else:
            # arch_id = arch + ',' + make_idstr(archkw)
            arch_id = arch + ',' + make_short_idstr(archkw)
        return arch_id

    def other_id(hyper):
        """
            >>> from clab.torch.hyperparams import *
            >>> hyper = HyperParams(other={'augment': True, 'n_classes': 10, 'n_channels': 5})
            >>> hyper.hyper_id()
        """
        otherid = make_short_idstr(hyper.other)
        return otherid
        # short_keys = util.shortest_unique_prefixes(list(hyper.other.keys()))
        # def shortval(v):
        #     if isinstance(v, bool):
        #         return int(v)
        #     return v
        # d = dict(zip(short_keys, map(shortval, hyper.other.values())))
        # def make_idstr(d):
        #     return ub.repr2(d, itemsep='', nobr=True, explicit=True, nl=0,
        #                     si=True, sort=True)
        # otherid = make_idstr(d)
        # return otherid

    def hyper_id(hyper):
        """
        Identification string that uniquely determined by training hyper.
        Suitable for hashing.

        CommandLine:
            python -m clab.torch.hyperparams HyperParams.hyper_id

        Example:
            >>> from clab.torch.hyperparams import *
            >>> hyper = HyperParams(other={'n_classes': 10, 'n_channels': 5})
            >>> print(hyper.hyper_id())
        """
        # hyper._normalize()

        id_parts = []
        # total = ub.odict()

        def _make_part(cls, params):
            """
            append an id-string derived from the class and params.
            TODO: what if we have an instance and not a cls/params tuple?
            """
            if cls is None:
                return
            d = ub.odict()
            for k, v in sorted(params.items()):
                # if k in total:
                #     raise KeyError(k)
                if isinstance(v, torch.Tensor):
                    v = v.numpy()
                if isinstance(v, np.ndarray):
                    if v.kind == 'f':
                        v = list(map(float, v))
                    else:
                        raise NotImplementedError()
                d[k] = v
                # total[k] = v
            type_str = cls.__name__
            param_str = make_idstr(d)
            # param_str = make_short_idstr(d)
            assert ' at 0x' not in param_str, 'probably hashing an object: {}'.format(param_str)
            id_parts.append(type_str)
            if param_str:
                id_parts.append(param_str)

        _make_part(hyper.model_cls, hyper.model_params)
        _make_part(hyper.initializer_cls, hyper.initializer_params)
        _make_part(hyper.optimizer_cls, hyper.optimizer_params)
        _make_part(hyper.scheduler_cls, hyper.scheduler_params)
        _make_part(hyper.criterion_cls, hyper.criterion_params)

        idstr = ','.join(id_parts)
        return idstr

if __name__ == '__main__':
    r"""
    CommandLine:
        python -m clab.hyperparams
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
