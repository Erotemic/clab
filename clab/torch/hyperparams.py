# -*- coding: utf-8 -*-
"""
Torch version of hyperparams
"""
import ubelt as ub
import torch
import six
from . import util
from . import criterions
from . import lr_schedule


def _lookup_scheduler(arg):
    if isinstance(arg, six.string_types):
        options = [
            lr_schedule.Constant,
            lr_schedule.Exponential,
        ]
        cls = {c.__name__: c for c in options}[arg]
    else:
        cls = arg
    return cls


def _lookup_optimizer(arg):
    if isinstance(arg, six.string_types):
        options = [
            torch.optim.Adam,
            torch.optim.SGD,
        ]
        cls = {c.__name__.lower(): c for c in options}[arg.lower()]
    else:
        cls = arg
    return cls


def _lookup_criterion(arg):
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
        arg = 'CrossEntropyLoss'
    cls, kw2 = _rectify_class(_lookup_criterion, arg, kw)
    return cls, kw2


def _rectify_optimizer(arg, kw):
    if arg is None:
        arg = 'SGD'
    cls, kw2 = _rectify_class(_lookup_optimizer, arg, kw)
    return cls, kw2


def _rectify_lr_scheduler(arg, kw):
    if arg is None:
        arg = 'Constant'
    cls, kw2 = _rectify_class(_lookup_scheduler, arg, kw)
    return cls, kw2


class HyperParams(object):
    """
    Holds hyper relavent to training strategy

    Example:
        >>> from clab.torch.hyperparams import *
        >>> hyper = HyperParams(
        >>>     criterion=('CrossEntropyLoss2D', {
        >>>         'weight': [0, 2, 1],
        >>>     }),
        >>>     optimizer=(torch.optim.SGD, {
        >>>         'nesterov': True, 'weight_decay': .0005,
        >>>         'momentum': 0.9
        >>>     }),
        >>>     scheduler=('Exponential', {}),
        >>> )
        >>> hyper.criterion_params
        >>> hyper.optimizer_params
        >>> hyper.scheduler_params
    """

    def __init__(hyper, criterion=None, optimizer=None, scheduler=None,
                 other=None, **kwargs):

        cls, kw = _rectify_lr_scheduler(scheduler, kwargs)
        hyper.scheduler_cls = cls
        hyper.scheduler_params = kw

        cls, kw = _rectify_optimizer(optimizer, kwargs)
        hyper.optimizer_cls = cls
        hyper.optimizer_params = kw
        hyper.optimizer_params.pop('lr', None)  # hack

        cls, kw = _rectify_criterion(criterion, kwargs)
        hyper.criterion_cls = cls
        hyper.criterion_params = kw

        hyper.other = other

    def _normalize(hyper):
        """
        normalize for hashid generation
        """
        weight = hyper.criterion_params.get('weight', None)
        if weight is not None:
            weight = list(map(float, weight))
            hyper.criterion_params['weight'] = weight

    def other_id(hyper):
        """
            >>> from clab.torch.hyperparams import *
            >>> hyper = HyperParams(other={'augment': True, 'n_classes': 10, 'n_channels': 5})
            >>> hyper.hyper_id()
        """
        short_keys = util.shortest_unique_prefixes(list(hyper.other.keys()))
        def shortval(v):
            if isinstance(v, bool):
                return int(v)
            return v
        d = dict(zip(short_keys, map(shortval, hyper.other.values())))
        def make_idstr(d):
            return ub.repr2(d, itemsep='', nobr=True, explicit=True, nl=0,
                            si=True, sort=True)
        otherid = make_idstr(d)
        return otherid

    def hyper_id(hyper):
        """
        Identification string that uniquely determined by training hyper.
        Suitable for hashing.

        CommandLine:
            python -m clab.hyperparams HyperParams.hyper_id

        Example:
            >>> from clab.torch.hyperparams import *
            >>> hyper = HyperParams(other={'n_classes': 10, 'n_channels': 5})
            >>> hyper.hyper_id()
        """
        hyper._normalize()

        def make_idstr(d):
            return ub.repr2(d, itemsep='', nobr=True, explicit=True, nl=0,
                            si=True)
        id_parts = []
        d = ub.odict()
        total = ub.odict()
        for k, v in sorted(hyper.scheduler_params.items()):
            if k in total:
                raise KeyError(k)
            d[k] = v
            total[k] = v
        id_parts.append(hyper.scheduler_cls.__name__)
        id_parts.append(make_idstr(d))

        d = ub.odict()
        for k, v in sorted(hyper.criterion_params.items()):
            if k in total:
                raise KeyError(k)
            d[k] = v
            total[k] = v
        id_parts.append(hyper.criterion_cls.__name__)
        id_parts.append(make_idstr(d))

        d = ub.odict()
        for k, v in sorted(hyper.optimizer_params.items()):
            if k in total:
                raise KeyError(k)
            assert k != 'lr'
            # if k == 'lr':
            #     continue
            #     # if 'lr_base' in total:
            #     #     v = total['lr_base']
            d[k] = v
            total[k] = v
        id_parts.append(hyper.optimizer_cls.__name__)
        id_parts.append(make_idstr(d))

        # d = ub.odict()
        # for k, v in sorted(hyper.other.items()):
        #     if k in total:
        #         raise KeyError(k)
        #     d[k] = v
        #     total[k] = v
        # id_parts.append('other')
        # id_parts.append(make_idstr(d))

        idstr = ','.join(id_parts)
        return idstr

if __name__ == '__main__':
    r"""
    CommandLine:
        python -m clab.hyperparams
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
