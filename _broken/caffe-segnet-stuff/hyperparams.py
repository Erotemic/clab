# -*- coding: utf-8 -*-
import six
import ubelt as ub


class HyperParams(object):
    """
    Holds params relavent to training strategy

        params = HyperParams()
    """
    def __init__(params, **kwargs):
        # # See this for info on params
        # https://github.com/BVLC/caffe/blob/master/src/caffe/proto/caffe.proto#L102
        params.solver_train = {
            'base_lr': 0.001,
            'lr_policy': "step",
            'gamma': 1.0,
            'stepsize': 10000000,
            'stepvalue': None,
            'momentum': 0.9,
            'weight_decay': 0.0005,
        }
        # these params dont influence the state of the model_train in any particular
        # iteration.
        params.solver_meta = {
            'test_initialization': False,
            'test_iter': 1,
            'test_interval': 10000000,
            'display': 20,
            'max_iter': 40000,
            'snapshot': 1000,
        }
        params.model_train = {
            'freeze_before': 0,
            'finetune_decay': 1,
        }
        params._dicts = [params.solver_train, params.solver_meta,
                         params.model_train]
        params.update(kwargs)

    def _normalize(params, d):
        """
        normalize for hashid generation
        """
        if d['lr_policy'] == 'step':
            del d['stepvalue']
        elif d['lr_policy'] == 'multistep':
            del d['stepsize']
        return d

    def protolines(params):
        r"""
        CommandLine:
            python -m pysseg.hyperparams HyperParams.protolines

        Example:
            >>> from pysseg.hyperparams import *  # NOQA
            >>> params = HyperParams(lr_policy='multistep', stepvalue=[1, 2, 3])
            >>> print('\n'.join(list(params.protolines())))
            base_lr: 0.001
            display: 20
            gamma: 1.0
            lr_policy: "multistep"
            max_iter: 40000
            momentum: 0.9
            snapshot: 1000
            stepvalue: 1
            stepvalue: 2
            stepvalue: 3
            test_initialization: false
            test_interval: 10000000
            test_iter: 1
            weight_decay: 0.0005
        """
        def format_item(key, value):
            if isinstance(value, list):
                for v in value:
                    yield from format_item(key, v)
            else:
                if isinstance(value, six.string_types):
                    value = '"{}"'.format(value)
                elif isinstance(value, bool):
                    value = str(value).lower()
                else:
                    value = str(value)
                yield '{}: {}'.format(key, value)

        for item in sorted(params.solverdict().items()):
            for line in format_item(*item):
                yield line

    def solverdict(params):
        ret = {}
        solver_dicts = [params.solver_train, params.solver_meta]
        for d in solver_dicts:
            ret.update(d)
        ret = params._normalize(ret)
        return ret

    def __setitem__(params, key, value):
        params.update({key: value})

    def __getitem__(params, key):
        for dict_ in params._dicts:
            if key in dict_:
                return dict_[key]
        raise KeyError(key)

    def update(params, other):
        remain = other.copy()
        # Update internal dicts, removing params as we see them
        for dict_ in params._dicts:
            for key in dict_:
                if key in remain:
                    dict_[key] = remain.pop(key)
        # Error if there were any unknown keys were given
        if len(remain) > 0:
            raise ValueError(
                'Specified Unknown Keys: {}'.format(list(remain.keys())))

    def hyper_id(params):
        """
        Identification string that uniquely determined by training params.
        Suitable for hashing.

        CommandLine:
            python -m pysseg.hyperparams HyperParams.hyper_id

        Example:
            >>> from pysseg.hyperparams import *  # NOQA
            >>> params = HyperParams(lr_policy='multistep', stepvalue=[1, 2, 3])
            >>> params.hyper_id()
            'base_lr=0.001,finetune_decay=1,freeze_before=0,gamma=1.0,lr_policy=multistep,momentum=0.9,stepvalue=[1,2,3],weight_decay=0.0005'
        """
        d = params.solver_train.copy()
        d.update(params.model_train)
        d = params._normalize(d)
        idstr = ub.repr2(d, itemsep='', nobr=True, explicit=True,
                         nl=0, si=True, sort=True)
        return idstr

if __name__ == '__main__':
    r"""
    CommandLine:
        python -m pysseg.hyperparams
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
