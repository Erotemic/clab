# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
# import numpy as np
import sys
import tqdm
import glob
from os.path import join
import os
import ubelt as ub
import torch
import torch.nn
from torch.autograd import Variable
import tensorboard_logger
import torchvision  # NOQA
import itertools as it
from clab.torch import metrics
from clab.torch import xpu_device
from clab.torch import early_stop
from clab import util  # NOQA
from clab import getLogger
import time
logger = getLogger(__name__)
print = util.protect_print(logger.info)


def number_of_parameters(model, trainable=True):
    """ TODO: move somewhere else """
    import numpy as np
    if trainable:
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    else:
        model_parameters = model.parameters()
    n_params = sum([np.prod(p.size()) for p in model_parameters])
    return n_params


def mnist_demo():
    """
    CommandLine:
        python -m clab.torch.fit_harness mnist_demo

    Example:
        >>> mnist_demo()
    """
    from clab.torch import models
    from clab.torch import hyperparams
    root = os.path.expanduser('~/data/mnist/')

    dry = ub.argflag('--dry')

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])
    datasets = {
        'train': torchvision.datasets.MNIST(root, transform=transform,
                                            train=True, download=True),
        'vali': torchvision.datasets.MNIST(root, transform=transform,
                                           train=False, download=True),
    }

    # take a subset of data
    factor = 1
    datasets['train'].train_data = datasets['train'].train_data[::factor]
    datasets['train'].train_labels = datasets['train'].train_labels[::factor]

    factor = 1
    datasets['vali'].test_data = datasets['vali'].test_data[::factor]
    datasets['vali'].test_labels = datasets['vali'].test_labels[::factor]

    # Give the training dataset an input_id
    datasets['train'].input_id = 'mnist'

    batch_size = 128
    n_classes = 10
    xpu = xpu_device.XPU.from_argv(min_memory=300)
    # model = models.MnistNet(n_channels=1, n_classes=n_classes)
    from clab.torch import nninit

    hyper = hyperparams.HyperParams(
        model=(models.MnistNet, dict(n_channels=1, n_classes=n_classes)),
        optimizer=torch.optim.Adam,
        scheduler='ReduceLROnPlateau',
        criterion=torch.nn.CrossEntropyLoss,
        initializer=(nninit.HeNormal, {}),
        # initializer=(nninit.Pretrained, dict(fpath=ub.truepath('~/data/work/mnist/arch/MnistNet/train/input_mnist/solver_mnist_MnistNet,ufbg,Pretrained,dwlk,Adam,apce,ReduceLROnPlateau,zagw,CrossEntropyLoss,ceip_n=10/torch_snapshots/_epoch_00000095.pt'))),
        # initializer=(nninit.Pretrained, dict(fpath=ub.truepath('~/data/work/mnist/arch/MnistNet/train/input_mnist/solver_mnist_MnistNet,ufbg,HeNormal,akyu,Adam,apce,ReduceLROnPlateau,zagw,CrossEntropyLoss,ceip_n=10/torch_snapshots/_epoch_00000001.pt'))),
        # stopping=('EarlyStopping', dict(patience=10)),  # TODO
        # initializer='he',
        other={
            'n_classes': n_classes,
            # 'n_channels': n_channels,
        }
    )

    # train_dpath = os.path.expanduser('~/data/work/mnist/harness/mnist-net')
    workdir = os.path.expanduser('~/data/work/mnist/')

    harn = FitHarness(
        datasets=datasets, batch_size=batch_size,
        xpu=xpu, hyper=hyper, dry=dry,
    )

    # all_labels = np.arange(n_classes)
    # @harn.add_metric_hook
    # def custom_metrics(harn, output, labels):
    #     # ignore_label = datasets['train'].ignore_label
    #     # labels = datasets['train'].task.labels
    #     label = labels[0]
    #     metrics_dict = metrics._clf_metrics(output, label, labels=all_labels)
    #     return metrics_dict

    train_dpath = harn.setup_dpath(workdir, hashed=True)
    print('train_dpath = {!r}'.format(train_dpath))

    if ub.argflag('--reset'):
        ub.delete(train_dpath)

    harn.run()

    # if False:
    #     import plottool as pt
    #     pt.qtensure()
    #     ims, gts = next(iter(harn.loaders['train']))
    #     pic = im_loaders.rgb_tensor_to_imgs(ims, norm=False)[0]
    #     pt.clf()
    #     pt.imshow(pic, norm=True, cmap='viridis', data_colorbar=True)

    #     with pt.RenderingContext() as render:
    #         tensor_data = datasets['train'][0][0][None, :]
    #         pic = im_loaders.rgb_tensor_to_imgs(tensor_data, norm=False)[0]
    #         pt.figure(fnum=1, doclf=True)
    #         pt.imshow(pic, norm=True, cmap='viridis', data_colorbar=True,
    #                   fnum=1)
    #     render.image


class FitHarness(object):
    def __init__(harn, datasets, batch_size=4, hyper=None, xpu=None,
                 train_dpath='./train', dry=False):

        harn.datasets = None
        harn.loaders = None

        harn.dry = dry
        if harn.dry:
            train_dpath = ub.ensure_app_cache_dir('clab/dry')
            ub.delete(train_dpath)
            train_dpath = ub.ensure_app_cache_dir('clab/dry')

        harn.dry = dry
        harn.train_dpath = train_dpath

        if harn.dry:
            harn.xpu = xpu_device.XPU(None)
        else:
            harn.xpu = xpu_device.XPU(xpu)

        harn._setup_loaders(datasets, batch_size)

        harn.hyper = hyper

        harn.main_prog = None

        harn._tensorboard_hooks = []
        harn._metric_hooks = []
        harn._run_metrics = None
        harn._custom_run_batch = None

        harn._epoch_callbacks = []
        harn._iter_callbacks = []

        harn.model = None
        # harn.initializer should be a hyperparam
        harn.optimizer = None
        harn.scheduler = None
        # should this be a hyperparam? YES, maybe it doesn't change the
        # directory output, but it should be configuarble.
        harn.stopping = early_stop.EarlyStop(patience=30)

        # this is optional and is designed for default solvers
        # can be overriden by a custom_run_batch
        harn.criterion = None

        harn.intervals = {
            'display_train': 1,
            'display_vali': 1,
            'display_test': 1,

            'vali': 1,
            'test': 1,

            'snapshot': 1,
        }
        harn.config = {
            'show_prog': True,
            'max_iter': 1000,
        }
        harn.epoch = 0

    def setup_dpath(harn, workdir='.', **kwargs):
        from clab.torch import folder_structure
        structure = folder_structure.DirectoryStructure(
            workdir=workdir, hyper=harn.hyper, datasets=harn.datasets,
        )
        dpath = structure.setup_dpath(**kwargs)
        harn.train_dpath = dpath
        return dpath

    def _setup_loaders(harn, datasets, batch_size):
        data_kw = {'batch_size': batch_size}
        if harn.xpu.is_gpu():
            data_kw.update({'num_workers': 6, 'pin_memory': True})

        tags = ['train', 'vali', 'test']

        harn.loaders = ub.odict()
        harn.datasets = datasets
        for tag in tags:
            if tag not in datasets:
                continue
            dset = datasets[tag]
            shuffle = tag == 'train'
            data_kw_ = data_kw.copy()
            if tag != 'train':
                data_kw_['batch_size'] = max(batch_size // 4, 1)
            loader = torch.utils.data.DataLoader(dset, shuffle=shuffle,
                                                 **data_kw_)
            harn.loaders[tag] = loader

    def initialize_training(harn):
        # TODO: Initialize the classes and then have a different function move
        # everything to GPU
        harn.xpu.set_as_default()

        if tensorboard_logger:
            train_base = os.path.dirname(harn.train_dpath)
            harn.log('dont forget to start: tensorboard --logdir ' + train_base)
            harn.log('Initializing tensorboard')
            harn.tlogger = tensorboard_logger.Logger(harn.train_dpath,
                                                     flush_secs=2)

        if harn.dry:
            harn.log('Dry run of training harness. xpu={}'.format(harn.xpu))
            harn.optimizer = None
        else:
            prev_states = harn.prev_snapshots()

            model_name = harn.hyper.model_cls.__name__

            if harn.hyper.criterion_cls:
                harn.log('Criterion: {}'.format(harn.hyper.criterion_cls.__name__))
            else:
                harn.log('Criterion: Custom')

            harn.log('Optimizer: {}'.format(harn.hyper.optimizer_cls.__name__))

            if harn.hyper.scheduler_cls:
                harn.log('Scheduler: {}'.format(harn.hyper.scheduler_cls.__name__))
            else:
                harn.log('No Scheduler')

            harn.log('Fitting {} model on {}'.format(model_name, harn.xpu))

            harn.model = harn.hyper.make_model()
            harn.initializer = harn.hyper.make_initializer()

            harn.initializer(harn.model)

            n_params = number_of_parameters(harn.model)
            print('Model has {!r} parameters'.format(n_params))

            harn.log('There are {} existing snapshots'.format(len(prev_states)))
            harn.xpu.to_xpu(harn.model)

            # more than one criterion?
            if harn.hyper.criterion_cls:

                # weight = harn.hyper.criterion_params.get('weight', None)
                # if weight is not None:
                #     harn.log('Casting weights')
                #     weight = torch.FloatTensor(harn.hyper.criterion_params['weight'])
                #     weight = harn.xpu.to_xpu(weight)
                #     harn.hyper.criterion_params['weight'] = weight

                harn.criterion = harn.hyper.criterion_cls(
                    **harn.hyper.criterion_params)

                harn.criterion = harn.xpu.to_xpu(harn.criterion)
            else:
                pass

            harn.optimizer = harn.hyper.make_optimizer(harn.model.parameters())

            if harn.hyper.scheduler_cls:
                harn.scheduler = harn.hyper.make_scheduler(harn.optimizer)

            if prev_states:
                harn.load_snapshot(prev_states[-1])

                for i, group in enumerate(harn.optimizer.param_groups):
                    if 'initial_lr' not in group:
                        raise KeyError("param 'initial_lr' is not specified "
                                       "in param_groups[{}] when resuming an optimizer".format(i))

            else:
                if not harn.dry:
                    for group in harn.optimizer.param_groups:
                        group.setdefault('initial_lr', group['lr'])

            harn.log('New snapshots will save harn.snapshot_dpath = {!r}'.format(harn.snapshot_dpath))

    def current_lrs(harn):
        if harn.scheduler is None:
            if harn.optimizer is None:
                assert harn.dry
                lrs = [.01]
            else:
                lrs = set(map(lambda group: group['lr'], harn.optimizer.param_groups))
        elif hasattr(harn.scheduler, 'get_lr'):
            lrs = set(harn.scheduler.get_lr())
        else:
            # workaround for ReduceLROnPlateau
            lrs = set(map(lambda group: group['lr'], harn.scheduler.optimizer.param_groups))
        return lrs

    def update_prog_description(harn):
        lrs = harn.current_lrs()
        lr_str = ','.join(['{:.2g}'.format(lr) for lr in lrs])
        desc = 'epoch lr:{} │ {}'.format(lr_str, harn.stopping.message())
        harn.main_prog.set_description(desc)
        harn.main_prog.set_postfix(
            {'wall': time.strftime('%H:%M') + ' ' + time.tzname[0]})
        harn.main_prog.refresh()

    def run(harn):
        """
        Main training loop
        """
        harn.log('Begin training')

        harn.initialize_training()

        if harn._check_termination():
            return

        harn.main_prog = tqdm.tqdm(desc='epoch', total=harn.config['max_iter'],
                                   disable=not harn.config['show_prog'],
                                   leave=True, dynamic_ncols=True, position=1,
                                   initial=harn.epoch)
        harn.update_prog_description()

        train_loader = harn.loaders['train']
        vali_loader  = harn.loaders.get('vali', None)
        test_loader  = harn.loaders.get('test', None)

        if not vali_loader:
            if not harn.scheduler:
                if harn.stopping:
                    raise ValueError('need a validataion dataset to use early stopping')
                if harn.scheduler.__class__.__name__ == 'ReduceLROnPlateau':
                    raise ValueError('need a validataion dataset to use ReduceLROnPlateau')

        # Keep track of moving metric averages across epochs
        harn._run_metrics = {
            tag: metrics.WindowedMovingAve(window=len(loader))
            for tag, loader in harn.loaders.items()
        }

        try:
            for harn.epoch in it.count(harn.epoch):

                # Run training epoch
                harn.run_epoch(train_loader, tag='train', learn=True)

                # Run validation epoch
                vali_metrics = None
                if vali_loader:
                    if harn.check_interval('vali', harn.epoch):
                        vali_metrics = harn.run_epoch(
                            vali_loader, tag='vali', learn=False)
                        harn.stopping.update(harn.epoch, vali_metrics['loss'])

                    harn.update_prog_description()

                # Run test epoch
                if test_loader:
                    if harn.check_interval('test', harn.epoch):
                        harn.run_epoch(test_loader, tag='test', learn=False)

                if harn.check_interval('snapshot', harn.epoch):
                    harn.save_snapshot()

                harn.main_prog.update(1)

                # check for termination
                if harn._check_termination():
                    raise StopIteration()

                # change learning rate (modified optimizer inplace)
                if harn.scheduler is None:
                    pass
                elif harn.scheduler.__class__.__name__ == 'ReduceLROnPlateau':
                    assert vali_metrics is not None, (
                        'must validate for ReduceLROnPlateau schedule')
                    harn.scheduler.step(vali_metrics['loss'])
                else:
                    harn.scheduler.step()

                harn.update_prog_description()
        except StopIteration:
            pass
        except Exception as ex:
            harn.log('An {} error occurred in the train loop'.format(type(ex)))
            harn._close_prog()
            raise
        harn.log('Best epochs / loss: {}'.format(ub.repr2(list(harn.stopping.memory), nl=1)))

    def run_epoch(harn, loader, tag, learn=False):
        """
        Evaluate the model on test / train / or validation data
        """
        # Use exponentially weighted or windowed moving averages across epochs
        iter_moving_metircs = harn._run_metrics[tag]
        # Use simple moving average within an epoch
        epoch_moving_metrics = metrics.CumMovingAve()

        # train batch
        if not harn.dry:
            # Flag if model is training (influences batch-norm / dropout)
            if harn.model.training != learn or learn:
                harn.model.train(learn)

        msg = harn.batch_msg({'loss': -1}, loader.batch_size)
        desc = tag + ' ' + msg
        position = (list(harn.loaders.keys()).index(tag) +
                    harn.main_prog.pos + 1)
        prog = tqdm.tqdm(desc=desc, total=len(loader),
                         disable=not harn.config['show_prog'],
                         position=position, leave=True, dynamic_ncols=True)
        prog.set_postfix({'wall': time.strftime('%H:%M') + ' ' + time.tzname[0]})

        for bx, (inputs, labels) in enumerate(loader):
            iter_idx = (harn.epoch * len(loader) + bx)

            # The dataset should return a inputs/target 2-tuple of lists.
            # In most cases each list will be length 1, unless there are
            # multiple input branches or multiple output branches.
            if not isinstance(inputs, (list, tuple)):
                inputs = [inputs]
            if not isinstance(labels, (list, tuple)):
                labels = [labels]

            if tuple(map(int, torch.__version__.split('.')[0:2])) < (0, 4):
                inputs = harn.xpu.to_xpu_var(*inputs, volatile=not learn)
                labels = harn.xpu.to_xpu_var(*labels, volatile=not learn)
            else:
                with torch.no_grad():
                    inputs = harn.xpu.to_xpu_var(*inputs)
                    labels = harn.xpu.to_xpu_var(*labels)

            # Core learning / backprop
            outputs, loss = harn.run_batch(inputs, labels, learn=learn)

            # Measure train accuracy and other informative metrics
            cur_metrics = harn._call_metric_hooks(outputs, labels, loss)

            # Accumulate measures
            epoch_moving_metrics.update(cur_metrics)
            iter_moving_metircs.update(cur_metrics)

            # display_train training info
            if harn.check_interval('display_' + tag, bx):
                ave_metrics = iter_moving_metircs.average()

                msg = harn.batch_msg({'loss': ave_metrics['loss']},
                                     loader.batch_size)
                prog.set_description(tag + ' ' + msg)

                for key, value in ave_metrics.items():
                    harn.log_value(tag + ' iter ' + key, value, iter_idx)

                prog.update(harn.intervals['display_' + tag])
                prog.set_postfix({'wall': time.strftime('%H:%M') + ' ' + time.tzname[0]})

            # Custom tensorboard output
            for _hook in harn._tensorboard_hooks:
                _hook(harn, tag, inputs, outputs, labels, bx, loader)

        prog.close()

        # Record a true average for the entire batch
        epoch_metrics = epoch_moving_metrics.average()

        for key, value in epoch_metrics.items():
            harn.log_value(tag + ' epoch ' + key, value, harn.epoch)

        return epoch_metrics

    def run_batch(harn, inputs, labels, learn=False):
        """
        Batch with weight updates

        https://github.com/meetshah1995/pytorch-semseg/blob/master/train.py
        """
        if harn.dry:
            # TODO: make general if model has output_size_for
            # output_shape = (label.shape[0],) + tuple(harn.single_output_shape)
            # output_shape = (label.shape[0], 3, label.shape[1], label.shape[2])
            # output = Variable(torch.rand(*output_shape))
            output = None
            base = float(sum(harn.current_lrs()))
            loss = Variable(base + torch.randn(1) * base)
            return output, loss

        # Run custom forward pass with loss computation, fallback to default
        if harn._custom_run_batch is None:
            outputs, loss = harn._default_run_batch(harn, inputs, labels)
        else:
            outputs, loss = harn._custom_run_batch(harn, inputs, labels)

        # Backprop and learn
        if learn:
            harn.optimizer.zero_grad()
            loss.backward()
            harn.optimizer.step()

        return outputs, loss

    def _default_run_batch(harn, harn_, inputs, labels):
        # What happens when there are multiple criterions?
        # How does hyperparam deal with that?

        # Forward prop through the model
        outputs = harn.model(*inputs)

        # Compute the loss
        # TODO: how do we support multiple losses for deep supervision?
        # AND: overwrite this
        if len(labels) == 1:
            labels = labels[0]

        try:
            loss = harn.criterion(outputs, labels)
        except Exception:
            print('may need to make a custom batch runner with set_batch_runner')
            raise
        return outputs, loss

    def check_interval(harn, tag, idx):
        """
        check if its time to do something that happens every few iterations
        """
        return (idx + 1) % harn.intervals[tag] == 0

    def set_batch_runner(harn, func):
        """
        Define custom logic to do a forward pass with loss computation.

        Args:
            func : accepts 3 args (harn, inputs, labels). The inputs and labels
            will be lists of Variables
        """
        harn._custom_run_batch = func
        return func

    def add_metric_hook(harn, hook):
        """
        Adds a hook that should take arguments
        (harn, output, label) and return a dictionary of scalar metrics
        """
        harn._metric_hooks.append(hook)

    def _call_metric_hooks(harn, output, label, loss):
        loss_sum = loss.data.sum()
        inf = float("inf")
        if loss_sum == inf or loss_sum == -inf:
            harn.log("WARNING: received an inf loss, setting loss value to 0")
            loss_value = 0
        else:
            loss_value = loss_sum

        metrics_dict = {
            'loss': loss_value,
        }
        if not harn.dry:
            for hook in harn._metric_hooks:
                _custom_dict = hook(harn, output, label)
                isect = set(_custom_dict).intersection(set(metrics_dict))
                if isect:
                    raise Exception('Conflicting metric hooks: {}'.format(isect))
                metrics_dict.update(_custom_dict)
        return metrics_dict

    def batch_msg(harn, metric_dict, batch_size):
        bs = 'x{}'.format(batch_size)
        metric_parts = ['{}:{:.3f}'.format(k, v) for k, v in metric_dict.items()]
        msg = ' │ ' .join([bs] + metric_parts) + ' │'
        return msg

    @property
    def snapshot_dpath(harn):
        return join(harn.train_dpath, 'torch_snapshots')

    def prev_snapshots(harn):
        ub.ensuredir(harn.snapshot_dpath)
        prev_states = sorted(glob.glob(join(harn.snapshot_dpath, '_epoch_*.pt')))
        return prev_states

    def load_snapshot(harn, load_path):
        """
        Sets the harness to its state just after an epoch finished
        """
        snapshot = harn.xpu.load(load_path)
        harn.log('Loading previous state: {}'.format(load_path))
        # the snapshot holds the previous epoch, so add one to move to current
        harn.epoch = snapshot['epoch'] + 1
        harn.model.load_state_dict(snapshot['model_state_dict'])
        if 'optimizer_state_dict' in snapshot:
            harn.optimizer.load_state_dict(snapshot['optimizer_state_dict'])
        harn.log('Resuming training...')

    def save_snapshot(harn):
        # save snapshot
        ub.ensuredir(harn.snapshot_dpath)
        save_path = join(harn.snapshot_dpath, '_epoch_{:08d}.pt'.format(harn.epoch))
        if harn.dry:
            # harn.log('VERY VERY VERY VERY VERY LONG MESSAGE ' * 5)
            harn.log('Would save snapshot to {}'.format(save_path))
            pass
        else:
            train = harn.datasets['train']
            if hasattr(train, 'dataset_metadata'):
                dataset_metadata = train.dataset_metadata()
            else:
                dataset_metadata = None

            # TODO: should we split the optimizer state into a different file?
            snapshot = {
                'model_class_name': harn.model.__class__.__name__,
                'dataset_metadata': dataset_metadata,
                'epoch': harn.epoch,
                'model_state_dict': harn.model.state_dict(),
                'optimizer_state_dict': harn.optimizer.state_dict(),
            }
            torch.save(snapshot, save_path)
            # harn.log('Snapshot saved to {}'.format(save_path))

    def log(harn, msg):
        # if harn.main_prog:
        #     harn.main_prog.write(str(msg), file=sys.stdout)
        # else:
        print(msg)
        # for line in msg.split('\n'):
        #     print(line)
        # print('line = {!r}'.format(line))

    def log_value(harn, key, value, n_iter):
        # if False:
        #     print('{}={} @ {}'.format(key, value, n_iter))
        if harn.tlogger:
            harn.tlogger.log_value(key, value, n_iter)

    def log_histogram(harn, key, value, n_iter):
        # if False:
        #     print('{}={} @ {}'.format(key, value, n_iter))
        if harn.tlogger:
            harn.tlogger.log_histogram(key, value, n_iter)

    def log_images(harn, key, value, n_iter):
        # if False:
        #     print('{}={} @ {}'.format(key, value, n_iter))
        if harn.tlogger:
            harn.tlogger.log_images(key, value, n_iter)

    def _check_termination(harn):
        if harn.epoch >= harn.config['max_iter']:
            harn._close_prog()
            harn.log('Maximum harn.epoch reached, terminating ...')
            return True
        if harn.stopping.is_done():
            harn._close_prog()
            harn.log('Validation set is not improving, terminating ...')
            return True
        return False

    def _close_prog(harn):
        harn.main_prog.close()
        harn.main_prog = None
        sys.stdout.write('\n\n\n\n')  # fixes progress bar formatting

    # def _tensorboard_inputs(harn, inputs, iter_idx, tag):
    #     """
    #         >>> tensor = torch.rand(4, 3, 42, 420)
    #         >>> inputs = [tensor]
    #     """
    #     # print("LOG IMAGES")
    #     if False:
    #         # todo: add dataset / task hook for extracting images
    #         pass
    #     else:
    #         # default_convert = im_loaders.rgb_tensor_to_imgs
    #         def default_convert(tensor):
    #             if len(tensor.shape) == 4:
    #                 arr = tensor.data.cpu().numpy().transpose(0, 2, 3, 1)
    #                 aux = []
    #                 if arr.shape[3] == 1:
    #                     ims = arr[:, :, :, 0]
    #                 elif arr.shape[3] == 3:
    #                     ims = arr
    #                 else:
    #                     ims = arr[:, :, :, 0:3]
    #                     aux = [arr[:, :, :, c] for c in range(3, arr.shape[3])]
    #                 imaux = [ims] + aux
    #                 return imaux
    #             elif len(tensor.shape) == 3:
    #                 return [tensor.data.cpu().numpy()]
    #             else:
    #                 raise NotImplementedError(str(tensor.shape))

    #     # def single_image_render(im):
    #     #     import plottool as pt
    #     #     with pt.RenderingContext() as render:
    #     #         pt.figure(fnum=1, pnum=(1, 1, 1), doclf=True)
    #     #         if len(im.shape) == 3:
    #     #             im = im[:, :, ::-1]
    #     #         print(im.shape)
    #     #         print(im.dtype)
    #     #         pt.imshow(im, norm=True, cmap='viridis', data_colorbar=True,
    #     #                   ax=pt.gca())
    #     #     out_im = render.image[:, :, ::-1]
    #     #     return out_im

    #     for i, input_tensor in enumerate(inputs):
    #         # print('\n\n')
    #         if len(input_tensor.shape) == 4:
    #             # print('input_tensor.shape = {!r}'.format(input_tensor.shape))
    #             imaux = default_convert(input_tensor[0:2])
    #             for c, images in enumerate(imaux):
    #                 extent = (images.max() - images.min())
    #                 images = (images - images.min()) / max(extent, 1e-6)
    #                 # print('images.shape = {!r}'.format(images.shape))
    #                 # images = [single_image_render(im) for im in images]
    #                 # for im in images:
    #                 #     print('im.shape = {!r}'.format(im.shape))
    #                 #     print('im.dtype = {!r}'.format(im.dtype))
    #                 #     print(im.max())
    #                 #     print(im.min())
    #                 harn.log_images(tag + '-c{c}-in{i}-iter{x}'.format(i=i, c=c, x=iter_idx), images, iter_idx)
    #         else:
    #             print('\nSKIPPING INPUT VISUALIZE\n')

    # def _tensorboard_extra(harn, inputs, output, label, tag, iter_idx, loader):
    #     # https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/04-utils/tensorboard/main.py#L83-L105
    #     # print("\nTENSORBOARD EXTRAS\n")
    #     # loader.dataset
    #     # print('\n\ninputs.shape = {!r}'.format(inputs[0].shape))

    #     if iter_idx == 0:
    #         harn._tensorboard_inputs(inputs, iter_idx, tag)

    #     if iter_idx % 1000 == 0:
    #         true = label.data.cpu().numpy().ravel()
    #         n_classes = harn.hyper.other['n_classes']
    #         counts = np.bincount(true, minlength=n_classes)
    #         bins = list(range(n_classes + 1))
    #         harn.log_histogram(tag + '-true-', (bins, counts), iter_idx)

    #     if iter_idx % 1000 == 0:
    #         if not harn.dry:
    #             n_classes = harn.hyper.other['n_classes']
    #             preds = output.max(dim=1)[1].data.cpu().numpy().ravel()
    #             counts = np.bincount(preds, minlength=n_classes)
    #             bins = list(range(n_classes + 1))
    #             harn.log_histogram(tag + '-pred-', (bins, counts), iter_idx)
    #             # import torch.nn.functional as F
    #             # probs = torch.exp(F.log_softmax(output, dim=1))


def get_snapshot(train_dpath, epoch='recent'):
    """
    Get a path to a particular epoch or the most recent one
    """
    import parse
    snapshots = sorted(glob.glob(train_dpath + '/*/_epoch_*.pt'))
    if epoch is None:
        epoch = 'recent'

    if epoch == 'recent':
        load_path = snapshots[-1]
    else:
        snapshot_nums = [parse.parse('{}_epoch_{num:d}.pt', path).named['num']
                         for path in snapshots]
        load_path = dict(zip(snapshot_nums, snapshots))[epoch]
    return load_path


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m clab.torch.fit_harness
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
