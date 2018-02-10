# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import sys
import shutil
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
from clab import metrics
from clab import xpu_device
from clab import monitor
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


class grad_context(object):
    """
    TODO: PR to pytorch

    TODO: make a torch contrib module
    torch_contrib

    Possibly dynamically inject into torch?
    """
    def __init__(self, flag):
        if tuple(map(int, torch.__version__.split('.')[0:2])) < (0, 4):
            self.prev = None
            self.flag = flag
        else:
            self.prev = torch.is_grad_enabled()
            self.flag = flag

    def __enter__(self):
        if self.prev is not None:
            torch.set_grad_enabled(self.flag)

    def __exit__(self, *args):
        if self.prev is not None:
            torch.set_grad_enabled(self.prev)
            return False


class FitHarness(object):
    def __init__(harn, datasets, batch_size=None, hyper=None, xpu=None,
                 loaders=None, train_dpath='./train', dry=False,
                 max_keys=[], min_keys=['loss']):

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
            harn.xpu = xpu_device.XPU.cast('cpu')
        else:
            harn.xpu = xpu_device.XPU.cast(xpu)

        harn.datasets = datasets
        if loaders is None:
            harn.loaders = harn._setup_loaders(datasets, batch_size)
        else:
            harn.loaders = loaders

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
        # harn.monitor = monitor.EarlyStop(patience=30)
        harn.monitor = monitor.Monitor(min_keys, max_keys, patience=40,
                                       smoothing=.6)

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
        from clab import folder_structure
        structure = folder_structure.FolderStructure(
            workdir=workdir, hyper=harn.hyper, datasets=harn.datasets,
        )
        train_info = structure.setup_dpath(**kwargs)
        harn.train_dpath = train_info['train_dpath']
        harn.link_dpath = train_info['link_dpath']
        return harn.train_dpath

    def _setup_loaders(harn, datasets, batch_size):
        """ automatic loader setup. Can be overriden by specifying loaders """
        data_kw = {'batch_size': batch_size}
        if harn.xpu.is_gpu():
            data_kw.update({'num_workers': 6, 'pin_memory': True})

        tags = ['train', 'vali', 'test']

        loaders = ub.odict()
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
            loaders[tag] = loader
        return loaders

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

            harn.model = harn.hyper.make_model()
            harn.initializer = harn.hyper.make_initializer()

            harn.log('Mounting {} model on {}'.format(model_name, harn.xpu))
            harn.model = harn.xpu.mount(harn.model)

            n_params = number_of_parameters(harn.model)
            harn.log('Model has {!r} parameters'.format(n_params))

            # more than one criterion? Wrap it in a single criterion OR
            # specify a custom batch runner.
            if harn.hyper.criterion_cls:
                harn.criterion = harn.hyper.criterion_cls(
                    **harn.hyper.criterion_params)
                harn.log('Move {} model to {}'.format(harn.criterion, harn.xpu))
                harn.criterion = harn.xpu.move(harn.criterion)
            else:
                pass

            harn.log('Make optimizer')
            harn.optimizer = harn.hyper.make_optimizer(harn.model.parameters())

            if harn.hyper.scheduler_cls:
                harn.log('Make scheduler')
                harn.scheduler = harn.hyper.make_scheduler(harn.optimizer)

            harn.log('There are {} existing snapshots'.format(len(prev_states)))
            if prev_states and not ub.argflag('--reset'):
                harn.log('Loading previous states')
                harn.load_snapshot(prev_states[-1])

                for i, group in enumerate(harn.optimizer.param_groups):
                    if 'initial_lr' not in group:
                        raise KeyError("param 'initial_lr' is not specified "
                                       "in param_groups[{}] when resuming an optimizer".format(i))

            else:
                harn.log('Initializing new model')
                if harn.initializer.__class__.__name__ == 'LSUV':
                    #hack LSUV needs a batch of data to run
                    with grad_context(False):
                        import utool
                        utool.embed()
                        loader = harn.loaders['train']
                        inputs, labels = next(iter(loader))
                        input = harn.xpu.variable(inputs[0])
                        harn.initializer(harn.model, input)
                else:
                    harn.initializer(harn.model)
                if not harn.dry:
                    for group in harn.optimizer.param_groups:
                        group.setdefault('initial_lr', group['lr'])

            harn.log('Snapshots will save to harn.snapshot_dpath = {!r}'.format(harn.snapshot_dpath))

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
        desc = 'epoch lr:{} │ {}'.format(lr_str, harn.monitor.message())
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
                if harn.monitor:
                    raise ValueError('need a validataion dataset to use early monitor')
                if harn.scheduler.__class__.__name__ == 'ReduceLROnPlateau':
                    raise ValueError('need a validataion dataset to use ReduceLROnPlateau')

        # Keep track of moving metric averages across epochs
        harn._run_metrics = {
            tag: metrics.WindowedMovingAve(window=len(loader))
            for tag, loader in harn.loaders.items()
        }

        try:
            for harn.epoch in it.count(harn.epoch):

                harn.log_value('epoch lr', np.mean(list(harn.current_lrs())),
                               harn.epoch)

                # Run training epoch
                harn.run_epoch(train_loader, tag='train', learn=True)

                # Run validation epoch
                vali_metrics = None
                improved = False
                if vali_loader:
                    if harn.check_interval('vali', harn.epoch):
                        vali_metrics = harn.run_epoch(
                            vali_loader, tag='vali', learn=False)
                        improved = harn.monitor.update(harn.epoch,
                                                       vali_metrics)

                    harn.update_prog_description()

                # Run test epoch
                if test_loader:
                    if harn.check_interval('test', harn.epoch):
                        harn.run_epoch(test_loader, tag='test', learn=False)

                # if harn.check_interval('snapshot', harn.epoch):
                save_path = harn.save_snapshot()
                if improved:
                    if save_path:
                        # Copy the best snapshot the the main directory
                        shutil.copy2(save_path, join(harn.train_dpath,
                                     'best_snapshot.pt'))

                harn.main_prog.update(1)

                # check for termination
                if harn._check_termination():
                    raise StopIteration()

                # change learning rate (modified optimizer inplace)
                harn._step_scheduler(improved)

                harn.update_prog_description()
        except StopIteration:
            pass
        except Exception as ex:
            harn.log('An {} error occurred in the train loop'.format(type(ex)))
            harn._close_prog()
            raise

        harn.log('\n\n\n')
        harn.log('Training completed')
        harn.log('Current LRs: {}'.format(harn.current_lrs()))
        # harn.log('Best epochs / loss: {}'.format(
        #     ub.repr2(list(harn.monitor.memory), nl=1)))
        harn.log('Exiting harness.')

    def _step_scheduler(harn, improved):
        """
        Helper function to change the learning rate that handles the way that
        different schedulers might be used.
        """
        if harn.scheduler is None:
            pass
        elif harn.scheduler.__class__.__name__ == 'ReduceLROnPlateau':
            assert improved is not None, 'must validate for ReduceLROnPlateau schedule'
            # assert vali_metrics is not None, (
            #     'must validate for ReduceLROnPlateau schedule')

            # old_lrs = set(harn.current_lrs())
            # Feed reduce on plateau dummy data from the monitor
            # harn.scheduler.step(vali_metrics['loss'])

            # harn.scheduler.step(vali_metrics['loss'], epoch=harn.epoch)
            def hack_lr_step(self, improved, epoch=None):
                if epoch is None:
                    epoch = self.last_epoch = self.last_epoch + 1
                self.last_epoch = epoch

                if improved:
                    self.num_bad_epochs = 0
                else:
                    self.num_bad_epochs += 1

                if self.in_cooldown:
                    self.cooldown_counter -= 1
                    self.num_bad_epochs = 0  # ignore any bad epochs in cooldown

                if self.num_bad_epochs > self.patience:
                    self._reduce_lr(epoch)
                    self.cooldown_counter = self.cooldown
                    self.num_bad_epochs = 0

                    # TODO: Make a pytorch PR where there is a callback on
                    # lr_reduction.
                    # The scheduler has stepped, we should now backtrack the
                    # weights to the previous best state
                    backtrack = False
                    if backtrack:
                        harn.backtrack_weights(harn.monitor.best_epoch)

            # # Hack to determine if the RLROP scheduler stepped
            hack_lr_step(harn.scheduler, improved)

            # new_lrs = set(harn.current_lrs())
            # if old_lrs != new_lrs:
            #     # The scheduler has stepped, we should now backtrack the
            #     # weights to the previous best state
            #     harn.backtrack_weights(harn.monitor.best_epoch)
        else:
            harn.scheduler.step()

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

        with grad_context(learn):
            for bx, (inputs, labels) in enumerate(loader):
                # iter_idx = (harn.epoch * len(loader) + bx)

                # The dataset should return a inputs/target 2-tuple of lists.
                # In most cases each list will be length 1, unless there are
                # multiple input branches or multiple output branches.
                if not isinstance(inputs, (list, tuple)):
                    inputs = [inputs]
                if not isinstance(labels, (list, tuple)):
                    labels = [labels]

                # note volatile is depricated
                inputs = list(harn.xpu.variables(*inputs))
                labels = list(harn.xpu.variables(*labels))

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

                    # for key, value in ave_metrics.items():
                    #     harn.log_value(tag + ' iter ' + key, value, iter_idx)

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

    def backtrack_weights(harn, epoch):
        """
        Reset the weights to a previous good state
        """
        load_path = join(harn.snapshot_dpath,
                         '_epoch_{:08d}.pt'.format(epoch))
        snapshot = harn.xpu.load(load_path)

        print('\n\n\n\n')
        harn.log('Backtracking to weights from previous state: {}'.format(load_path))
        # harn.log('Backtracking to weights from previous state: {}'.format(load_path))
        # only load the model state, the optimizer and other state items stay
        # as is.
        harn.model.load_state_dict(snapshot['model_state_dict'])

    def load_snapshot(harn, load_path):
        """
        Sets the harness to its state just after an epoch finished
        """
        snapshot = harn.xpu.load(load_path)
        harn.log('Loading previous state: {}'.format(load_path))
        # the snapshot holds the previous epoch, so add one to move to current
        harn.epoch = snapshot['epoch'] + 1
        harn.model.load_state_dict(snapshot['model_state_dict'])

        if 'monitor_state_dict' in snapshot:
            harn.monitor.load_state_dict(snapshot['monitor_state_dict'])

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
                'monitor_state_dict': harn.monitor.state_dict(),
            }
            torch.save(snapshot, save_path)
            return save_path
            # harn.log('Snapshot saved to {}'.format(save_path))

    def log(harn, msg):
        # if harn.main_prog:
        #     harn.main_prog.write(str(msg), file=sys.stdout)
        # else:
        print(msg)
        # for line in msg.split('\n'):
        #     print(line)
        # print('line = {!r}'.format(line))

    def debug(harn, msg):
        # TODO: add in a file logging mechanism here
        # print(msg)
        pass

    def log_value(harn, key, value, n_iter):
        if harn.tlogger:
            harn.tlogger.log_value(key, value, n_iter)

    def log_histogram(harn, key, value, n_iter):
        if harn.tlogger:
            harn.tlogger.log_histogram(key, value, n_iter)

    def log_images(harn, key, value, n_iter):
        if harn.tlogger:
            harn.tlogger.log_images(key, value, n_iter)

    def _check_termination(harn):
        if harn.epoch >= harn.config['max_iter']:
            harn._close_prog()
            harn.log('Maximum harn.epoch reached, terminating ...')
            return True
        if harn.monitor.is_done():
            harn._close_prog()
            harn.log('Validation set is not improving, terminating ...')
            return True
        return False

    def _close_prog(harn):
        if harn.main_prog is not None:
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
        python -m clab.fit_harness
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
