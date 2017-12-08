# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
# import numpy as np
import collections
import sys
import glob
import numpy as np
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
from clab.torch import nnio
from clab import util  # NOQA
from clab import getLogger
logger = getLogger(__name__)
print = util.protect_print(logger.info)
# print = logger.info


def demo():
    """
    python -m clab.live.fit_harn2 demo

    Example:
        >>> from clab.live.fit_harn2 import *
        >>> demo()
    """
    from clab.torch import hyperparams
    from clab.torch import layers
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])
    n_classes = 2

    datakw = dict(
        image_size=(1, 16, 16),
        num_classes=n_classes,
        transform=transform,
    )

    SSEG = 1
    if SSEG:
        datakw['image_size'] = (1, 200, 200)
        def _gt_as_sseg_mask(gt):
            # transform groundtruth into a random semantic segmentation mask
            return (torch.rand(datakw['image_size'][1:]) * n_classes).long()
        datakw['target_transform'] = _gt_as_sseg_mask
        factor = 1
    else:
        factor = 200

    datasets = {
        'train': torchvision.datasets.FakeData(size=10 * factor, random_offset=0, **datakw),
        'vali': torchvision.datasets.FakeData(size=5 * factor, random_offset=110000, **datakw),
        'test': torchvision.datasets.FakeData(size=2 * factor, random_offset=1110000, **datakw),
    }
    class DummyModel(torch.nn.Module):
        def __init__(self, n_classes):
            super(DummyModel, self).__init__()
            self.seq = torch.nn.Sequential(*[
                layers.Flatten(),
                torch.nn.Linear(int(np.prod(datakw['image_size'])), 10),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(10, n_classes),
            ])
            self.n_classes = n_classes

        def forward(self, inputs):
            if isinstance(inputs, (tuple, list)):
                assert len(inputs) == 1
                inputs = inputs[0]
            return self.seq(inputs)

    if SSEG:
        from clab.torch.models import unet
        from clab.torch import criterions
        model = unet.UNet(in_channels=1, n_classes=n_classes, feature_scale=64)
        hyper = hyperparams.HyperParams(
            criterion_cls=criterions.CrossEntropyLoss2D,
        )
    else:
        model = DummyModel(n_classes)
        hyper = hyperparams.HyperParams(
            criterion_cls=torch.nn.CrossEntropyLoss,
        )

    xpu = xpu_device.XPU()

    # hack
    def compute_loss(harn, outputs, labels):
        # Compute the loss
        if isinstance(outputs, list):
            outputs = outputs[0]
        if isinstance(labels, list):
            target = labels[0]
        if SSEG:
            loss = harn.criterion(outputs, target)
        else:
            target = labels[0].long()
            pred = torch.nn.functional.log_softmax(outputs, dim=1)
            loss = torch.nn.functional.nll_loss(pred, target)
        return loss

    train_dpath = ub.ensure_app_cache_dir('clab/demo/fit_harness')
    ub.delete(train_dpath)

    dry = 0
    harn = FitHarness(
        model=model, datasets=datasets, xpu=xpu, hyper=hyper,
        train_dpath=train_dpath, dry=dry,
    )
    harn.compute_loss = compute_loss
    harn.run()

    if not dry:
        # reload and continue training the model
        # stopping criterion should trigger immediately
        harn = FitHarness(
            model=model, datasets=datasets, xpu=xpu, hyper=hyper,
            train_dpath=train_dpath, dry=dry,
        )
        harn.compute_loss = compute_loss
        harn.run()


class EarlyStop(object):
    """

    TODO: Poisson based early stop

    Example:
        >>> early_stop = EarlyStop()
        >>> early_stop.update(1, .5)
        >>> early_stop.update(2, .4)
        >>> early_stop.update(3, .6)
        >>> early_stop.update(4, .3)
        >>> early_stop.update(5, .31)
        >>> early_stop.update(6, .2)
        >>> early_stop.best_epochs()
        >>> print('Best epochs / loss: {}'.format(ub.repr2(list(early_stop.memory), nl=1, precision=6)))
    """
    def __init__(early_stop, patience=10):
        # import sortedcontainers
        # store tuples of (loss, epoch)
        early_stop.memory = collections.deque(maxlen=3)
        # util.SortedQueue(maxsize=3)

        early_stop.prev_epoch = None
        early_stop.prev_loss = None

        early_stop.best_epoch = None
        early_stop.best_loss = None

        early_stop.patience = patience
        early_stop.n_bad_epochs = 0

    def update(early_stop, epoch, loss):
        early_stop.prev_epoch = epoch
        early_stop.prev_loss = loss

        # Dont allow overfitting epochs to be recorded as top-epochs
        if len(early_stop.memory) == 0:
            early_stop.memory.appendleft((epoch, loss))
        elif loss < early_stop.memory[0][1]:
            early_stop.memory.appendleft((epoch, loss))

        # if len(early_stop.memory) == early_stop.memory.maxsize:
        #     if loss < early_stop.memory.peek()[1]:
        #         [epoch] = loss
        # early_stop.memory[epoch] = loss

        if early_stop.best_loss is None or loss < early_stop.best_loss:
            early_stop.best_loss = loss
            early_stop.best_epoch = epoch
            early_stop.n_bad_epochs = 0
        else:
            early_stop.n_bad_epochs += 1

    def is_improved(early_stop):
        """
        returns True if the last update improved the validation loss
        """
        return early_stop.n_bad_epochs == 0

    def is_done(early_stop):
        return early_stop.n_bad_epochs >= early_stop.patience

    def best_epochs(early_stop):
        return [epoch for epoch, loss in early_stop.memory]

    def message(early_stop):
        if early_stop.prev_epoch is None:
            return 'vloss is unevaluated'
        # if early_stop.is_improved():
            # message = 'vloss: {:.4f} (new_best)'.format(early_stop.best_loss)
        message = 'vloss: {:.4f} (n_bad_epochs={:2d}, best={:.4f})'.format(
            early_stop.prev_loss, early_stop.n_bad_epochs,
            early_stop.best_loss,
        )
        if early_stop.n_bad_epochs <= int(early_stop.patience * .25):
            message = ub.color_text(message, 'green')
        elif early_stop.n_bad_epochs >= int(early_stop.patience * .75):
            message = ub.color_text(message, 'red')
        else:
            message = ub.color_text(message, 'yellow')
        return message


class FitHarness(object):
    def __init__(harn, model, datasets, batch_size=4,
                 criterion_cls='cross_entropy', hyper=None, xpu=None,
                 train_dpath=None, dry=False):

        harn.dry = dry
        if harn.dry:
            train_dpath = ub.ensure_app_cache_dir('clab/dry')
            ub.delete(train_dpath)
            train_dpath = ub.ensure_app_cache_dir('clab/dry')

        harn.dry = dry
        harn.train_dpath = train_dpath
        harn.snapshot_dpath = ub.ensuredir((harn.train_dpath, 'torch_snapshots'))

        if harn.dry:
            harn.xpu = xpu_device.XPU(None)
        else:
            harn.xpu = xpu_device.XPU(xpu)

        data_kw = {'batch_size': batch_size}
        if harn.xpu.is_gpu():
            num_workers = 0 if ub.argflag('--serial') else 6
            pin_memory = False if ub.argflag('--nopin') else True
            data_kw.update({'num_workers': num_workers, 'pin_memory': pin_memory})
            # data_kw.update({'num_workers': 0, 'pin_memory': False})

        harn.loaders = ub.odict()
        harn.datasets = datasets
        assert set(harn.datasets.keys()).issubset({'train', 'vali', 'test'})
        for tag in ['train', 'vali', 'test']:
            dset = harn.datasets.get(tag, None)
            if dset:
                shuffle = tag == 'train'
                data_kw_ = data_kw.copy()
                if tag != 'train':
                    data_kw_['batch_size'] = max(batch_size // 4, 1)
                loader = torch.utils.data.DataLoader(dset, shuffle=shuffle,
                                                     **data_kw_)
                harn.loaders[tag] = loader

        harn.model = model

        harn.hyper = hyper

        harn.lr_scheduler = hyper.scheduler_cls(**hyper.scheduler_params)
        harn.criterion_cls = hyper.criterion_cls
        harn.optimizer_cls = hyper.optimizer_cls

        harn.criterion_params = hyper.criterion_params
        harn.optimizer_params = hyper.optimizer_params

        harn._metric_hooks = []
        harn._run_metrics = None

        harn._epoch_callbacks = []
        harn._iter_callbacks = []

        harn.early_stop = EarlyStop()

        harn.intervals = {
            'display_train': 1,
            'display_vali': 1,
            'display_test': 1,

            'vali': 1,
            'test': 1,

            'snapshot': 1,
        }
        harn.config = {
            'max_iter': 500,
        }
        harn.epoch = 0

        harn.tlogger = None
        harn.prog = None

    def log(harn, msg):
        if harn.prog is not None:
            harn.prog.write(msg)
        else:
            print(msg)

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

    def initialize_training(harn):
        harn.xpu.set_as_default()

        if tensorboard_logger:
            train_base = os.path.dirname(harn.train_dpath)
            harn.log('dont forget to start: tensorboard --logdir ' + train_base)
            harn.log('Initializing tensorboard')
            harn.tlogger = tensorboard_logger.Logger(harn.train_dpath,
                                                     flush_secs=2)
            # tensorboard_logger.configure(harn.train_dpath, flush_secs=2)

        if harn.dry:
            harn.log('Dry run of training harness. xpu={}'.format(harn.xpu))
            harn.optimizer = None
        else:
            prev_states = harn.prev_snapshots()

            model_name = harn.model.__class__.__name__
            harn.log('There are {} existing snapshots'.format(len(prev_states)))

            harn.log('Fitting {} model on {}'.format(model_name, harn.xpu))
            # harn.xpu.to_xpu(harn.model)

            weight = harn.criterion_params.get('weight', None)
            if weight is not None:
                harn.log('Casting weights')
                weight = torch.FloatTensor(harn.criterion_params['weight'])
                # weight = harn.xpu.to_xpu(weight)
                harn.criterion_params['weight'] = weight

            harn.log('Criterion: {}'.format(harn.criterion_cls.__name__))
            harn.criterion = harn.criterion_cls(**harn.criterion_params)

            # TODO: port this to main test harness and have hyperparams know to
            # convert tensors to lists before they use repr2
            # if hasattr(harn, 'criterion2'):
            #     harn.criterion2 = harn.xpu.to_xpu(harn.criterion2)

            # harn.log('Optimizer: {}'.format(harn.optimizer_cls.__name__))
            # if harn.lr_scheduler:
            #     lr = harn.lr_scheduler(harn.epoch)
            #     harn.optimizer = harn.optimizer_cls(
            #         harn.model.parameters(), lr=lr, **harn.optimizer_params)
            # else:
            #     harn.optimizer = harn.optimizer_cls(
            #         harn.model.parameters(), **harn.optimizer_params)

            if prev_states:
                harn.load_snapshot(prev_states[-1])
                # TODO: Load saved params into the optimizer?
            else:
                print('harn.snapshot_dpath = {!r}'.format(harn.snapshot_dpath))

    def move_model_to_xpu(harn):
        if not harn.dry:
            harn.log('Moving model and criterion to {}'.format(harn.xpu))
            harn.xpu.to_xpu(harn.model)
            harn.criterion = harn.xpu.to_xpu(harn.criterion)
            if hasattr(harn, 'criterion2'):
                harn.criterion2 = harn.xpu.to_xpu(harn.criterion2)

            harn.log('Optimizer: {}'.format(harn.optimizer_cls.__name__))
            if harn.lr_scheduler:
                lr = harn.lr_scheduler(harn.epoch)
                harn.optimizer = harn.optimizer_cls(
                    harn.model.parameters(), lr=lr, **harn.optimizer_params)
            else:
                harn.optimizer = harn.optimizer_cls(
                    harn.model.parameters(), **harn.optimizer_params)

    def run(harn):

        harn.log('Begin training')

        harn.initialize_training()

        if harn.early_stop.is_done():
            harn.log('The early stopping criterion already triggered')
            harn.log('Best epochs / loss: {}'.format(ub.repr2(list(harn.early_stop.memory), nl=1)))
            return
        if harn.epoch >= harn.config['max_iter']:
            harn.log('Maximum harn.epoch already reached.')
            return

        harn.move_model_to_xpu()

        # train loop
        import tqdm

        show_prog = False

        harn.prog = tqdm.tqdm(desc='epoch', total=harn.config['max_iter'],
                                                disable=not show_prog,
                              dynamic_ncols=True,
                              initial=harn.epoch)

        if harn.lr_scheduler:
            lr = harn.lr_scheduler(harn.epoch, harn.optimizer)
            harn.prog.set_description('epoch lr:{} │'.format(lr))

        train_loader = harn.loaders['train']
        vali_loader  = harn.loaders.get('vali', None)
        test_loader  = harn.loaders.get('test', None)

        # Keep track of moving metric averages across epochs
        harn._run_metrics = {
            tag: metrics.WindowedMovingAve(window=len(loader))
            for tag, loader in harn.loaders.items() if loader
        }

        harn._subprogs = {}
        def _reset_subprog(tag):
            loader = harn.loaders.get(tag, None)
            if loader:
                msg = harn.batch_msg({'loss': -1}, loader.batch_size)
                desc = tag + ' ' + msg
                harn._subprogs[tag] = tqdm.tqdm(desc=desc, total=len(loader),
                                                disable=not show_prog,
                                                dynamic_ncols=True)

        def _close_prog():
            harn._subprogs = None
            harn.prog.close()
            harn.prog = None
            sys.stdout.write('\n\n\n\n')  # fixes progress bar formatting

        try:
            for harn.epoch in it.count(harn.epoch):

                # Make subprogress bars for each epoch
                for tag in ['train', 'vali', 'test']:
                    _reset_subprog(tag)

                # change learning rate (modified optimizer inplace)
                if harn.lr_scheduler:
                    lr = harn.lr_scheduler(harn.epoch, harn.optimizer)
                harn.prog.set_description('epoch lr:{:.2g} │ {}'.format(lr, harn.early_stop.message()))

                # Run training epoch
                harn.run_epoch(train_loader, tag='train', learn=True)

                # Run validation epoch
                if vali_loader:
                    if (harn.epoch + 1) % harn.intervals['vali'] == 0:
                        vali_metrics = harn.run_epoch(
                            vali_loader, tag='vali', learn=False)
                        harn.early_stop.update(harn.epoch, vali_metrics['loss'])

                    harn.prog.set_description('epoch lr:{:.2g} │ {}'.format(lr, harn.early_stop.message()))

                # Run test epoch
                if test_loader:
                    if (harn.epoch + 1) % harn.intervals['test'] == 0:
                        harn.run_epoch(test_loader, tag='test', learn=False)

                for p in harn._subprogs.values():
                    p.close()

                # if harn.early_stop.is_improved():
                if (harn.epoch + 1) % harn.intervals['snapshot'] == 0:
                    harn.save_snapshot()

                harn.prog.update()

                # check for termination
                if harn.epoch >= harn.config['max_iter']:
                    _close_prog()
                    harn.log('Maximum harn.epoch reached, terminating ...')
                    break
                if harn.early_stop.is_done():
                    _close_prog()
                    harn.log('Validation set is not improving, terminating ...')
                    break
        except Exception as ex:
            harn.log('An {} error occurred in the train loop'.format(type(ex)))
            _close_prog()
            raise

        harn.log('Best epochs / loss: {}'.format(ub.repr2(list(harn.early_stop.memory), nl=1)))

    def run_epoch(harn, loader, tag, learn=False):
        # Use exponentially weighted or windowed moving averages across epochs
        iter_moving_metircs = harn._run_metrics[tag]
        # Use simple moving average within an epoch
        epoch_moving_metrics = metrics.CumMovingAve()

        # train batch
        if not harn.dry:
            # Flag if model is training (influences batch-norm / dropout)
            if harn.model.training != learn or learn:
                harn.model.train(learn)

        display_interval = harn.intervals['display_' + tag]

        prog = harn._subprogs[tag]

        for bx, (inputs, labels) in enumerate(loader):
            iter_idx = (harn.epoch * len(loader) + bx)

            # The dataset should return a inputs/target 2-tuple of lists.
            # In most cases each list will be length 1, unless there are
            # multiple input branches or multiple output branches.
            if not isinstance(inputs, (list, tuple)):
                inputs = [inputs]
            if not isinstance(labels, (list, tuple)):
                labels = [labels]

            inputs = harn.xpu.to_xpu_var(*inputs)
            labels = harn.xpu.to_xpu_var(*labels)

            # Core learning / backprop
            outputs, loss = harn.run_batch(inputs, labels, learn=learn)

            # Measure train accuracy and other informative metrics
            cur_metrics = harn._call_metric_hooks(outputs, labels, loss)

            # if 1:
            #     harn._tensorboard_extra(inputs, outputs, labels, tag,
            #                             iter_idx, loader)

            # Accumulate measures
            epoch_moving_metrics.update(cur_metrics)
            iter_moving_metircs.update(cur_metrics)

            # display_train training info
            if (bx + 1) % display_interval == 0:
                import utool
                utool.embed()
                ave_metrics = iter_moving_metircs.average()

                msg = harn.batch_msg({'loss': ave_metrics['loss']},
                                     loader.batch_size)
                prog.set_description(tag + ' ' + msg)

                for key, value in ave_metrics.items():
                    # harn.log_value(tag + ' ' + key, value, iter_idx)
                    # TODO: use this one:
                    harn.log_value(tag + ' iter ' + key, value, iter_idx)

                prog.update(harn.intervals['display_' + tag])

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
            # output_shape = labels.shape  # TODO: make sure this works
            # TODO: make general
            # output_shape = (labels.shape[0],) + tuple(harn.single_output_shape)
            # output_shape = (labels.shape[0], 3, labels.shape[1], labels.shape[2])
            # outputs = Variable(torch.rand(*output_shape))
            outputs = None
            loss = Variable(torch.rand(1))
            return outputs, loss

        # Forward prop through the model
        outputs = harn.model(inputs)

        # Compute the loss
        loss = harn.compute_loss(harn, outputs, labels)
        # loss = harn.criterion(outputs, labels)

        # Backprop and learn
        if learn:
            harn.optimizer.zero_grad()
            loss.backward()
            harn.optimizer.step()

        return outputs, loss

    def _tensorboard_inputs(harn, inputs, iter_idx, tag):
        """
            >>> tensor = torch.rand(4, 3, 42, 420)
            >>> inputs = [tensor]
        """
        # print("LOG IMAGES")
        if False:
            # todo: add dataset / task hook for extracting images
            pass
        else:
            # default_convert = im_loaders.rgb_tensor_to_imgs
            def default_convert(tensor):
                if len(tensor.shape) == 4:
                    arr = tensor.data.cpu().numpy().transpose(0, 2, 3, 1)
                    aux = []
                    if arr.shape[3] == 1:
                        ims = arr[:, :, :, 0]
                    elif arr.shape[3] == 3:
                        ims = arr
                    else:
                        ims = arr[:, :, :, 0:3]
                        aux = [arr[:, :, :, c] for c in range(3, arr.shape[3])]
                    imaux = [ims] + aux
                    return imaux
                elif len(tensor.shape) == 3:
                    return [tensor.data.cpu().numpy()]
                else:
                    raise NotImplementedError(str(tensor.shape))

        # def single_image_render(im):
        #     import plottool as pt
        #     with pt.RenderingContext() as render:
        #         pt.figure(fnum=1, pnum=(1, 1, 1), doclf=True)
        #         if len(im.shape) == 3:
        #             im = im[:, :, ::-1]
        #         print(im.shape)
        #         print(im.dtype)
        #         pt.imshow(im, norm=True, cmap='viridis', data_colorbar=True,
        #                   ax=pt.gca())
        #     out_im = render.image[:, :, ::-1]
        #     return out_im

        for i, input_tensor in enumerate(inputs):
            # print('\n\n')
            if len(input_tensor.shape) == 4:
                # print('input_tensor.shape = {!r}'.format(input_tensor.shape))
                imaux = default_convert(input_tensor[0:2])
                for c, images in enumerate(imaux):
                    extent = (images.max() - images.min())
                    images = (images - images.min()) / max(extent, 1e-6)
                    # print('images.shape = {!r}'.format(images.shape))
                    # images = [single_image_render(im) for im in images]
                    # for im in images:
                    #     print('im.shape = {!r}'.format(im.shape))
                    #     print('im.dtype = {!r}'.format(im.dtype))
                    #     print(im.max())
                    #     print(im.min())
                    harn.log_images(tag + '-c{c}-in{i}-iter{x}'.format(i=i, c=c, x=iter_idx), images, iter_idx)
            else:
                print('\nSKIPPING INPUT VISUALIZE\n')

    def _tensorboard_extra(harn, inputs, outputs, labels, tag, iter_idx, loader):
        # https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/04-utils/tensorboard/main.py#L83-L105
        # print("\nTENSORBOARD EXTRAS\n")
        # loader.dataset
        # print('\n\ninputs.shape = {!r}'.format(inputs[0].shape))

        if iter_idx == 0:
            harn._tensorboard_inputs(inputs, iter_idx, tag)

        if iter_idx % 1000 == 0:
            true = labels.data.cpu().numpy().ravel()
            n_classes = harn.hyper.other['n_classes']
            counts = np.bincount(true, minlength=n_classes)
            bins = list(range(n_classes + 1))
            harn.log_histogram(tag + '-true-', (bins, counts), iter_idx)

        if iter_idx % 1000 == 0:
            if not harn.dry:
                n_classes = harn.hyper.other['n_classes']
                preds = outputs.max(dim=1)[1].data.cpu().numpy().ravel()
                counts = np.bincount(preds, minlength=n_classes)
                bins = list(range(n_classes + 1))
                harn.log_histogram(tag + '-pred-', (bins, counts), iter_idx)
                # import torch.nn.functional as F
                # probs = torch.exp(F.log_softmax(outputs, dim=1))

    def add_metric_hook(harn, hook):
        """
        Adds a hook that should take arguments
        (harn, outputs, labels) and return a dictionary of scalar metrics
        """
        harn._metric_hooks.append(hook)

    def _call_metric_hooks(harn, outputs, labels, loss):
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
                _custom_dict = hook(harn, outputs, labels)
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

    def prev_snapshots(harn):
        prev_states = sorted(glob.glob(join(harn.snapshot_dpath, '_epoch_*.pt')))
        return prev_states

    def load_snapshot(harn, load_path):
        harn.log('Loading previous state: {}'.format(load_path))
        # snapshot = torch.load(load_path, map_location=nnio.device_mapping(harn.xpu.num))
        snapshot = torch.load(load_path, map_location=nnio.device_mapping(None))
        harn.epoch = snapshot['epoch']
        harn.model.load_state_dict(snapshot['model_state_dict'])
        # if 'optimizer_state_dict' in snapshot:
        #     harn.optimizer.load_state_dict(snapshot['optimizer_state_dict'])
        if 'early_stop' in snapshot:
            harn.early_stop = snapshot['early_stop']
        harn.log('Resuming training from epoch={}...'.format(harn.epoch))

    def save_snapshot(harn):
        # save snapshot
        save_path = join(harn.snapshot_dpath, '_epoch_{:08d}.pt'.format(harn.epoch))
        if harn.dry:
            harn.log('Would save snapshot to {}'.format(save_path))
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
                'early_stop': harn.early_stop,
            }
            torch.save(snapshot, save_path)
            # harn.log('Snapshot saved to {}'.format(save_path))


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
        python -m clab.torch.fit_harn2
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
