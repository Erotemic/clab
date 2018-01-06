# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import torch
import numpy as np
import ubelt as ub
import os  # NOQA
# from torch.autograd import Variable
# from collections import defaultdict
from os.path import join

import torchvision  # NOQA
from clab.torch import xpu_device
from clab.torch import models
from clab.torch import metrics
from clab.torch import hyperparams
from clab.torch import fit_harness
from clab.torch import im_loaders
from clab.torch import criterions
from clab import util  # NOQA
from clab.torch.transforms import (RandomWarpAffine, RandomGamma, RandomBlur,)
from clab.torch.transforms import (ImageCenterScale, DTMCenterScale, ZipTransforms)
from clab.util import imutil


class SSegInputsWrapper(torch.utils.data.Dataset):
    """
    Ignore:
        >>> from clab.live.urban_train import *
        >>> task = get_task('urban_mapper_3d')
        >>> learn, test = next(task.xval_splits())
        >>> inputs = learn
        >>> self = SSegInputsWrapper(inputs, task)
        >>> im, gt = self[0]

        for i in range(len(self)):
            im = self[i]
            dtm = im[:, 3, :, :]
            dsm = im[:, 4, :, :]

        np.bincount(gt.cpu().numpy().ravel())

    """
    def __init__(self, inputs, task, colorspace='RGB'):

        self.inputs = inputs
        self.task = task

        self.colorspace = colorspace

        self.loader = im_loaders.np_loader
        self.rng = np.random.RandomState(432432)

        inputs_base = ub.ensuredir((task.workdir, 'inputs'))
        inputs.base_dpath = inputs_base
        if len(inputs):
            inputs.prepare_images(force=True)
            inputs.prepare_input()
            self.input_id = inputs.input_id
            self.with_gt = self.inputs.gt_paths
        else:
            self.input_id = ''

        self.augment = None
        self.im_augment = torchvision.transforms.Compose([
            RandomGamma(rng=self.rng),
            RandomBlur(rng=self.rng),
        ])
        self.rand_aff = RandomWarpAffine(self.rng)

        if self.inputs.aux_paths:
            self.aux_keys = sorted(self.inputs.aux_paths.keys())
        else:
            self.aux_keys = []

        self.center_inputs = None
        self.use_aux_diff = ub.argflag('--use_aux_diff')
        self.use_dual_gt = ub.argval('--arch', default='unet')

    # def _make_normalizer(self, mode=2):
    def _make_normalizer(self, mode=3):
        transforms = []
        nan_value = -32767.0  # hack: specific number for DTM
        if len(self.inputs):

            if mode != 3:
                self.center_stats = self.inputs.prepare_center_stats(
                    self.task, nan_value=nan_value, colorspace=self.colorspace,
                    with_im=(mode == 3), stride=100,
                )
                # self.center_stats['image'].pop('detail')
                # if self.aux_keys:
                #     self.center_stats['aux'].pop('detail')

            if self.colorspace == 'LAB':
                # Do per-channel mean / std centering independently for LAB images
                channel_stats = self.center_stats['image']['simple']['channel']
                im_mean = channel_stats['mean']
                im_scale = channel_stats['std']
            elif self.colorspace == 'RGB':
                # Normalize across channels for RGB
                # scalar_stats = self.center_stats['image']['simple']['image']
                if mode == 1:
                    im_mean = 0
                    im_scale = 1
                elif mode == 2 or mode == 3:
                    im_mean = .5
                    im_scale = .75
                else:
                    raise KeyError(mode)
                # self.im_center = ub.identity
                print('im_mean = {!r}'.format(im_mean))
                print('im_scale = {!r}'.format(im_scale))
            else:
                raise Exception()

            im_center = ImageCenterScale(im_mean, im_scale)
            transforms.append(im_center)

            # im_scale = np.ceil(channel_stats['max']) - np.floor(channel_stats['min'])

            if self.aux_keys:
                # Note: Internal stats are not gaurenteed to make sense outside
                # the DTM domain.
                if mode == 1:
                    internal_aux_stats = self.center_stats['aux']['internal']
                    scale = internal_aux_stats['image']['mean_absdev_from_median']['mean']
                elif mode == 2:
                    scale = self.center_stats['aux']['internal']['image']['std']['mean']
                elif mode == 3:
                    scale = 4.7431301577290377
                else:
                    raise KeyError(mode)
                # zero the median on a per-chip basis, but use
                # the global internal_std to normalize extent
                # aux_std =
                print('aux scale = {!r}'.format(scale))
                aux_center = DTMCenterScale(scale,  # 2.8718751612937639
                                            nan_value=nan_value, fill='median')
                transforms.append(aux_center)

        center_inputs = ZipTransforms(transforms)
        self.center_inputs = center_inputs
        return center_inputs

    def dataset_metadata(self):
        """ metadata that should be saved along with each model snapshot """
        meta = {
            'colorspace': 'RGB',
            'center': [(x.__class__.__name__, x.__getstate__())
                       for x in self.center_inputs.transforms],
            'n_classes': self.n_classes,
            'n_channels': self.n_channels,
            'classnames': self.task.classnames,
            'ignore_label': self.ignore_label,
            # 'class_weights': self.class_weights(),
        }
        return  meta

    def _original_urban_mapper_normalizer(self, imcenter=.5, imscale=1.0):
        return self._custom_urban_mapper_normalizer(imcenter, imscale, 5.3757350869126723)

    def _custom_urban_mapper_normalizer(self, imcenter, imscale, aux_scale):
        nan_value = -32767.0  # hack: specific number for DTM
        assert self.colorspace == 'RGB'
        im_center = ImageCenterScale(imcenter, imscale)
        aux_center = DTMCenterScale(aux_scale, nan_value=nan_value,
                                    fill='median')
        transforms = [im_center, aux_center]
        return ZipTransforms(transforms)

    def __len__(self):
        return len(self.inputs)

    def from_tensor(self, im, gt=None):
        if len(im.shape) == 3:
            im = im.cpu().numpy().transpose(2, 0, 1)
        else:
            im = im.cpu().numpy().transpose(0, 2, 3, 1)
        if gt is not None:
            gt = gt.cpu().numpy()
        return im, gt

    def to_tensor(self, input_tuple, gt):
        # NHWC -> NCHW
        input_tuple = [im_loaders.image_to_float_tensor(data)
                       for data in input_tuple]
        if gt is None:
            gt_tensor = None
        else:
            gt_tensor = im_loaders.label_to_long_tensor(gt)
        return input_tuple, gt_tensor

    def load_inputs(self, index):
        im_fpath = self.inputs.im_paths[index]

        if self.inputs.gt_paths:
            gt_fpath = self.inputs.gt_paths[index]
            gt_hwc = self.loader(gt_fpath, colorspace=None)
        else:
            gt_hwc = None

        # Load in RGB for now, we will convert right before we center the data
        im = self.loader(im_fpath, colorspace='RGB')

        aux_channels = []
        if self.aux_keys:
            aux_paths = [self.inputs.aux_paths[k][index]
                         for k in self.aux_keys]
            aux_channel = np.dstack([
                self.loader(p, colorspace=None)
                for p in aux_paths
            ])
            aux_channels = [aux_channel]

        if self.augment:
            # Image augmentation must be done in RGB
            # Augment intensity independently
            im = self.im_augment(im)
            # Augment geometry consistently
            im, aux_channels, gt_hwc = self.rand_aff.sseg_warp(
                im, aux_channels, gt_hwc)

        im = imutil.convert_colorspace(im, src_space='RGB',
                                       dst_space=self.colorspace)

        # Do centering of inputs
        input_tuple_hwc = [im] + aux_channels
        input_tuple_hwc = self.center_inputs(input_tuple_hwc)

        if self.use_aux_diff:
            # add residual between dtm and dsm
            dtm_dsm = input_tuple_hwc[-1]
            residual = dtm_dsm[:, :, 0:1] - dtm_dsm[:, :, 1:2]
            input_tuple_hwc += [residual]

        # gt_tuple_hwc = [], if gt_hwc is None else [gt_hwc]

        return input_tuple_hwc, gt_hwc

    def __getitem__(self, index):
        """

        Ignore:
            >>> from clab.live.urban_train import *
            >>> from clab.tasks.urban_mapper_3d import UrbanMapper3D
            >>> self = load_task_dataset('urban_mapper_3d')['train']
            >>> self.augment = True
            >>> index = 0
            >>> self.center_inputs = self._make_normalizer()
            >>> self.use_aux_diff = True
            >>> im, gt = self[0]
        """
        input_tuple_hwc, gt_hwc = self.load_inputs(index)
        input_tuple, gt_tensor = self.to_tensor(input_tuple_hwc, gt_hwc)

        data_tensor = torch.cat(input_tuple, dim=0)

        if self.with_gt:
            # print('gotitem: ' + str(data_tensor.shape))
            # print('gt_tensor: ' + str(gt_tensor.shape))
            if self.use_dual_gt:
                mask = gt_tensor >= 2
                gt_tensor_alt = gt_tensor.clone()
                gt_tensor_alt[mask] = gt_tensor_alt[mask] - 1
                labels = [gt_tensor, gt_tensor_alt]
                inputs = [data_tensor]
                return inputs, labels

            return data_tensor, gt_tensor
        else:
            return data_tensor

    def show(self):
        # self.augment = False
        # self.augment = True
        loader = torch.utils.data.DataLoader(self, batch_size=6)
        iter_ = iter(loader)
        im_tensor, gt_tensor = next(iter_)
        # im_tensor = next(iter_)

        im_list, gt_list = self.from_tensor(im_tensor, gt_tensor)

        stacked_img = np.hstack([im[:, :, 0:3] for im in im_list])
        stacked_gt = np.hstack(gt_list)

        # stacked_gtblend = self.task.colorize(stacked_gt, stacked_img)

        import plottool as pt
        n_rows = 2
        if self.aux_keys:
            aux_imgs = [im[:, :, 3] for im in im_list]
            stacked_aux =  np.hstack(aux_imgs)
            aux_imgs2 = [im[:, :, 4] for im in im_list]
            stacked_aux2 =  np.hstack(aux_imgs2)
            n_rows += 2

        n_rows = 6
        pt.imshow(stacked_img[:, :, 0], pnum=(n_rows, 1, 1), cmap='viridis',
                  norm=True)
        pt.imshow(stacked_img[:, :, 1], pnum=(n_rows, 1, 2), cmap='viridis',
                  norm=True)
        pt.imshow(stacked_img[:, :, 2], pnum=(n_rows, 1, 3), cmap='viridis',
                  norm=True)
        pt.imshow(stacked_gt[:, :], pnum=(n_rows, 1, 4), cmap='viridis',
                  norm=True)
        # pt.imshow(stacked_img, pnum=(n_rows, 1, 1))
        # pt.imshow(stacked_gtblend, pnum=(n_rows, 1, 2))
        if self.aux_keys:
            pt.imshow(stacked_aux, pnum=(n_rows, 1, 5), cmap='viridis',
                      norm=True)
            pt.imshow(stacked_aux2, pnum=(n_rows, 1, 6), cmap='viridis',
                      norm=True)

    @property
    def n_channels(self):
        if self.aux_keys:
            c = 3 + len(self.aux_keys)
        else:
            c = 3
        return c + int(self.use_aux_diff)
        # return c + 1

    @property
    def n_classes(self):
        return int(self.task.labels.max() + 1)

    @property
    def ignore_label(self):
        return self.task.ignore_label

    def class_weights(self):
        """
            >>> from clab.live.urban_train import *
            >>> self = load_task_dataset('urban_mapper_3d')['train']
            >>> self.class_weights()
        """
        # HACK
        class_weights = np.array([ 0.05496113,  0.67041818,  1.96697962,  0. ])
        print('class_weights = {!r}'.format(class_weights))
        print('class_names   = {!r}'.format(self.task.classnames))
        return class_weights

        # Handle class weights
        print('prep class weights')
        gtstats = self.inputs.prepare_gtstats(self.task)
        gtstats = self.inputs.gtstats
        # Take class weights (ensure they are in the same order as labels)
        mfweight_dict = gtstats['mf_weight'].to_dict()
        class_weights = np.array(list(ub.take(mfweight_dict, self.task.classnames)))
        class_weights[self.task.ignore_labels] = 0

        if 'inner-building' in self.task.classnames:
            # increase weight of inner building
            class_weights[1] *= 2

        # HACK
        # class_weights[0] = 1.0
        # class_weights[1] = 0.7
        print('class_weights = {!r}'.format(class_weights))
        print('class_names   = {!r}'.format(self.task.classnames))
        return class_weights


def get_task(taskname, boundary=True, arch=None):
    # the arch param is a hack
    if taskname == 'urban_mapper_3d':
        from clab.tasks.urban_mapper_3d import UrbanMapper3D
        if boundary:
            workdir = '~/data/work/urban_mapper2'
            if arch.startswith('dense_unet'):
                workdir = '~/data/work/urban_mapper4'
        else:
            workdir = '~/data/work/urban_mapper'

        task = UrbanMapper3D(root='~/remote/aretha/data/UrbanMapper3D',
                             workdir=workdir, boundary=boundary)
        print(task.classnames)
        task.prepare_fullres_inputs()
        task.preprocess()
    else:
        assert False
    return task


def load_task_dataset(taskname, vali_frac=0, colorspace='RGB', combine=None, boundary=True, arch=None, halfcombo=None):
    task = get_task(taskname, boundary=boundary, arch=arch)
    learn, test = next(task.xval_splits())
    learn.tag = 'learn'

    # Split everything in the learning set into training / validation
    n_learn = len(learn)
    n_vali = int(n_learn * vali_frac)
    train = learn[n_vali:]
    vali = learn[:n_vali]
    vali.tag = 'vali'
    train.tag = 'train'

    if combine is None:
        combine = ub.argflag('--combine')

    if halfcombo is None:
        halfcombo = ub.argflag('--halfcombo')

    if halfcombo:
        # decrease testing for training
        n = len(test) // 2
        new_test = test[n:]
        train = learn + test[:n]
        test = new_test
        train.tag = 'train_h'
        test.tag = 'test_h'

    if combine:
        # HACK EVERYTHING TOGETHER
        train = learn + test
        from clab import inputs
        vali = inputs.Inputs()
        test = inputs.Inputs()

    train_dataset = SSegInputsWrapper(train, task, colorspace=colorspace)
    vali_dataset = SSegInputsWrapper(vali, task, colorspace=colorspace)
    test_dataset = SSegInputsWrapper(test, task, colorspace=colorspace)
    # train_dataset.augment = True

    print('* len(train_dataset) = {}'.format(len(train_dataset)))
    print('* len(vali_dataset) = {}'.format(len(vali_dataset)))
    print('* len(test_dataset) = {}'.format(len(test_dataset)))
    datasets = {
        'train': train_dataset,
        'vali': vali_dataset,
        'test': test_dataset,
    }
    return datasets


def directory_structure(workdir, arch, datasets, pretrained=None,
                        train_hyper_id=None, suffix=''):
    """
    from .torch.urban_train import *
    datasets = load_task_dataset('urban_mapper_3d')
    datasets['train']._make_normalizer()
    arch = 'foobar'
    workdir = datasets['train'].task.workdir
    ut.exec_funckw(directory_structure, globals())
    """
    arch_dpath = ub.ensuredir((workdir, 'arch', arch))
    train_base = ub.ensuredir((arch_dpath, 'train'))

    if pretrained is None:
        train_init_id = 'None'
    elif  len(pretrained) < 8:
        train_init_id = pretrained
    else:
        train_init_id = util.hash_data(pretrained)[:8]

    train_hyper_hashid = util.hash_data(train_hyper_id)[:8]

    train_id = '{}_{}_{}_{}'.format(
        datasets['train'].input_id, arch, train_init_id, train_hyper_hashid) + suffix

    train_dpath = ub.ensuredir((
        train_base,
        'input_' + datasets['train'].input_id, 'solver_{}'.format(train_id)
    ))

    train_info =  {
        'arch': arch,
        'train_id': datasets['train'].input_id,
        'train_hyper_id': train_hyper_id,
        'train_hyper_hashid': train_hyper_hashid,
        'colorspace': datasets['train'].colorspace,
    }
    if hasattr(datasets['train'], 'center_inputs'):
        # Hack in centering information
        train_info['hack_centers'] = [
            (t.__class__.__name__, t.__getstate__())
            # ub.map_vals(str, t.__dict__)
            for t in datasets['train'].center_inputs.transforms
        ]
    util.write_json(join(train_dpath, 'train_info.json'), train_info)

    print('+=========')
    # print('hyper_strid = {!r}'.format(params.hyper_id()))
    print('train_init_id = {!r}'.format(train_init_id))
    print('arch = {!r}'.format(arch))
    print('train_hyper_hashid = {!r}'.format(train_hyper_hashid))
    print('train_hyper_id = {!r}'.format(train_hyper_id))
    print('train_id = {!r}'.format(train_id))
    print('+=========')

    return train_dpath


def urban_fit():
    """

    CommandLine:
        python -m clab.live.urban_train urban_fit --profile

        python -m clab.live.urban_train urban_fit --task=urban_mapper_3d --arch=segnet

        python -m clab.live.urban_train urban_fit --task=urban_mapper_3d --arch=unet --noaux
        python -m clab.live.urban_train urban_fit --task=urban_mapper_3d --arch=unet

        python -m clab.live.urban_train urban_fit --task=urban_mapper_3d --dry

        python -m clab.live.urban_train urban_fit --task=urban_mapper_3d --arch=unet --colorspace=RGB --combine


        python -m clab.live.urban_train urban_fit --task=urban_mapper_3d --arch=unet --dry

        python -m clab.live.urban_train urban_fit --task=urban_mapper_3d --arch=unet2 --colorspace=RGB --combine
        python -m clab.live.urban_train urban_fit --task=urban_mapper_3d --arch=unet2 --colorspace=RGB --use_aux_diff

        python -m clab.live.urban_train urban_fit --task=urban_mapper_3d --arch=dense_unet --colorspace=RGB --use_aux_diff


        # Train a variant of the dense net with more parameters
        python -m clab.live.urban_train urban_fit --task=urban_mapper_3d --arch=dense_unet --colorspace=RGB --use_aux_diff --combine \
                --pretrained '/home/local/KHQ/jon.crall/data/work/urban_mapper4/arch/dense_unet/train/input_25800-phpjjsqu/solver_25800-phpjjsqu_dense_unet_mmavmuou_zeosddyf_a=1,c=RGB,n_ch=6,n_cl=4/torch_snapshots/_epoch_00000030.pt' --gpu=1

        # Fine tune the model using all the available data
        python -m clab.live.urban_train urban_fit --task=urban_mapper_3d --arch=unet2 --colorspace=RGB --use_aux_diff --combine \
                --pretrained '/home/local/KHQ/jon.crall/data/work/urban_mapper2/arch/unet2/train/input_25800-hemanvft/solver_25800-hemanvft_unet2_mmavmuou_stuyuerd_a=1,c=RGB,n_ch=6,n_cl=4/torch_snapshots/_epoch_00000041.pt' --gpu=3 --finetune


        # Keep a bit of the data for validation but use more
        python -m clab.live.urban_train urban_fit --task=urban_mapper_3d --arch=unet2 --colorspace=RGB --use_aux_diff --halfcombo \
                --pretrained '/home/local/KHQ/jon.crall/data/work/urban_mapper2/arch/unet2/train/input_25800-hemanvft/solver_25800-hemanvft_unet2_mmavmuou_stuyuerd_a=1,c=RGB,n_ch=6,n_cl=4/torch_snapshots/_epoch_00000041.pt' --gpu=3

    Example:
        >>> from clab.torch.fit_harness import *
        >>> harn = urban_fit()
    """
    arch = ub.argval('--arch', default='unet')
    colorspace = ub.argval('--colorspace', default='RGB').upper()

    datasets = load_task_dataset('urban_mapper_3d', colorspace=colorspace, arch=arch)
    datasets['train'].augment = True

    # Make sure we use consistent normalization
    # TODO: give normalization a part of the hashid
    # TODO: save normalization type with the model
    # datasets['train'].center_inputs = datasets['train']._make_normalizer()

    # if ub.argflag('--combine'):
    #     # custom centering from the initialization point I'm going to use
    #     datasets['train'].center_inputs = datasets['train']._custom_urban_mapper_normalizer(
    #         0.3750553785198646, 1.026544662398811, 2.5136079110849674)
    # else:
    # datasets['train'].center_inputs = datasets['train']._make_normalizer(mode=2)
    datasets['train'].center_inputs = datasets['train']._make_normalizer(mode=3)
    # datasets['train'].center_inputs = _custom_urban_mapper_normalizer(0, 1, 2.5)

    datasets['test'].center_inputs = datasets['train'].center_inputs
    datasets['vali'].center_inputs = datasets['train'].center_inputs

    # Ensure normalization is the same for each dataset
    datasets['train'].augment = True

    # turn off aux layers
    if ub.argflag('--noaux'):
        for v in datasets.values():
            v.aux_keys = []

    batch_size = 14
    if arch == 'segnet':
        batch_size = 6
    elif arch == 'dense_unet':
        batch_size = 6
        # dense_unet batch memsizes
        # idle =   11 MiB
        # 0    =  438 MiB
        # 3   ~= 5000 MiB
        # 5    = 8280 MiB
        # 6    = 9758 MiB
        # each image adds (1478 - 1568.4) MiB

    n_classes = datasets['train'].n_classes
    n_channels = datasets['train'].n_channels
    class_weights = datasets['train'].class_weights()
    ignore_label = datasets['train'].ignore_label

    print('n_classes = {!r}'.format(n_classes))
    print('n_channels = {!r}'.format(n_channels))
    print('batch_size = {!r}'.format(batch_size))

    hyper = hyperparams.HyperParams(
        criterion=(criterions.CrossEntropyLoss2D, {
            'ignore_label': ignore_label,
            # TODO: weight should be a FloatTensor
            'weight': class_weights,
        }),
        optimizer=(torch.optim.SGD, {
            # 'weight_decay': .0006,
            'weight_decay': .0005,
            'momentum': 0.99 if arch == 'dense_unet' else .9,
            'nesterov': True,
        }),
        scheduler=('Exponential', {
            'gamma': 0.99,
            # 'base_lr': 0.0015,
            'base_lr': 0.001 if not ub.argflag('--halfcombo') else 0.0005,
            'stepsize': 2,
        }),
        other={
            'n_classes': n_classes,
            'n_channels': n_channels,
            'augment': datasets['train'].augment,
            'colorspace': datasets['train'].colorspace,
        }
    )

    starting_points = {

        'unet_rgb_4k': ub.truepath('~/remote/aretha/data/work/urban_mapper/arch/unet/train/input_4214-yxalqwdk/solver_4214-yxalqwdk_unet_vgg_nttxoagf_a=1,n_ch=5,n_cl=3/torch_snapshots/_epoch_00000236.pt'),

        # 'unet_rgb_8k': ub.truepath('~/remote/aretha/data/work/urban_mapper/arch/unet/train/input_8438-haplmmpq/solver_8438-haplmmpq_unet_None_kvterjeu_a=1,c=RGB,n_ch=5,n_cl=3/torch_snapshots/_epoch_00000402.pt'),
        # "ImageCenterScale", {"im_mean": [[[0.3750553785198646]]], "im_scale": [[[1.026544662398811]]]}
        # "DTMCenterScale", "std": 2.5136079110849674, "nan_value": -32767.0 }

        # 'unet_rgb_8k': ub.truepath(
        #     '~/data/work/urban_mapper2/arch/unet/train/input_4214-guwsobde/'
        #     'solver_4214-guwsobde_unet_mmavmuou_eqnoygqy_a=1,c=RGB,n_ch=5,n_cl=4/torch_snapshots/_epoch_00000189.pt'
        # )

        'unet_rgb_8k': ub.truepath(
            '~/remote/aretha/data/work/urban_mapper2/arch/unet2/train/input_4214-guwsobde/'
            'solver_4214-guwsobde_unet2_mmavmuou_tqynysqo_a=1,c=RGB,n_ch=5,n_cl=4/torch_snapshots/_epoch_00000100.pt'
        )
    }

    pretrained = ub.argval('--pretrained', default=None)
    if pretrained is None:
        if arch == 'segnet':
            pretrained = 'vgg'
        else:
            pretrained = None
            if ub.argflag('--combine'):
                pretrained = starting_points['unet_rgb_8k']

                if arch == 'unet2':
                    pretrained = '/home/local/KHQ/jon.crall/data/work/urban_mapper2/arch/unet2/train/input_25800-hemanvft/solver_25800-hemanvft_unet2_mmavmuou_stuyuerd_a=1,c=RGB,n_ch=6,n_cl=4/torch_snapshots/_epoch_00000042.pt'
                elif arch == 'dense_unet2':
                    pretrained = '/home/local/KHQ/jon.crall/data/work/urban_mapper2/arch/unet2/train/input_25800-hemanvft/solver_25800-hemanvft_unet2_mmavmuou_stuyuerd_a=1,c=RGB,n_ch=6,n_cl=4/torch_snapshots/_epoch_00000042.pt'
            else:
                pretrained = starting_points['unet_rgb_4k']

    train_dpath = directory_structure(
        datasets['train'].task.workdir, arch, datasets,
        pretrained=pretrained,
        train_hyper_id=hyper.hyper_id(),
        suffix='_' + hyper.other_id())

    print('arch = {!r}'.format(arch))
    dry = ub.argflag('--dry')
    if dry:
        model = None
    elif arch == 'segnet':
        model = models.SegNet(in_channels=n_channels, n_classes=n_classes)
        model.init_he_normal()
        if pretrained == 'vgg':
            model.init_vgg16_params()
    elif arch == 'linknet':
        model = models.LinkNet(in_channels=n_channels, n_classes=n_classes)
    elif arch == 'unet':
        model = models.UNet(in_channels=n_channels, n_classes=n_classes,
                            nonlinearity='leaky_relu')
        snapshot = xpu_device.XPU(None).load(pretrained)
        model_state_dict = snapshot['model_state_dict']
        model.load_partial_state(model_state_dict)
    elif arch == 'unet2':
        from clab.live import unet2
        model = unet2.UNet2(n_alt_classes=3, in_channels=n_channels,
                            n_classes=n_classes, nonlinearity='leaky_relu')
        snapshot = xpu_device.XPU(None).load(pretrained)
        model_state_dict = snapshot['model_state_dict']
        model.load_partial_state(model_state_dict)

    elif arch == 'dense_unet':
        from clab.live import unet3
        model = unet3.DenseUNet(n_alt_classes=3, in_channels=n_channels,
                                n_classes=n_classes)
        model.init_he_normal()
        snapshot = xpu_device.XPU(None).load(pretrained)
        model_state_dict = snapshot['model_state_dict']
        model.load_partial_state(model_state_dict)
    elif arch == 'dense_unet2':
        from clab.live import unet3
        model = unet3.DenseUNet2(n_alt_classes=3, in_channels=n_channels,
                                 n_classes=n_classes)
        # model.init_he_normal()
        snapshot = xpu_device.XPU(None).load(pretrained)
        model_state_dict = snapshot['model_state_dict']
        model.load_partial_state(model_state_dict, shock_partial=False)
    elif arch == 'dummy':
        model = models.SSegDummy(in_channels=n_channels, n_classes=n_classes)
    else:
        raise ValueError('unknown arch')

    if ub.argflag('--finetune'):
        # Hack in a reduced learning rate
        hyper = hyperparams.HyperParams(
            criterion=(criterions.CrossEntropyLoss2D, {
                'ignore_label': ignore_label,
                # TODO: weight should be a FloatTensor
                'weight': class_weights,
            }),
            optimizer=(torch.optim.SGD, {
                # 'weight_decay': .0006,
                'weight_decay': .0005,
                'momentum': 0.99 if arch == 'dense_unet' else .9,
                'nesterov': True,
            }),
            scheduler=('Constant', {
                'base_lr': 0.0001,
            }),
            other={
                'n_classes': n_classes,
                'n_channels': n_channels,
                'augment': datasets['train'].augment,
                'colorspace': datasets['train'].colorspace,
            }
        )

    xpu = xpu_device.XPU.from_argv()

    if datasets['train'].use_aux_diff:
        # arch in ['unet2', 'dense_unet']:

        from clab.live import fit_harn2
        harn = fit_harn2.FitHarness(
            model=model, hyper=hyper, datasets=datasets, xpu=xpu,
            train_dpath=train_dpath, dry=dry,
            batch_size=batch_size,
        )
        harn.criterion2 = criterions.CrossEntropyLoss2D(
            weight=torch.FloatTensor([.1, 1, 0]),
            ignore_label=2
        )

        def compute_loss(harn, outputs, labels):

            output1, output2 = outputs
            label1, label2 = labels

            # Compute the loss
            loss1 = harn.criterion(output1, label1)
            loss2 = harn.criterion2(output2, label2)
            loss = (.45 * loss1 + .55 * loss2)
            return loss

        harn.compute_loss = compute_loss

        # z = harn.loaders['train']
        # b = next(iter(z))
        # print('b = {!r}'.format(b))
        # import sys
        # sys.exit(0)

        def custom_metrics(harn, output, label):
            ignore_label = datasets['train'].ignore_label
            labels = datasets['train'].task.labels

            metrics_dict = metrics._sseg_metrics(output[1], label[1],
                                                 labels=labels,
                                                 ignore_label=ignore_label)
            return metrics_dict
    else:
        harn = fit_harness.FitHarness(
            model=model, hyper=hyper, datasets=datasets, xpu=xpu,
            train_dpath=train_dpath, dry=dry,
            batch_size=batch_size,
        )

        def custom_metrics(harn, output, label):
            ignore_label = datasets['train'].ignore_label
            labels = datasets['train'].task.labels

            metrics_dict = metrics._sseg_metrics(output, label, labels=labels,
                                                 ignore_label=ignore_label)
            return metrics_dict

    harn.add_metric_hook(custom_metrics)

    # HACK
    # im = datasets['train'][0][0]
    # w, h = im.shape[-2:]
    # single_output_shape = (n_classes, w, h)
    # harn.single_output_shape = single_output_shape
    # print('harn.single_output_shape = {!r}'.format(harn.single_output_shape))

    harn.run()
    return harn


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m clab.live.urban_train
        python -m clab.live.urban_train urban_fit
        python -m clab.live.urban_train urban_fit --dry
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
