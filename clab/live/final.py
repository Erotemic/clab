import torch
import numpy as np
import ubelt as ub
import torch
import torchvision  # NOQA
from clab.torch.transforms import (RandomWarpAffine, RandomGamma, RandomBlur,)
from clab.torch.transforms import (ImageCenterScale, DTMCenterScale, ZipTransforms)
from clab.torch import xpu_device
from clab.torch import models
from clab.torch import metrics
from clab.torch import hyperparams
from clab.torch import fit_harness
from clab.torch import im_loaders
from clab.torch import criterions


class UrbanDataset(torch.utils.data.Dataset):
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
    def __init__(self, inputs, task):

        self.inputs = inputs
        self.task = task
        self.colorspace = 'RGB'

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
        self.use_aux_diff = True
        self.use_dual_gt = True

    # def _make_normalizer(self, mode=2):
    def _make_normalizer(self):
        transforms = []
        nan_value = -32767.0  # hack: specific number for DTM
        if len(self.inputs):
            # if mode != 3:
            #     self.center_stats = self.inputs.prepare_center_stats(
            #         self.task, nan_value=nan_value, colorspace=self.colorspace,
            #         with_im=(mode == 3), stride=100,
            #     )
            #     # self.center_stats['image'].pop('detail')
            #     # if self.aux_keys:
            #     #     self.center_stats['aux'].pop('detail')

            # Normalize across channels for RGB
            # scalar_stats = self.center_stats['image']['simple']['image']
            im_mean = .5
            im_scale = .75
            # self.im_center = ub.identity
            print('im_mean = {!r}'.format(im_mean))
            print('im_scale = {!r}'.format(im_scale))

            im_center = ImageCenterScale(im_mean, im_scale)
            transforms.append(im_center)

            # im_scale = np.ceil(channel_stats['max']) - np.floor(channel_stats['min'])

            if self.aux_keys:
                scale = 4.7431301577290377
                # zero the median on a per-chip basis, but use
                # the global internal_std to normalize extent
                # aux_std =
                print('aux scale = {!r}'.format(scale))
                aux_center = DTMCenterScale(scale, nan_value=nan_value,
                                            fill='median')
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

        # # Handle class weights
        # print('prep class weights')
        # gtstats = self.inputs.prepare_gtstats(self.task)
        # gtstats = self.inputs.gtstats
        # # Take class weights (ensure they are in the same order as labels)
        # mfweight_dict = gtstats['mf_weight'].to_dict()
        # class_weights = np.array(list(ub.take(mfweight_dict, self.task.classnames)))
        # class_weights[self.task.ignore_labels] = 0

        # if 'inner-building' in self.task.classnames:
        #     # increase weight of inner building
        #     class_weights[1] *= 2

        # # HACK
        # # class_weights[0] = 1.0
        # # class_weights[1] = 0.7
        # print('class_weights = {!r}'.format(class_weights))
        # print('class_names   = {!r}'.format(self.task.classnames))
        # return class_weights


def train(train_data_path):
    """
    train_data_path = ub.truepath('~/remote/aretha/data/UrbanMapper3D/training')
    """
    workdir = ub.truepath('~/work')

    from clab.tasks.urban_mapper_3d import UrbanMapper3D
    from clab import preprocess
    task = UrbanMapper3D(root=train_data_path, workdir=workdir, boundary=True)

    fullres = task.load_fullres_inputs('.')
    fullres = task.create_boundary_groundtruth(fullres)
    del fullres.paths['gti']

    rng = np.random.RandomState(0)
    idxs = np.arange(len(fullres))
    rng.shuffle(idxs)

    vali_frac = .15
    n_vali = int(len(idxs) * vali_frac)

    train_idx = idxs[0:-n_vali][0:1]
    vali_idx = idxs[-n_vali:][0:1]

    train_fullres_inputs = fullres.take(train_idx)
    vali_fullres_inputs = fullres.take(vali_idx)
    # take doesnt take the dump_im_names
    train_fullres_inputs.dump_im_names = list(ub.take(fullres.dump_im_names, train_idx))
    vali_fullres_inputs.dump_im_names = list(ub.take(fullres.dump_im_names, vali_idx))

    prep = preprocess.Preprocessor(ub.ensuredir((task.workdir, 'data_train1')))
    # prep.part_config['overlap'] = .75
    prep.part_config['overlap'] = 0
    prep.ignore_label = task.ignore_label
    train_part_inputs = prep.make_parts(train_fullres_inputs, scale=1, clear=0)

    prep = preprocess.Preprocessor(ub.ensuredir((task.workdir, 'data_vali1')))
    # prep.part_config['overlap'] = .75
    prep.part_config['overlap'] = 0
    prep.ignore_label = task.ignore_label
    vali_part_inputs = prep.make_parts(vali_fullres_inputs, scale=1, clear=0)

    train_dataset = UrbanDataset(train_part_inputs, task)
    vali_dataset = UrbanDataset(vali_part_inputs, task)

    print('* len(train_dataset) = {}'.format(len(train_dataset)))
    print('* len(vali_dataset) = {}'.format(len(vali_dataset)))
    datasets = {
        'train': train_dataset,
        'vali': vali_dataset,
    }

    datasets['train'].center_inputs = datasets['train']._make_normalizer()
    datasets['vali'].center_inputs = datasets['train'].center_inputs

    datasets['train'].augment = True

    batch_size = 14

    n_classes = datasets['train'].n_classes
    n_channels = datasets['train'].n_channels
    class_weights = datasets['train'].class_weights()
    ignore_label = datasets['train'].ignore_label

    print('n_classes = {!r}'.format(n_classes))
    print('n_channels = {!r}'.format(n_channels))
    print('batch_size = {!r}'.format(batch_size))

    arches = [
        'dense_unet',
        'unet2',
    ]

    xpu = xpu_device.XPU.from_argv()

    arch_to_train_dpath = {}
    arch_to_best_epochs = {}

    for arch in arches:

        hyper = hyperparams.HyperParams(
            criterion=(criterions.CrossEntropyLoss2D, {
                'ignore_label': ignore_label,
                # TODO: weight should be a FloatTensor
                'weight': class_weights,
            }),
            optimizer=(torch.optim.SGD, {
                # 'weight_decay': .0006,
                'weight_decay': .0005,
                'momentum': .9,
                'nesterov': True,
            }),
            scheduler=('Exponential', {
                'gamma': 0.99,
                'base_lr': 0.001,
                'stepsize': 2,
            }),
            other={
                'n_classes': n_classes,
                'n_channels': n_channels,
                'augment': datasets['train'].augment,
                'colorspace': datasets['train'].colorspace,
            }
        )

        from clab.live.urban_train import directory_structure
        train_dpath = directory_structure(
            datasets['train'].task.workdir, arch, datasets,
            pretrained=None,
            train_hyper_id=hyper.hyper_id(),
            suffix='_' + hyper.other_id())

        arch_to_train_dpath[arch] = train_dpath

        if arch == 'unet2':
            from clab.live import unet2
            model = unet2.UNet2(n_alt_classes=3, in_channels=n_channels,
                                n_classes=n_classes, nonlinearity='leaky_relu')
        elif arch == 'dense_unet':
            from clab.live import unet3
            model = unet3.DenseUNet(n_alt_classes=3, in_channels=n_channels,
                                    n_classes=n_classes)

        dry = 0

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
        harn.config['max_iter'] = 3

        def compute_loss(harn, outputs, labels):

            output1, output2 = outputs
            label1, label2 = labels

            # Compute the loss
            loss1 = harn.criterion(output1, label1)
            loss2 = harn.criterion2(output2, label2)
            loss = (.45 * loss1 + .55 * loss2)
            return loss

        harn.compute_loss = compute_loss

        def custom_metrics(harn, output, label):
            ignore_label = datasets['train'].ignore_label
            labels = datasets['train'].task.labels

            metrics_dict = metrics._sseg_metrics(output[1], label[1],
                                                 labels=labels,
                                                 ignore_label=ignore_label)
            return metrics_dict

        harn.add_metric_hook(custom_metrics)

        harn.run()
        arch_to_best_epochs[arch] = harn.early_stop.best_epochs()
        fit_harn2.get_snapshot(train_dpath)

        # free up memory for the next model
        del harn
        del hyper
        del model


def test(train_data_path, test_data_path, output_file):
    """
    train_data_path
    test_data_path
    output_file
    """
    pass


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m clab.live.final train
    """
    import sys
    if sys.argv[1] == 'train':
        train_data_path = ub.truepath('~/remote/aretha/data/UrbanMapper3D/training')
        train(train_data_path)
