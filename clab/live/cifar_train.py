import numpy as np
import ubelt as ub
import torch
import torchvision
import pandas as pd
from torchvision.datasets import cifar
from clab.torch import xpu_device
from clab.torch import nninit
from clab.torch import hyperparams
from clab.torch import fit_harness
from clab.torch.transforms import (ImageCenterScale,)
from clab.torch.transforms import (RandomWarpAffine, RandomGamma, RandomBlur,)
from clab import util


class Task(object):
    def __init__(task, labelnames=None, ignore_labelnames=[], alias={}):
        if labelnames is not None:
            task.set_labelnames(labelnames, ignore_labelnames, alias)

    def set_labelnames(task, labelnames, ignore_labelnames=[], alias={}):
        task.labelnames = list(labelnames)
        task.labelname_alias = alias
        task.ignore_labelnames = ignore_labelnames

        # Remove aliased classes
        for k in alias.keys():
            if k in task.labelnames:
                task.labelnames.remove(k)

        # Assign an integer label to each labelname
        task.labelname_to_id = ub.invert_dict(dict(enumerate(task.labelnames)))

        # Map aliased classes to a different label
        for k, v in alias.items():
            task.labelname_to_id[k] = task.labelname_to_id[v]

        task.ignore_labelnames = ignore_labelnames
        task.ignore_labels = np.array(list(ub.take(task.labelname_to_id, task.ignore_labelnames)))

        task.labels = np.arange(len(task.labelnames))
        task.relevant_labels = np.setdiff1d(task.labels, task.ignore_labels)


class CIFAR100_Task(Task):
    """
    task = CIFAR100_Task()
    task._initialize()
    ignore_labelnames = []
    alias = {}
    """
    def __init__(task, root=None):
        if root is None:
            root = ub.ensure_app_cache_dir('clab')
        task.root = root
        task._initialize()

    def _initialize(task):
        from os.path import join
        import pickle
        train_dset = cifar.CIFAR100(root=task.root, download=False, train=True)

        fpath = join(train_dset.root, cifar.CIFAR100.base_folder, 'meta')
        with open(fpath, 'rb') as fo:
            entry = pickle.load(fo, encoding='latin1')
            labelnames = entry['fine_label_names']
        task.set_labelnames(labelnames)


def mutex_clf_gt_info(gt_labels, task):
    """
    gt_labels = train_dset.train_labels
    """
    index = pd.Index(task.labels, name='label')
    gtstats = pd.DataFrame(0, index=index, columns=['freq'], dtype=np.int)

    label_freq = pd.value_counts(gt_labels)
    gtstats.freq = pd.to_numeric(label_freq)

    gtstats['classname'] = list(ub.take(task.labelnames, gtstats.index))
    gtstats['mf_weight'] = gtstats.freq.median() / gtstats.freq
    gtstats.loc[~np.isfinite(gtstats.mf_weight), 'mf_weight'] = 1

    # Clip weights, so nothing gets crazy high weights, low weights are ok
    gtstats = gtstats.sort_index()
    gtstats.index.name = 'label'
    gtstats = gtstats.reset_index().set_index('classname', drop=False)
    return gtstats


class InMemoryInputs(ub.NiceRepr):
    """
    Change inputs.Inputs to OnDiskInputs
    """
    def __init__(inputs, tag=''):
        inputs.tag = tag
        inputs.im = None
        inputs.gt = None
        inputs.colorspace = None
        inputs.input_id = None

    def __nice__(inputs):
        n = len(inputs)
        return '{} {}'.format(inputs.tag, n)

    def __len__(inputs):
        if inputs.im is not None:
            n = len(inputs.im)
        elif inputs.gt is not None:
            n = len(inputs.gt)
        else:
            n = 0
        return n

    @classmethod
    def from_bhwc_rgb(cls, bhwc, labels=None, **kw):
        # convert to bhwc
        inputs = cls(**kw)
        inputs.im = bhwc
        inputs.gt = labels
        inputs.colorspace = 'rgb'
        return inputs

    def convert_colorspace(inputs, colorspace):
        if colorspace.lower() == inputs.colorspace.lower():
            return
        im_out = np.empty_like(inputs.im)
        dst = np.ascontiguousarray(np.empty_like(inputs.im[0]))
        for ix, im in enumerate(inputs.im):
            util.convert_colorspace(im, src_space=inputs.colorspace,
                                    dst_space=colorspace, dst=dst)
            im_out[ix] = dst
        inputs.im = im_out
        inputs.colorspace = colorspace

    def take(inputs, idxs, **kw):
        new_inputs = inputs.__class__(**kw)
        new_inputs.im = inputs.im.take(idxs, axis=0)
        new_inputs.gt = inputs.gt.take(idxs, axis=0)
        new_inputs.colorspace = inputs.colorspace
        return new_inputs

    def prepare_id(self, force=False):
        if self.input_id is not None and not force:
            return

        depends = []
        depends.append(self.im)
        depends.append(self.gt)

    def _set_id_from_dependency(self, depends):
        """
        Allow for arbitrary representation of dependencies
        (user must ensure that it is consistent)
        """
        print('Preparing id for {} images'.format(self.tag))
        abbrev = 8
        hashid = util.hash_data(depends)[:abbrev]
        n_input = len(self)
        self.input_id = '{}-{}'.format(n_input, hashid)
        print(' * n_input = {}'.format(n_input))
        print(' * input_id = {}'.format(self.input_id))


class CIFAR_Wrapper(torch.utils.data.Dataset):  # cifar.CIFAR10):
    def __init__(self, inputs, task, workdir):
        self.inputs = inputs
        self.task = task

        self.colorspace = 'RGB'

        self.rng = np.random.RandomState(432432)

        inputs_base = ub.ensuredir((workdir, 'inputs'))
        inputs.base_dpath = inputs_base
        if len(inputs):
            inputs.prepare_id()
            self.input_id = inputs.input_id
            self.with_gt = self.inputs.gt is not None
        else:
            self.input_id = ''

        self.augment = None
        self.im_augment = torchvision.transforms.Compose([
            RandomGamma(rng=self.rng),
            RandomBlur(rng=self.rng),
        ])
        self.rand_aff = RandomWarpAffine(self.rng)
        self.center_inputs = None

    def _make_normalizer(self, mode='dependant'):
        if len(self.inputs):
            if mode == 'dependant':
                # dependent centering per channel (for RGB)
                im_mean = self.inputs.im.mean()
                im_scale = self.inputs.im.std()
            elif mode == 'independent':
                # Independent centering per channel (for LAB)
                im_mean = self.inputs.im.mean(axis=(0, 1, 2))
                im_scale = self.inputs.im.std(axis=(0, 1, 2))

            center_inputs = ImageCenterScale(im_mean, im_scale)

        self.center_inputs = center_inputs
        return center_inputs

    def __len__(self):
        return len(self.inputs)

    def load_inputs(self, index):
        im = self.inputs.im[index]

        if self.inputs.gt is not None:
            gt = self.inputs.gt[index]
        else:
            gt = None

        if self.augment:
            # Image augmentation must be done in RGB
            # Augment intensity independently
            im = self.im_augment(im)
            # Augment geometry consistently
            params = self.rand_aff.random_params()
            im = self.rand_aff.warp(im, params, interp='cubic')

        im = util.convert_colorspace(im, src_space=self.inputs.colorspace,
                                     dst_space=self.colorspace)

        # Do centering of inputs
        im = self.center_inputs(im)
        return im, gt

    def __getitem__(self, index):
        """

        Ignore:
            >>> from clab.live.sseg_train import *
            >>> self = load_task_dataset('urban_mapper_3d')['train']
            >>> self.augment = True
            >>> index = 0
            >>> self.center_inputs = self._make_normalizer()
            >>> im, gt = self[0]
        """
        from clab.torch import im_loaders
        im, gt = self.load_inputs(index)
        input_tensor = im_loaders.numpy_image_to_float_tensor(im)

        if self.with_gt:
            # print('gotitem: ' + str(data_tensor.shape))
            # print('gt_tensor: ' + str(gt_tensor.shape))
            return input_tensor, gt
        else:
            return input_tensor

    @property
    def n_channels(self):
        return 3

    @property
    def n_classes(self):
        return int(self.task.labels.max() + 1)

    @property
    def ignore_labels(self):
        return self.task.ignore_labels

    def class_weights(self):
        """
            >>> from clab.live.sseg_train import *
            >>> self = load_task_dataset('urban_mapper_3d')['train']
            >>> self.class_weights()
        """
        # # Handle class weights
        # print('prep class weights')
        # gtstats = self.inputs.prepare_gtstats(self.task)
        # gtstats = self.inputs.gtstats
        # # Take class weights (ensure they are in the same order as labels)
        # mfweight_dict = gtstats['mf_weight'].to_dict()
        # class_weights = np.array(list(ub.take(mfweight_dict, self.task.classnames)))
        # class_weights[self.task.ignore_labels] = 0
        # # HACK
        # # class_weights[0] = 1.0
        # # class_weights[1] = 0.7
        # print('class_weights = {!r}'.format(class_weights))
        # print('class_names   = {!r}'.format(self.task.classnames))
        class_weights = np.ones(self.n_classes)
        return class_weights


def cifar_training_datasets():
    """
    """
    root = ub.ensure_app_cache_dir('clab')
    train_dset = cifar.CIFAR100(root=root, download=True, train=True)
    bchw = (train_dset.train_data).astype(np.float32) / 255.0
    labels = np.array(train_dset.train_labels)
    inputs = InMemoryInputs.from_bhwc_rgb(bchw, labels=labels)
    # inputs.convert_colorspace('lab')

    vali_frac = .1
    n_vali = int(len(inputs) * vali_frac)
    input_idxs = util.random_indices(len(inputs), seed=0)
    train_idxs = sorted(input_idxs[:-n_vali])
    vali_idxs = sorted(input_idxs[-n_vali:])

    train_inputs = inputs.take(train_idxs, tag='train')
    vali_inputs = inputs.take(vali_idxs, tag='vali')
    # The dataset name and indices should fully specifiy dependencies
    train_inputs._set_id_from_dependency(['cifar100-train', train_idxs])
    vali_inputs._set_id_from_dependency(['cifar100-train', vali_idxs])

    workdir = ub.ensuredir(ub.truepath('~/data/work/cifar'))

    task = CIFAR100_Task()

    train_dset = CIFAR_Wrapper(train_inputs, task, workdir)
    vali_dset = CIFAR_Wrapper(vali_inputs, task, workdir)

    datasets = {
        'train': train_dset,
        'vali': vali_dset,
    }
    return datasets


def train():
    """
    Example:
        >>> train()
    """
    datasets = cifar_training_datasets()
    datasets['train'].augment = True

    datasets['train'].center_inputs = datasets['train']._make_normalizer('dependant')
    datasets['vali'].center_inputs = datasets['train'].center_inputs

    criterion = torch.nn.CrossEntropyLoss()

    # from clab.torch.models.densenet_efficient import DenseNetEfficient
    import clab.torch.models.densenet

    hyper = hyperparams.HyperParams(
        # criterion=(torch.nn.CrossEntropyLoss, {
        #     # 'ignore_label': ignore_label,
        #     # TODO: weight should be a FloatTensor
        #     # 'weight': class_weights,
        # }),

        model=(clab.torch.models.densenet.DenseNet, {
            'cifar': True,
            'num_classes': datasets['train'].n_classes,
        }),
        optimizer=(torch.optim.SGD, {
            # 'weight_decay': .0006,
            'weight_decay': .0005,
            'momentum': 0.9,
            'nesterov': True,
            'lr': 0.001,
        }),
        scheduler=('ReduceLROnPlateau', {
            # 'gamma': 0.99,
            # 'base_lr': 0.001,
            # 'stepsize': 2,
        }),
        # Specify anything else that is special about your hyperparams here
        # Especially if you make a custom_batch_runner
        other={
            'augment': datasets['train'].augment,
            'colorspace': datasets['train'].colorspace,
            'n_classes': datasets['train'].n_classes,
            'criterion': 'cross_entropy',
        },
        init_method='he',
    )

    xpu = xpu_device.XPU.from_argv()

    # TODO: need something to auto-generate a cachable directory structure
    # train_dpath = ub.ensuredir('train_cifar_dense')

    batch_size = 32
    harn = fit_harness.FitHarness(
        hyper=hyper, datasets=datasets, xpu=xpu, batch_size=batch_size,
    )

    @harn.set_batch_runner
    def batch_runner(harn, inputs, labels):
        # Forward pass and compute the loss
        output = harn.model(*inputs)
        # Compute the loss
        label = labels[0]
        # loss = harn.criterion(output, label)
        loss = criterion(output, label)
        return [output], loss

    workdir = ub.ensuredir('train_cifar_work')
    harn.setup_dpath(workdir)

    harn.run()


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m clab.live.cifar_train train
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
