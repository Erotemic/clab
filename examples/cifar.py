import numpy as np
import ubelt as ub
import torch
import torchvision
import pandas as pd
from torchvision.datasets import cifar
from clab.torch import xpu_device
from clab.torch import early_stop
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

    def convert_colorspace(inputs, colorspace, inplace=False):
        if colorspace.lower() == inputs.colorspace.lower():
            if not inplace:
                return inputs.im
            return
        im_out = np.empty_like(inputs.im)
        dst = np.ascontiguousarray(np.empty_like(inputs.im[0]))
        for ix, im in enumerate(inputs.im):
            util.convert_colorspace(im, src_space=inputs.colorspace,
                                    dst_space=colorspace, dst=dst)
            im_out[ix] = dst
        if inplace:
            inputs.im = im_out
            inputs.colorspace = colorspace
        else:
            return im_out

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
    def __init__(dset, inputs, task, workdir, output_colorspace='RGB'):
        dset.inputs = inputs
        dset.task = task

        dset.output_colorspace = output_colorspace

        dset.rng = np.random.RandomState(432432)

        inputs_base = ub.ensuredir((workdir, 'inputs'))
        inputs.base_dpath = inputs_base
        if len(inputs):
            inputs.prepare_id()
            dset.input_id = inputs.input_id
            dset.with_gt = dset.inputs.gt is not None
        else:
            dset.input_id = ''

        dset.augment = None
        dset.im_augment = torchvision.transforms.Compose([
            RandomGamma(rng=dset.rng),
            RandomBlur(rng=dset.rng),
        ])
        dset.rand_aff = RandomWarpAffine(dset.rng)
        dset.center_inputs = None

    def _make_normalizer(dset, mode='dependant'):
        """
        Example:
            >>> inputs = cifar_test_inputs()
            >>> task = CIFAR100_Task()
            >>> workdir = ub.ensuredir(ub.truepath('~/data/work/cifar'))
            >>> dset = CIFAR_Wrapper(inputs, task, workdir, 'LAB')
            >>> center_inputs = dset._make_normalizer('independent')
        """
        if len(dset.inputs):
            # compute normalizers in the output colorspace
            out_im = dset.inputs.convert_colorspace(dset.output_colorspace,
                                                    inplace=False)
            if mode == 'dependant':
                # dependent centering per channel (for RGB)
                im_mean = out_im.mean()
                im_scale = out_im.std()
            elif mode == 'independent':
                # Independent centering per channel (for LAB)
                im_mean = out_im.mean(axis=(0, 1, 2))
                im_scale = out_im.std(axis=(0, 1, 2))

            center_inputs = ImageCenterScale(im_mean, im_scale)

        dset.center_inputs = center_inputs
        return center_inputs

    def __len__(dset):
        return len(dset.inputs)

    def load_inputs(dset, index):
        """

        Ignore:
            >>> inputs = cifar_test_inputs()
            >>> task = CIFAR100_Task()
            >>> workdir = ub.ensuredir(ub.truepath('~/data/work/cifar'))
            >>> dset = CIFAR_Wrapper(inputs, task, workdir, 'LAB')
            >>> dset._make_normalizer('independent')
            >>> index = 0
            >>> im, gt = dset.load_inputs(index)
        """
        assert dset.inputs.colorspace.lower() == 'rgb', (
            'we must be in rgb for augmentation')
        im = dset.inputs.im[index]

        if dset.inputs.gt is not None:
            gt = dset.inputs.gt[index]
        else:
            gt = None

        if dset.augment:
            # Image augmentation must be done in RGB
            # Augment intensity independently
            im = dset.im_augment(im)
            # Augment geometry consistently
            params = dset.rand_aff.random_params()
            im = dset.rand_aff.warp(im, params, interp='cubic', backend='cv2')

        im = util.convert_colorspace(im, src_space=dset.inputs.colorspace,
                                     dst_space=dset.output_colorspace)
        # Do centering of inputs
        im = dset.center_inputs(im)
        return im, gt

    def __getitem__(dset, index):
        from clab.torch import im_loaders
        im, gt = dset.load_inputs(index)
        input_tensor = im_loaders.numpy_image_to_float_tensor(im)

        if dset.with_gt:
            # print('gotitem: ' + str(data_tensor.shape))
            # print('gt_tensor: ' + str(gt_tensor.shape))
            return input_tensor, gt
        else:
            return input_tensor

    @property
    def n_channels(dset):
        return 3

    @property
    def n_classes(dset):
        return int(dset.task.labels.max() + 1)

    @property
    def ignore_labels(dset):
        return dset.task.ignore_labels

    def class_weights(dset):
        """
            >>> from clab.live.sseg_train import *
            >>> dset = load_task_dataset('urban_mapper_3d')['train']
            >>> dset.class_weights()
        """
        # # Handle class weights
        # print('prep class weights')
        # gtstats = dset.inputs.prepare_gtstats(dset.task)
        # gtstats = dset.inputs.gtstats
        # # Take class weights (ensure they are in the same order as labels)
        # mfweight_dict = gtstats['mf_weight'].to_dict()
        # class_weights = np.array(list(ub.take(mfweight_dict, dset.task.classnames)))
        # class_weights[dset.task.ignore_labels] = 0
        # # HACK
        # # class_weights[0] = 1.0
        # # class_weights[1] = 0.7
        # print('class_weights = {!r}'.format(class_weights))
        # print('class_names   = {!r}'.format(dset.task.classnames))
        class_weights = np.ones(dset.n_classes)
        return class_weights


def cifar_inputs(train=False):
    root = ub.ensure_app_cache_dir('clab')
    train_dset = cifar.CIFAR100(root=root, download=True, train=train)
    if train:
        bchw = (train_dset.train_data).astype(np.float32) / 255.0
        labels = np.array(train_dset.train_labels)
    else:
        bchw = (train_dset.test_data).astype(np.float32) / 255.0
        labels = np.array(train_dset.test_labels)
    inputs = InMemoryInputs.from_bhwc_rgb(bchw, labels=labels)
    if train:
        inputs.tag = 'learn'
    else:
        inputs.tag = 'test'
    return inputs


def cifar_training_datasets(output_colorspace='RGB', norm_mode='independent'):
    """
    Example:
        >>> datasets = cifar_training_datasets()
    """
    inputs = cifar_inputs(train=True)

    # split training into train / validation
    vali_frac = .1  # 10%  is 5K images
    n_vali = int(len(inputs) * vali_frac)
    # n_vali = 10000  # 10K validation as in http://torch.ch/blog/2015/07/30/cifar.html

    input_idxs = util.random_indices(len(inputs), seed=1184576173)
    train_idxs = sorted(input_idxs[:-n_vali])
    vali_idxs = sorted(input_idxs[-n_vali:])

    train_inputs = inputs.take(train_idxs, tag='train')
    vali_inputs = inputs.take(vali_idxs, tag='vali')
    test_inputs = cifar_inputs(train=False)
    # The dataset name and indices should fully specifiy dependencies
    train_inputs._set_id_from_dependency(['cifar100-train', train_idxs])
    vali_inputs._set_id_from_dependency(['cifar100-train', vali_idxs])
    test_inputs._set_id_from_dependency(['cifar100-test'])

    workdir = ub.ensuredir(ub.truepath('~/data/work/cifar'))

    task = CIFAR100_Task()

    train_dset = CIFAR_Wrapper(train_inputs, task, workdir, output_colorspace=output_colorspace)
    vali_dset = CIFAR_Wrapper(vali_inputs, task, workdir, output_colorspace=output_colorspace)
    test_dset = CIFAR_Wrapper(test_inputs, task, workdir,
                              output_colorspace=output_colorspace)
    print('built datasets')

    datasets = {
        'train': train_dset,
        'vali': vali_dset,
        'test': test_dset,
    }

    print('computing normalizers')
    datasets['train'].center_inputs = datasets['train']._make_normalizer(norm_mode)
    for key in datasets.keys():
        datasets[key].center_inputs = datasets['train'].center_inputs
    print('computed normalizers')

    datasets['train'].augment = True
    return datasets


def train():
    """
    Example:
        >>> train()
    """
    import random
    np.random.seed(1031726816 % 4294967295)
    torch.manual_seed(137852547 % 4294967295)
    random.seed(2497950049 % 4294967295)

    if ub.argflag('--lab'):
        datasets = cifar_training_datasets(output_colorspace='LAB', norm_mode='independent')
    elif ub.argflag('--rgb-indie'):
        datasets = cifar_training_datasets(output_colorspace='RGB', norm_mode='independent')
    else:
        datasets = cifar_training_datasets(output_colorspace='RGB', norm_mode='dependant')

    import clab.torch.models.densenet

    hyper = hyperparams.HyperParams(
        model=(clab.torch.models.densenet.DenseNet, {
            'cifar': True,
            'num_classes': datasets['train'].n_classes,
        }),
        optimizer=(torch.optim.SGD, {
            'weight_decay': .0005,
            'momentum': 0.9,
            'nesterov': True,
            'lr': 0.01,
        }),
        scheduler=(torch.optim.lr_scheduler.ReduceLROnPlateau, {
        }),
        initializer=(nninit.KaimingNormal, {
            'nonlinearity': 'relu',
        }),
        criterion=(torch.nn.CrossEntropyLoss, {
        }),
        # Specify anything else that is special about your hyperparams here
        # Especially if you make a custom_batch_runner
        other={
            'augment': datasets['train'].augment,
            'colorspace': datasets['train'].output_colorspace,
            # 'n_classes': datasets['train'].n_classes,
        },
    )
    if ub.argflag('--rgb-indie'):
        hyper.other['norm'] = 'dependant'
    hyper.input_ids['train'] = datasets['train'].input_id

    xpu = xpu_device.XPU.from_argv()

    batch_size = 128
    data_kw = {'batch_size': batch_size}
    if xpu.is_gpu():
        data_kw.update({'num_workers': 8, 'pin_memory': True})

    tags = ['train', 'vali', 'test']

    loaders = ub.odict()
    for tag in tags:
        dset = datasets[tag]
        shuffle = tag == 'train'
        data_kw_ = data_kw.copy()
        if tag != 'train':
            data_kw_['batch_size'] = max(batch_size // 4, 1)
        loader = torch.utils.data.DataLoader(dset, shuffle=shuffle, **data_kw_)
        loaders[tag] = loader

    harn = fit_harness.FitHarness(
        hyper=hyper, datasets=datasets, xpu=xpu,
        loaders=loaders,
    )
    harn.monitor = early_stop.EarlyStop(patience=40)

    @harn.set_batch_runner
    def batch_runner(harn, inputs, labels):
        """
        Custom function to compute the output of a batch and its loss.
        """
        output = harn.model(*inputs)
        label = labels[0]
        loss = harn.criterion(output, label)
        outputs = [output]
        return outputs, loss

    task = harn.datasets['train'].task
    all_labels = task.labels
    # ignore_label = datasets['train'].ignore_label
    # from clab.torch import metrics
    from clab.metrics import (confusion_matrix,
                              pixel_accuracy_from_confusion,
                              pixel_accuracy_from_confusion)

    @harn.add_metric_hook
    def custom_metrics(harn, outputs, labels):
        label = labels[0]
        output = outputs[0]

        y_pred = output.data.max(dim=1)[1].cpu().numpy()
        y_true = label.data.cpu.numpy()

        cfsn = confusion_matrix(y_pred, y_true, labels=all_labels)

        global_tpr = pixel_accuracy_from_confusion(cfsn)  # same as tpr
        perclass_acc = perclass_accuracy_from_confusion(cfsn)
        class_accuracy = perclass_acc.fillna(0).mean()

        metrics_dict = ub.odict()
        metrics_dict['global_tpr'] = global_tpr
        metrics_dict['class_tpr'] = class_accuracy
        return metrics_dict

    workdir = ub.ensuredir('train_cifar_work')
    harn.setup_dpath(workdir)

    harn.run()


if __name__ == '__main__':
    r"""
    CommandLine:
        python examples/cifar.py train
        python examples/cifar.py train --lab
        python examples/cifar.py train --rgb-indie
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
