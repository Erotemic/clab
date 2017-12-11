import numpy as np
import ubelt as ub
import torchvision
import pandas as pd
from torchvision.datasets import cifar
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


class InMemoryInputs(object):
    """
    Change inputs.Inputs to OnDiskInputs
    """
    def __init__(inputs, tag=''):
        inputs.tag = tag
        inputs.im = None
        inputs.gt = None
        inputs.colorspace = None

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

    def prepare_images(self, force=False):
        """
        If not already done, loads paths to images into memory and constructs a
        unique id for that set of im/gt images.

        It the paths are already set, then only the input-id is constructed.
        """
        if self.n_input is not None and not force:
            return

        self.prepare_image_paths()
        print('Preparing {} images'.format(self.tag))

        if self.aux_paths:
            # new way
            depends = sorted(self.paths.items())
        else:
            depends = []
            depends.append(self.im_paths)
            depends.append(self.gt_paths)
            if self.gt_paths:
                # HACK: We will assume image data depends only on the filename
                # HACK: be respectful of gt label changes (ignore aug)
                # stride>1 is faster but might break
                # stride=1 is the safest
                hashes = [
                    util.hash_file(p, stride=32)
                    for p in ub.ProgIter(self.gt_paths, label='hashing')
                    if 'aug' not in basename(p) and 'part' not in basename(p)
                ]
                label_hashid = util.hash_data(hashes)
                depends.append(label_hashid)
        n_im = None if self.im_paths is None else len(self.im_paths)
        n_gt = None if self.gt_paths is None else len(self.gt_paths)
        self.n_input = n_im or n_gt

        hashid = hashutil.hash_data(depends)[:self.abbrev]
        self.input_id = '{}-{}'.format(self.n_input, hashid)

        print(' * n_images = {}'.format(n_im))
        print(' * n_groundtruth = {}'.format(n_gt))
        print(' * input_id = {}'.format(self.input_id))


def cifar_training_datasets():
    root = ub.ensure_app_cache_dir('clab')
    train_dset = cifar.CIFAR100(root=root, download=True, train=True)
    bchw = (train_dset.train_data).astype(np.float32) / 255.0
    labels = np.array(train_dset.train_labels)
    inputs = InMemoryInputs.from_bhwc_rgb(bchw, labels=labels)
    # inputs.convert_colorspace('lab')

    vali_frac = .1
    n_vali = int(len(inputs) * vali_frac)
    input_idxs = np.arange(len(inputs))
    rng = np.random.RandomState(0)
    rng.shuffle(input_idxs)
    train_idxs = input_idxs[:-n_vali]
    vali_idxs = input_idxs[-n_vali:]

    train_inputs = inputs.take(train_idxs, tag='train')
    vali_inputs = inputs.take(vali_idxs, tag='vali')


class CIFAR_Wrapper(cifar.CIFAR10):
    def __init__(self, inputs, task, workdir):
        self.inputs = inputs
        self.task = task

        self.rng = np.random.RandomState(432432)

        inputs_base = ub.ensuredir((workdir, 'inputs'))
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

    def _make_normalizer(self):
        from .torch.transforms import (ImageCenterScale, DTMCenterScale,
                                             ZipTransforms)
        transforms = []
        nan_value = -32767.0  # hack: specific number for DTM
        if len(self.inputs):
            self.center_stats = self.inputs.prepare_center_stats(
                self.task, nan_value=nan_value, colorspace=self.colorspace)
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
                scalar_stats = self.center_stats['image']['simple']['image']
                im_mean = scalar_stats['mean']
                im_scale = scalar_stats['std']
                # self.im_center = ub.identity
            else:
                raise Exception()

            im_center = ImageCenterScale(im_mean, im_scale)
            transforms.append(im_center)

            # im_scale = np.ceil(channel_stats['max']) - np.floor(channel_stats['min'])

            if self.aux_keys:
                # Note: Internal stats are not gaurenteed to make sense outside
                # the DTM domain.
                internal_aux_stats = self.center_stats['aux']['internal']
                mean_deviation = internal_aux_stats['image']['mean_absdev_from_median']['mean']
                # zero the median on a per-chip basis, but use
                # the global internal_std to normalize extent
                # aux_std =
                aux_center = DTMCenterScale(mean_deviation,
                                            nan_value=nan_value, fill='median')
                transforms.append(aux_center)

        center_inputs = ZipTransforms(transforms)
        self.center_inputs = center_inputs
        return center_inputs

    def __len__(self):
        return len(self.inputs)

    def load_inputs(self, index):
        im_fpath = self.inputs.im_paths[index]
        if self.inputs.gt_paths:
            gt_fpath = self.inputs.gt_paths[index]
            gt = self.loader(gt_fpath, colorspace=None)
        else:
            gt = None

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
            im, aux_channels, gt = self.rand_aff.sseg_warp(
                im, aux_channels, gt)

        im = util.convert_colorspace(im, src_space='RGB',
                                     dst_space=self.colorspace)

        # Do centering of inputs
        input_tuple = [im] + aux_channels
        input_tuple = self.center_inputs(input_tuple)

        return input_tuple, gt

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
        input_tuple, gt = self.load_inputs(index)
        input_tuple, gt_tensor = self.to_tensor(input_tuple, gt)

        data_tensor = torch.cat(input_tuple, dim=0)

        if self.with_gt:
            # print('gotitem: ' + str(data_tensor.shape))
            # print('gt_tensor: ' + str(gt_tensor.shape))
            return data_tensor, gt_tensor
        else:
            return data_tensor

    @property
    def n_channels(self):
        if self.aux_keys:
            return 3 + len(self.aux_keys)
        else:
            return 3

    @property
    def n_classes(self):
        return int(self.task.labels.max() + 1)

    @property
    def ignore_label(self):
        return self.task.ignore_label

    def class_weights(self):
        """
            >>> from clab.live.sseg_train import *
            >>> self = load_task_dataset('urban_mapper_3d')['train']
            >>> self.class_weights()
        """
        # Handle class weights
        print('prep class weights')
        gtstats = self.inputs.prepare_gtstats(self.task)
        gtstats = self.inputs.gtstats
        # Take class weights (ensure they are in the same order as labels)
        mfweight_dict = gtstats['mf_weight'].to_dict()
        class_weights = np.array(list(ub.take(mfweight_dict, self.task.classnames)))
        class_weights[self.task.ignore_labels] = 0
        # HACK
        # class_weights[0] = 1.0
        # class_weights[1] = 0.7
        print('class_weights = {!r}'.format(class_weights))
        print('class_names   = {!r}'.format(self.task.classnames))
        return class_weights

def train():
    import ubelt as ub
    root = ub.ensure_app_cache_dir('clab')
    train_dset = cifar.CIFAR100(root=root, download=True, train=True)
    test_dest = cifar.CIFAR100(root=root, download=True, train=False)
