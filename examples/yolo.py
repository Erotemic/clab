"""
Need to compile yolo first.

Currently its hacked into the system.


pip install cffi
cd $HOME/code/clab/clab/models/yolo2
./make.sh
"""
from os.path import exists
from os.path import join
import scipy
import scipy.sparse
import cv2
import torch
import glob
import ubelt as ub
import numpy as np
import torch.utils.data as torch_data
import torch.utils.data.sampler as torch_sampler


class MultiScaleBatchSampler(torch_sampler.BatchSampler):
    """
    Indicies returned in the batch are tuples indicating data index and scale
    index.

    Example:
        >>> class DummyDatset(torch_data.Dataset):
        >>>     def __init__(self):
        >>>         super(DummyDatset, self).__init__()
        >>>         self.multi_scale_inp_size = [1, 2, 3, 4]
        >>>     def __len__(self):
        >>>         return 34
        >>> batch_size = 16
        >>> data_source = DummyDatset()
        >>> rand = MultiScaleBatchSampler(data_source, shuffle=1)
        >>> seq = MultiScaleBatchSampler(data_source, shuffle=0)
        >>> rand_idxs = list(iter(rand))
        >>> seq_idxs = list(iter(seq))
        >>> assert len(rand_idxs[0]) == 16
        >>> assert len(rand_idxs[0][0]) == 2
        >>> assert len(rand_idxs[-1]) == 2
        >>> assert {len({x[1] for x in xs}) for xs in rand_idxs} == {1}
        >>> assert {x[1] for xs in seq_idxs for x in xs} == {0}
    """
    def __init__(self, data_source, shuffle=False, batch_size=16,
                 drop_last=False):
        if shuffle:
            self.sampler = torch_sampler.RandomSampler(data_source)
        else:
            self.sampler = torch_sampler.SequentialSampler(data_source)
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.num_scales = len(data_source.multi_scale_inp_size)

    def __iter__(self):
        batch = []
        if self.shuffle:
            scale_index = int(torch.rand(1) * self.num_scales)
        else:
            scale_index = 0

        for idx in self.sampler:
            batch.append((int(idx), scale_index))
            if len(batch) == self.batch_size:
                yield batch
                if self.shuffle:
                    # choose a new scale index
                    scale_index = int(torch.rand(1) * self.num_scales)
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch


class VOCDataset(torch_data.Dataset, ub.NiceRepr):
    """
    Example:
        >>> assert len(VOCDataset(split='train')) == 2501
        >>> assert len(VOCDataset(split='test')) == 4952
        >>> assert len(VOCDataset(split='val')) == 2510

    Example:
        >>> self = VOCDataset()
        >>> for i in range(10):
        ...     a, bc = self[i]
        ...     #print(bc[0].shape)
        ...     print(bc[1].shape)
        ...     print(a.shape)
    """
    def __init__(self, devkit_dpath=None, split='train'):
        if devkit_dpath is None:
            # ub.truepath('~/data/VOC/VOCdevkit')
            devkit_dpath = self.ensure_voc2007()

        data_dpath = join(devkit_dpath, 'VOC2007')
        image_dpath = join(data_dpath, 'JPEGImages')
        annot_dpath = join(data_dpath, 'Annotations')

        # determine train / test splits
        split_dpath = join(data_dpath, 'ImageSets', 'Main')

        def get_split(split):
            import re
            split_idxs = []
            pattern = join(split_dpath, '*_' + split + '.txt')
            for p in sorted(glob.glob(pattern)):
                rows = [list(map(int, re.split(' +', t)))
                        for t in ub.readfrom(p).split('\n') if t]
                # code = -1 if the image does not contain the object
                # code = 1 if the image contains at least one instance
                # code = 0 if the image contains only hard instances of the object
                idxs = [idx for idx, code in rows if code == 1]
                split_idxs.extend(idxs)
            return sorted(set(split_idxs))

        split_idxs = get_split(split)
        self.split = split
        self.gpaths = [join(image_dpath, '{:06d}.jpg'.format(idx))
                       for idx in split_idxs]
        self.apaths = [join(annot_dpath, '{:06d}.xml'.format(idx))
                       for idx in split_idxs]

        self.label_names = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                            'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
                            'dog', 'horse', 'motorbike', 'person',
                            'pottedplant', 'sheep', 'sofa', 'train',
                            'tvmonitor')
        self._class_to_ind = ub.invert_dict(dict(enumerate(self.label_names)))

        self.anchors = np.asarray([(1.08, 1.19), (3.42, 4.41),
                                   (6.63, 11.38), (9.42, 5.11),
                                   (16.62, 10.52)],
                                  dtype=np.float)

        base_wh = np.array([320, 320], dtype=np.int)
        self.multi_scale_inp_size = [base_wh + (32 * i) for i in range(8)]
        self.multi_scale_out_size = [s / 32 for s in self.multi_scale_inp_size]

        self.num_classes = len(self.label_names)
        self.num_anchors = len(self.anchors)
        self.input_id = 'voc2007_' + self.split

    @classmethod
    def ensure_voc2007(cls, dpath=None, force=False):
        """
        Download the Pascal VOC 2007 data if it does not already exist.
        """
        if dpath is None:
            dpath = ub.truepath('~/data/VOC')
        devkit_dpath = join(dpath, 'VOCdevkit')
        if force or not exists(devkit_dpath):
            ub.ensuredir(dpath)
            fpath1 = ub.grabdata('http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar', dpath=dpath)
            fpath2 = ub.grabdata('http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar', dpath=dpath)
            fpath3 = ub.grabdata('http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCdevkit_08-Jun-2007.tar', dpath=dpath)

            ub.cmd('tar xvf "{}" -C "{}"'.format(fpath1, dpath), verbout=1)
            ub.cmd('tar xvf "{}" -C "{}"'.format(fpath2, dpath), verbout=1)
            ub.cmd('tar xvf "{}" -C "{}"'.format(fpath3, dpath), verbout=1)
        return devkit_dpath

    def __nice__(self):
        return '{} {}'.format(self.split, len(self))

    def make_loader(self, *args, **kwargs):
        """
        We need to do special collation to deal with different numbers of
        bboxes per item.

        Args:
            batch_size (int, optional): how many samples per batch to load
                (default: 1).
            shuffle (bool, optional): set to ``True`` to have the data
                reshuffled at every epoch (default: False).
            sampler (Sampler, optional): defines the strategy to draw samples
                from the dataset. If specified, ``shuffle`` must be False.
            batch_sampler (Sampler, optional): like sampler, but returns a
                batch of indices at a time. Mutually exclusive with batch_size,
                shuffle, sampler, and drop_last.
            num_workers (int, optional): how many subprocesses to use for data
                loading. 0 means that the data will be loaded in the main
                process.  (default: 0)
            pin_memory (bool, optional): If ``True``, the data loader will copy
                tensors into CUDA pinned memory before returning them.
            drop_last (bool, optional): set to ``True`` to drop the last
                incomplete batch, if the dataset size is not divisible by the
                batch size. If ``False`` and the size of dataset is not
                divisible by the batch size, then the last batch will be
                smaller. (default: False)
            timeout (numeric, optional): if positive, the timeout value for
                collecting a batch from workers. Should always be non-negative.
                (default: 0)
            worker_init_fn (callable, optional): If not None, this will be
                called on each worker subprocess with the worker id (an int in
                ``[0, num_workers - 1]``) as input, after seeding and before
                data loading. (default: None)

        References:
            https://github.com/pytorch/pytorch/issues/1512

        Example:
            >>> self = VOCDataset()
            >>> #inbatch = [self[i] for i in range(10)]
            >>> loader = self.make_loader(batch_size=10)
            >>> batch = next(iter(loader))
            >>> images, labels = batch
            >>> assert len(images) == 10
            >>> assert len(labels) == 2
            >>> assert len(labels[0]) == len(images)
        """
        def custom_collate_fn(inbatch):
            # we know the order of data in __getitem__ so we can choose not to
            # stack the variable length bboxes and labels
            inimgs, inlabels = list(map(list, zip(*inbatch)))
            imgs = torch_data.dataloader.default_collate(inimgs)
            labels = list(map(list, zip(*inlabels)))
            batch = imgs, labels
            return batch
        kwargs['collate_fn'] = custom_collate_fn
        loader = torch_data.DataLoader(self, *args, **kwargs)
        return loader

    def __len__(self):
        return len(self.gpaths)

    def __getitem__(self, index):
        """
        Returns:
            image, (bbox, class_idxs)

            bbox and class_idxs are variable-length
            bbox is in x1,y1,x2,y2 format
        """
        if isinstance(index, tuple):
            # Get size index from the batch loader
            index, size_index = index
        else:
            size_index = 0
        hwc, boxes, gt_classes = self._load_item(index, size_index)

        # TODO: augmentation
        chw = torch.FloatTensor(hwc.transpose(2, 0, 1))
        gt_classes = torch.LongTensor(gt_classes)
        boxes = torch.LongTensor(boxes.astype(np.int32))
        label = (boxes, gt_classes,)
        return chw, label

    def _load_item(self, index, size_index=0):
        # from clab.models.yolo2.utils.yolo import _offset_boxes
        image = self._load_image(index)
        annot = self._load_annotation(index)

        boxes = annot['boxes'].astype(np.float32)
        gt_classes = annot['gt_classes']

        inp_size = self.multi_scale_inp_size[size_index]

        # squish the bounding box and image into a standard size
        w, h = inp_size
        boxes[:, 0::2] *= float(w) / image.shape[1]
        boxes[:, 1::2] *= float(h) / image.shape[0]
        hwc = cv2.resize(image, (w, h))
        return hwc, boxes, gt_classes

    def _load_image(self, index):
        fpath = self.gpaths[index]
        imbgr = cv2.imread(fpath)
        imrgb = cv2.cvtColor(imbgr, cv2.COLOR_BGR2RGB)
        imrgb_01 = imrgb.astype(np.float32) / 255.0
        return imrgb_01

    def _load_annotation(self, index):
        import xml.etree.ElementTree as ET
        fpath = self.apaths[index]
        tree = ET.parse(fpath)
        objs = tree.findall('object')

        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        # "Seg" area for pascal is just the box area
        seg_areas = np.zeros((num_objs), dtype=np.float32)
        ishards = np.zeros((num_objs), dtype=np.int32)

        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):
            bbox = obj.find('bndbox')
            # Make pixel indexes 0-based
            x1 = float(bbox.find('xmin').text) - 1
            y1 = float(bbox.find('ymin').text) - 1
            x2 = float(bbox.find('xmax').text) - 1
            y2 = float(bbox.find('ymax').text) - 1

            diffc = obj.find('difficult')
            difficult = 0 if diffc is None else int(diffc.text)
            ishards[ix] = difficult

            clsname = obj.find('name').text.lower().strip()
            cls = self._class_to_ind[clsname]
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0
            seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)

        overlaps = scipy.sparse.csr_matrix(overlaps)

        annots = {'boxes': boxes,
                  'gt_classes': gt_classes,
                  'gt_ishard': ishards,
                  'gt_overlaps': overlaps,
                  'flipped': False,
                  'seg_areas': seg_areas}
        return annots


def make_loaders(datasets, train_batch_size=16, vali_batch_size=1, workers=0):
    """
    Example:
        >>> datasets = {'train': VOCDataset(split='train'),
        >>>             'vali': VOCDataset(split='val')}
        >>> torch.random.manual_seed(0)
        >>> loaders = make_loaders(datasets)
        >>> train_iter = iter(loaders['train'])
        >>> # training batches should have multiple shapes
        >>> shapes = set()
        >>> for batch in train_iter:
        >>>     shapes.add(batch[0].shape[-1])
        >>>     if len(shapes) > 1:
        >>>         break
        >>> assert len(shapes) > 1

        >>> vali_loader = iter(loaders['vali'])
        >>> vali_iter = iter(loaders['vali'])
        >>> # vali batches should have one shape
        >>> shapes = set()
        >>> for batch, _ in zip(vali_iter, [1, 2, 3, 4]):
        >>>     shapes.add(batch[0].shape[-1])
        >>> assert len(shapes) == 1
    """
    loaders = {}
    for key, dset in datasets.items():
        assert len(dset) > 0, 'must have some data'
        batch_size = train_batch_size if key == 'train' else vali_batch_size
        # use custom sampler that does multiscale training
        batch_sampler = MultiScaleBatchSampler(
            dset, batch_size=batch_size, shuffle=(key == 'train')
        )
        loader = dset.make_loader(batch_sampler=batch_sampler,
                                  num_workers=workers)
        loaders[key] = loader
    return loaders


def grab_darknet19_initial_weights():
    # TODO: setup a global url
    import ubelt as ub
    url = 'http://acidalia.kitware.com:8000/weights/darknet19.weights.npz'
    npz_fpath = ub.grabdata(url, dpath=ub.ensure_app_cache_dir('clab'))
    torch_fpath = ub.augpath(npz_fpath, ext='.pt')
    if not exists(torch_fpath):
        from clab.models.yolo2 import darknet
        # hack to transform initial state
        model = darknet.Darknet19(num_classes=20)
        model.load_from_npz(npz_fpath, num_conv=18)
        torch.save(model.state_dict(), torch_fpath)
    return torch_fpath


class cfg(object):
    start_step = 0
    lr_decay = 1. / 10

    workers = 0

    max_epoch = 160

    weight_decay = 0.0005
    momentum = 0.9
    init_learning_rate = 1e-3

    # for training yolo2
    # object_scale = 5.
    # noobject_scale = 1.
    # class_scale = 1.
    # coord_scale = 1.
    # iou_thresh = 0.6

    # dataset
    vali_batch_size = 1
    train_batch_size = 16

    lr_step_points = {
        0: init_learning_rate * lr_decay ** 0,
        60: init_learning_rate * lr_decay ** 1,
        90: init_learning_rate * lr_decay ** 2,
    }

    workdir = ub.truepath('~/work/VOC2007')
    devkit_dpath = ub.truepath('~/data/VOC/VOCdevkit')


def setup_harness():
    cfg.pretrained_fpath = grab_darknet19_initial_weights()
    datasets = {
        'train': VOCDataset(cfg.devkit_dpath, split='train'),
        'vali': VOCDataset(cfg.devkit_dpath, split='val'),
    }

    loaders = make_loaders(datasets,
                           train_batch_size=cfg.train_batch_size,
                           vali_batch_size=cfg.vali_batch_size,
                           workers=cfg.workers)

    from clab import hyperparams
    from clab import xpu_device
    from clab import fit_harness
    from clab import monitor
    from clab import nninit
    from clab.lr_scheduler import ListedLR
    from clab.models.yolo2 import darknet

    hyper = hyperparams.HyperParams(

        model=(darknet.Darknet19, {
            'num_classes': datasets['train'].num_classes,
            'anchors': datasets['train'].anchors
        }),

        criterion=(darknet.DarknetLoss, {
            'anchors': datasets['train'].anchors
        }),

        optimizer=(torch.optim.SGD, dict(
            lr=cfg.init_learning_rate,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay
        )),

        # initializer=(nninit.KaimingNormal, {}),
        initializer=(nninit.Pretrained, {
            'fpath': cfg.pretrained_fpath,
        }),

        scheduler=(ListedLR, dict(
            step_points=cfg.lr_step_points
        )),

        other={
            'train_batch_size': cfg.train_batch_size,
        },
        centering=None,
        # centering=datasets['train'].centering,
        # augmenter=datasets['train'].augmenter,
    )

    xpu = xpu_device.XPU.cast('auto')
    harn = fit_harness.FitHarness(
        hyper=hyper, datasets=datasets, xpu=xpu,
        loaders=loaders, max_iter=100,
    )
    harn.monitor = monitor.Monitor(min_keys=['loss'],
                                   # max_keys=['global_acc', 'class_acc'],
                                   patience=100)

    @harn.set_batch_runner
    def batch_runner(harn, inputs, labels):
        """
        Custom function to compute the output of a batch and its loss.

        Example:
            >>> import sys
            >>> sys.path.append('/home/joncrall/code/clab/examples')
            >>> from yolo import *
            >>> harn = setup_harness()
            >>> harn.initialize_training()
            >>> batch = harn._demo_batch(0, 'train')
            >>> inputs, labels = batch
            >>> criterion = harn.criterion
        """
        outputs = harn.model(*inputs)

        # darknet criterion needs to know the input image shape
        inp_size = tuple(inputs[0].shape[-2:])

        # assert np.sqrt(outputs[1].shape[1]) == inp_size[0] / 32

        bbox_pred, iou_pred, prob_pred = outputs
        gt_boxes, gt_classes = labels
        dontcare = np.array([[]] * len(gt_boxes))

        loss = harn.criterion(*outputs, *labels, dontcare=dontcare,
                              inp_size=inp_size)
        return outputs, loss

    @harn.add_metric_hook
    def custom_metrics(harn, outputs, labels):
        # label = labels[0]
        # output = outputs[0]

        # y_pred = output.data.max(dim=1)[1].cpu().numpy()
        # y_true = label.data.cpu().numpy()

        metrics_dict = ub.odict()
        metrics_dict['L_bbox'] = float(harn.criterion.bbox_loss.data.cpu().numpy())
        metrics_dict['L_iou'] = float(harn.criterion.iou_loss.data.cpu().numpy())
        metrics_dict['L_cls'] = float(harn.criterion.cls_loss.data.cpu().numpy())
        # metrics_dict['global_acc'] = global_acc
        # metrics_dict['class_acc'] = class_accuracy
        return metrics_dict
    return harn


def train():
    """
    python ~/code/clab/examples/yolo.py train
    """
    harn = setup_harness()
    harn.setup_dpath(ub.ensuredir(cfg.workdir))
    harn.run()


if __name__ == '__main__':
    r"""
    CommandLine:
        export PYTHONPATH=$PYTHONPATH:/home/joncrall/code/clab/examples
        python ~/code/clab/examples/yolo.py
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
