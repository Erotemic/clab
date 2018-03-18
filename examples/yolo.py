"""
Need to compile yolo first.

Currently its hacked into the system.


pip install cffi
cd $HOME/code/clab/clab/models/yolo2
./make.sh
"""
from os.path import join
import scipy
import cv2
import torch
import glob
import ubelt as ub
import numpy as np
import torch.utils.data


class MultiScaleBatchSampler(torch.utils.data.BatchSampler):
    def __init__(self, data_source, shuffle=False, batch_size=16,
                 drop_last=True):
        if shuffle:
            self.sampler = torch.utils.data.RandomSampler(data_source)
        else:
            self.sampler = torch.utils.data.SequentialSampler(data_source)
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.num_scales = len(self.data_source.multi_scale_inp_size)

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


class VOCDataset(torch.utils.data.Dataset):
    """
    Notes:
        mkdir -p ~/data/VOC
        cd ~/data/VOC
        wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
        wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
        wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCdevkit_08-Jun-2007.tar

        tar xvf VOCtrainval_06-Nov-2007.tar
        tar xvf VOCtest_06-Nov-2007.tar
        tar xvf VOCdevkit_08-Jun-2007.tar

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
            devkit_dpath = ub.truepath('~/data/VOC/VOCdevkit')

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

    def make_loader(self, *args, **kwargs):
        """
        We need to do special collation to deal with different numbers of
        bboxes per item.

        Args:
            batch_size (int, optional): how many samples per batch to load
                (default: 1).
            shuffle (bool, optional): set to ``True`` to have the data reshuffled
                at every epoch (default: False).
            sampler (Sampler, optional): defines the strategy to draw samples from
                the dataset. If specified, ``shuffle`` must be False.
            batch_sampler (Sampler, optional): like sampler, but returns a batch of
                indices at a time. Mutually exclusive with batch_size, shuffle,
                sampler, and drop_last.
            num_workers (int, optional): how many subprocesses to use for data
                loading. 0 means that the data will be loaded in the main process.
                (default: 0)
            pin_memory (bool, optional): If ``True``, the data loader will copy tensors
                into CUDA pinned memory before returning them.
            drop_last (bool, optional): set to ``True`` to drop the last incomplete batch,
                if the dataset size is not divisible by the batch size. If ``False`` and
                the size of dataset is not divisible by the batch size, then the last batch
                will be smaller. (default: False)
            timeout (numeric, optional): if positive, the timeout value for collecting a batch
                from workers. Should always be non-negative. (default: 0)
            worker_init_fn (callable, optional): If not None, this will be called on each
                worker subprocess with the worker id (an int in ``[0, num_workers - 1]``) as
                input, after seeding and before data loading. (default: None)

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
        from torch.utils.data import DataLoader
        from torch.utils.data.dataloader import default_collate
        def custom_collate_fn(inbatch):
            # we know the order of data in __getitem__ so we can choose not to
            # stack the variable length bboxes and labels
            inimgs, inlabels = list(map(list, zip(*inbatch)))
            imgs = default_collate(inimgs)
            labels = list(map(list, zip(*inlabels)))
            batch = imgs, labels
            return batch
        kwargs['collate_fn'] = custom_collate_fn
        loader = DataLoader(self, *args, **kwargs)
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
        # size_index = randint(0, len(cfg.multi_scale_inp_size) - 1)
        if isinstance(index, tuple):
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
    loaders = {}
    for key, dset in datasets.items():
        batch_size = train_batch_size if key == 'train' else vali_batch_size
        # use custom sampler that does multiscale training
        batch_sampler = MultiScaleBatchSampler(
            dset, batch_size=batch_size, shuffle=(key == 'train')
        )
        loader = dset.make_loader(batch_sampler=batch_sampler,
                                  num_workers=workers)
        loaders[key] = loader
    return loaders


def train():
    from clab.models.yolo2 import darknet

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

        lr_mapping = {
            0: init_learning_rate * lr_decay ** 0,
            60: init_learning_rate * lr_decay ** 1,
            90: init_learning_rate * lr_decay ** 2,
        }

        workdir = ub.truepath('~/work/VOC2007')
        devkit_dpath = ub.truepath('~/data/VOC/VOCdevkit')

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
    from clab.lr_scheduler import ListedLR

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

        scheduler=(ListedLR, dict(
            mapping=cfg.lr_mapping
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
        loaders=loaders,
    )
    harn.monitor = monitor.Monitor(min_keys=['loss'],
                                   # max_keys=['global_acc', 'class_acc'],
                                   patience=40)

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

    # @harn.add_metric_hook
    # def custom_metrics(harn, outputs, labels):
    #     label = labels[0]
    #     output = outputs[0]

    #     y_pred = output.data.max(dim=1)[1].cpu().numpy()
    #     y_true = label.data.cpu().numpy()

    #     metrics_dict = ub.odict()
    #     metrics_dict['global_acc'] = global_acc
    #     metrics_dict['class_acc'] = class_accuracy
    #     return metrics_dict

    harn.setup_dpath(ub.ensuredir(cfg.workdir))
    harn.run()
