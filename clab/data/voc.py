from os.path import exists
from os.path import join
import re
import scipy
import scipy.sparse
import cv2
import torch
import glob
import ubelt as ub
import numpy as np
import torch.utils.data as torch_data


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

        self.data_dpath = join(devkit_dpath, 'VOC2007')
        image_dpath = join(self.data_dpath, 'JPEGImages')
        annot_dpath = join(self.data_dpath, 'Annotations')

        # determine train / test splits

        split_idxs = self._read_split_indices(split)
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
        self.base_size = [320, 320]

        self.anchors = np.asarray([(1.08, 1.19), (3.42, 4.41),
                                   (6.63, 11.38), (9.42, 5.11),
                                   (16.62, 10.52)],
                                  dtype=np.float)

        self.num_classes = len(self.label_names)
        self.num_anchors = len(self.anchors)
        self.input_id = 'voc2007_' + self.split

    def _read_split_indices(self, split):
        split_idxs = []
        split_dpath = join(self.data_dpath, 'ImageSets', 'Main')
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
            index, inp_size = index
        else:
            inp_size = self.base_size
        hwc, boxes, gt_classes = self._load_item(index, inp_size)

        # TODO: augmentation
        chw = torch.FloatTensor(hwc.transpose(2, 0, 1))
        gt_classes = torch.LongTensor(gt_classes)
        boxes = torch.LongTensor(boxes.astype(np.int32))
        label = (boxes, gt_classes,)
        return chw, label

    def _load_item(self, index, inp_size):
        # from clab.models.yolo2.utils.yolo import _offset_boxes
        image = self._load_image(index)
        annot = self._load_annotation(index)

        boxes = annot['boxes'].astype(np.float32)
        gt_classes = annot['gt_classes']

        # squish the bounding box and image into a standard size
        w, h = inp_size
        sx = float(w) / image.shape[1]
        sy = float(h) / image.shape[0]
        boxes[:, 0::2] *= sx
        boxes[:, 1::2] *= sy

        # TODO: postpone resize until augmentation
        interpolation = cv2.INTER_AREA if (sx + sy) <= 2 else cv2.INTER_CUBIC
        hwc = cv2.resize(image, (w, h), interpolation=interpolation)
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
