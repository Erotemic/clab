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
from torch.utils.data import Dataset


class VOCDataset(Dataset):
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
        >>> datadir = ub.truepath('~/data/VOC')
        >>> self = VOCDataset(datadir)
        >>> for i in range(10):
        ...     a, bc = self[i]
        ...     #print(bc[0].shape)
        ...     print(bc[1].shape)
        ...     print(a.shape)
    """
    def __init__(self, datadir=None, split='train'):
        if datadir is None:
            datadir = ub.truepath('~/data/VOC')

        data_dpath = join(datadir, 'VOCdevkit', 'VOC2007')
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

        # self.gpaths = sorted(glob.glob(join(image_dpath, '*.jpg')))
        # self.apaths = sorted(glob.glob(join(annot_dpath, '*.xml')))

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

    def make_loader(self, **kwargs):
        """
        We need to do special collation to deal with different numbers of
        bboxes per item.

        References:
            https://github.com/pytorch/pytorch/issues/1512

        Example:
            >>> datadir = ub.truepath('~/data/VOC')
            >>> self = VOCDataset(datadir)
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
        def collate_fn(inbatch):
            # we know the order of data in __getitem__ so we can choose not to
            # stack the variable length bboxes and labels
            inimgs, inlabels = list(map(list, zip(*inbatch)))
            imgs = default_collate(inimgs)
            labels = list(map(list, zip(*inlabels)))
            batch = imgs, labels
            return batch
        loader = DataLoader(self, collate_fn=collate_fn, **kwargs)
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


def train():
    from clab.models.yolo2 import darknet
    datasets = {
        'train': VOCDataset(split='train'),
        'vali': VOCDataset(split='val'),
    }
    batch_size = 4
    workers = 0

    # TODO: custom sampler that does multiscale training
    loaders = {
        key: dset.make_loader(batch_size=batch_size, num_workers=workers)
        for key, dset in datasets.items()
    }

    dset = datasets['train']
    model = (darknet.Darknet19,
             {'num_classes': dset.num_classes,
              'anchors': dset.anchors})
    criterion = (darknet.DarknetLoss, {'anchors': dset.anchors})

    from clab import hyperparams
    hyper = hyperparams.HyperParams(
        model=model,
        criterion=criterion,
    )
