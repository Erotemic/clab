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
        >>> from torch.utils.data import DataLoader
        >>> loader = DataLoader(self, batch_size=16)
        >>> batch = next(iter(loader))
        >>> images, labels = batch
    """
    def __init__(self, datadir=None):
        if datadir is None:
            datadir = ub.truepath('~/data/VOC')

        data_dpath = join(datadir, 'VOCdevkit', 'VOC2007')
        image_dpath = join(data_dpath, 'JPEGImages')
        annot_dpath = join(data_dpath, 'Annotations')
        self.gpaths = sorted(glob.glob(join(image_dpath, '*.jpg')))
        self.apaths = sorted(glob.glob(join(annot_dpath, '*.xml')))

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

        self.num_classes = len(self.label_names)
        self.num_anchors = len(self.anchors)

    def __len__(self):
        return len(self.gpaths)

    def __getitem__(self, index):
        image = self._load_image(index)
        annot = self._load_annotation(index)

        # size_index = randint(0, len(cfg.multi_scale_inp_size) - 1)

        boxes = annot['boxes']
        gt_classes = annot['gt_classes']

        # TODO: augmentation

        chw = torch.FloatTensor(image.transpose(2, 0, 1))

        gt_classes = torch.LongTensor(gt_classes)
        boxes = torch.LongTensor(boxes.astype(np.int32))

        dontcare = []
        label = (boxes, gt_classes, dontcare)
        return chw, label

    def _load_image(self, index):
        fpath = self.gpaths[index]
        imbgr = cv2.imread(fpath)
        imrgb = cv2.cvtColor(imbgr, cv2.COLOR_BGR2RGB)
        imrgb_01 = imrgb.astype(np.float32) / 255.0
        # TODO: multi-scale training?
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
