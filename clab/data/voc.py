"""
Simple dataset for loading the VOC 2007 object detection dataset without extra
bells and whistles. Simply loads the images, boxes, and class labels and
resizes images to a standard size.
"""
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

        self.num_classes = len(self.label_names)
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
            default_collate = torch_data.dataloader.default_collate
            inimgs, inlabels = list(map(list, zip(*inbatch)))
            imgs = default_collate(inimgs)

            # Just transpose the list if we cant collate the labels
            # However, try to collage each part.
            n_labels = len(inlabels[0])
            labels = [None] * n_labels
            for i in range(n_labels):
                simple = [x[i] for x in inlabels]
                if ub.allsame(map(len, simple)):
                    labels[i] = default_collate(simple)
                else:
                    labels[i] = simple

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
            bbox is in x1,y1,x2,y2 (i.e. tlbr) format

        Example:
            >>> self = VOCDataset()
            >>> chw, label = self[1]
            >>> hwc = chw.numpy().transpose(1, 2, 0)
            >>> boxes, class_idxs = label
            >>> # xdoc: +REQUIRES(--show)
            >>> from clab.util import mplutil
            >>> mplutil.qtensure()  # xdoc: +SKIP
            >>> mplutil.imshow(hwc, colorspace='rgb')
            >>> mplutil.draw_boxes(boxes.numpy(), box_format='tlbr')
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

    def _load_item(self, index, inp_size=None):
        # from clab.models.yolo2.utils.yolo import _offset_boxes
        image = self._load_image(index)
        annot = self._load_annotation(index)

        boxes = annot['boxes'].astype(np.float32)
        gt_classes = annot['gt_classes']

        # squish the bounding box and image into a standard size
        if inp_size is None:
            return image, boxes, gt_classes
        else:
            w, h = inp_size
            sx = float(w) / image.shape[1]
            sy = float(h) / image.shape[0]
            boxes[:, 0::2] *= sx
            boxes[:, 1::2] *= sy
            interpolation = cv2.INTER_AREA if (sx + sy) <= 2 else cv2.INTER_CUBIC
            hwc = cv2.resize(image, (w, h), interpolation=interpolation)
            return hwc, boxes, gt_classes

    def _load_image(self, index):
        fpath = self.gpaths[index]
        imbgr = cv2.imread(fpath)
        imrgb_255 = cv2.cvtColor(imbgr, cv2.COLOR_BGR2RGB)
        return imrgb_255
        # imrgb_01 = imrgb.astype(np.float32) / 255.0

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


class EvaluateVOC(object):
    """
    Example:
        >>> all_true_boxes, all_pred_boxes = EvaluateVOC.demodata_boxes()
        >>> print(ub.repr2(all_true_boxes, nl=3, precision=2))
        >>> print(ub.repr2(all_pred_boxes, nl=3, precision=2))
        >>> self = EvaluateVOC(all_true_boxes, all_pred_boxes)
    """

    def __init__(self, all_true_boxes, all_pred_boxes):
        self.all_true_boxes = all_true_boxes
        self.all_pred_boxes = all_pred_boxes

    @classmethod
    def demodata_boxes(cls):
        all_true_boxes = [
            # class 1
            [
                # image 1
                [[100, 100, 200, 200]],
                # image 2
                np.empty((0, 4)),
                # image 3
                [[0, 10, 10, 20], [10, 10, 20, 20], [20, 10, 30, 20]],
            ],
            # class 2
            [
                # image 1
                [[0, 0, 100, 100], [0, 0, 50, 50]],
                # image 2
                [[0, 0, 50, 50], [50, 50, 100, 100]],
                # image 3
                [[0, 0, 10, 10], [10, 0, 20, 10], [20, 0, 30, 10]],
            ],
        ]
        # convert to numpy
        for cx, class_boxes in enumerate(all_true_boxes):
            for gx, boxes in enumerate(class_boxes):
                all_true_boxes[cx][gx] = np.array(boxes)

        # setup perterbubed demo predicted boxes
        rng = np.random.RandomState(0)

        def make_dummy(boxes, rng):
            if boxes.shape[0] == 0:
                boxes = np.array([[10, 10, 50, 50]])
            boxes = np.vstack([boxes, boxes])
            xywh = np.hstack([boxes[:, 0:2], boxes[:, 2:4] - boxes[:, 0:2]])
            scale = np.sqrt(xywh.max()) / 2
            pred_xywh = xywh + rng.randn(*xywh.shape) * scale
            keep = rng.rand(len(pred_xywh)) > 0.5
            pred = pred_xywh[keep].astype(np.uint8)
            pred_boxes = np.hstack([pred[:, 0:2], pred[:, 0:2] + pred[:, 2:4]])
            # give dummy scores
            pred_boxes = np.hstack([pred_boxes, rng.rand(len(pred_boxes), 1)])
            return pred_boxes

        all_pred_boxes = []
        for cx, class_boxes in enumerate(all_true_boxes):
            all_pred_boxes.append([])
            for gx, boxes in enumerate(class_boxes):
                pred_boxes = make_dummy(boxes, rng)
                all_pred_boxes[cx].append(pred_boxes)

        return all_true_boxes, all_pred_boxes

    @classmethod
    def find_overlap(cls, true_boxes, pred_box):
        """
        Compute iou of `pred_box` with each `true_box in true_boxes`.
        Return the index and score of the true box with maximum overlap.

        Example:
            >>> all_true_boxes, all_pred_boxes = EvaluateVOC.demodata_boxes()
            >>> true_boxes = np.array([[ 0,  0, 10, 10],
            >>>                        [10,  0, 20, 10],
            >>>                        [20,  0, 30, 10]])
            >>> pred_box = np.array([6, 2, 20, 10, .9])
            >>> ovmax, ovidx = EvaluateVOC.find_overlap(true_boxes, pred_box)
            >>> print('ovidx = {!r}'.format(ovidx))
            ovidx = 1
        """
        bb = pred_box
        # intersection
        ixmin = np.maximum(true_boxes[:, 0], bb[0])
        iymin = np.maximum(true_boxes[:, 1], bb[1])
        ixmax = np.minimum(true_boxes[:, 2], bb[2])
        iymax = np.minimum(true_boxes[:, 3], bb[3])
        iw = np.maximum(ixmax - ixmin + 1., 0.)
        ih = np.maximum(iymax - iymin + 1., 0.)
        inters = iw * ih

        # union
        uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
               (true_boxes[:, 2] - true_boxes[:, 0] + 1.) *
               (true_boxes[:, 3] - true_boxes[:, 1] + 1.) - inters)

        overlaps = inters / uni
        ovidx = overlaps.argmax()
        ovmax = overlaps[ovidx]
        return ovmax, ovidx

    def compute(self, ovthresh=0.5):
        """
        Example:
            >>> all_true_boxes, all_pred_boxes = EvaluateVOC.demodata_boxes()
            >>> self = EvaluateVOC(all_true_boxes, all_pred_boxes)
            >>> ovthresh = 0.5
            >>> mean_ap = self.compute(ovthresh)
            >>> print('mean_ap = {:.2f}'.format(mean_ap))
            mean_ap = 0.18
        """
        num_classes = len(self.all_true_boxes)
        ap_list = []
        for cx in range(num_classes):
            rec, prec, ap = self.eval_class(cx, ovthresh)
            ap_list.append(ap)
        mean_ap = np.mean(ap_list)
        return mean_ap

    def eval_class(self, cx, ovthresh=0.5):
        all_true_boxes = self.all_true_boxes
        all_pred_boxes = self.all_pred_boxes

        batch_true_boxes = all_true_boxes[cx]
        batch_pred_boxes = all_pred_boxes[cx]

        # Flatten the predicted boxes
        import pandas as pd
        flat_pred_boxes = []
        flat_pred_gxs = []
        for gx, pred_boxes in enumerate(batch_pred_boxes):
            flat_pred_boxes.extend(pred_boxes)
            flat_pred_gxs.extend([gx] * len(pred_boxes))
        flat_pred_boxes = np.array(flat_pred_boxes)
        if len(flat_pred_boxes) > 0:
            flat_preds = pd.DataFrame({
                'box': flat_pred_boxes[:, 0:4].tolist(),
                'conf': flat_pred_boxes[:, 4],
                'gx': flat_pred_gxs
            })
            flat_preds = flat_preds.sort_values('conf', ascending=False)

            # Keep track of which true boxes have been assigned in this class /
            # image pair.
            assign = {}

            # Greedy assignment for scoring
            # Iterate through bounding boxes in order of descending confidence
            nd = len(flat_preds)
            tp = np.zeros(nd)
            fp = np.zeros(nd)
            for sx, (pred_id, pred) in enumerate(flat_preds.iterrows()):
                gx, pred_box = pred[['gx', 'box']]
                true_boxes = batch_true_boxes[gx]
                ovmax = -np.inf
                true_id = None
                if len(true_boxes):
                    ovmax, ovidx = self.find_overlap(true_boxes, pred_box)
                    true_id = (gx, ovidx)

                if ovmax > ovthresh and true_id not in assign:
                    assign[true_id] = pred_id
                    tp[sx] = 1
                else:
                    fp[sx] = 1

            # compute precision recall
            npos = sum(map(len, batch_true_boxes))
            fp = np.cumsum(fp)
            tp = np.cumsum(tp)
            rec = tp / float(npos)
            # avoid divide by zero in case the first detection matches a difficult
            # ground truth
            prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)

            def voc_ap(rec, prec, use_07_metric=False):
                """ ap = voc_ap(rec, prec, [use_07_metric])
                Compute VOC AP given precision and recall.
                If use_07_metric is true, uses the
                VOC 07 11 point method (default:False).
                """
                if use_07_metric:
                    # 11 point metric
                    ap = 0.
                    for t in np.arange(0., 1.1, 0.1):
                        if np.sum(rec >= t) == 0:
                            p = 0
                        else:
                            p = np.max(prec[rec >= t])
                        ap = ap + p / 11.
                else:
                    # correct AP calculation
                    # first append sentinel values at the end
                    mrec = np.concatenate(([0.], rec, [1.]))
                    mpre = np.concatenate(([0.], prec, [0.]))

                    # compute the precision envelope
                    for i in range(mpre.size - 1, 0, -1):
                        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

                    # to calculate area under PR curve, look for points
                    # where X axis (recall) changes value
                    i = np.where(mrec[1:] != mrec[:-1])[0]

                    # and sum (\Delta recall) * prec
                    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
                return ap

            ap = voc_ap(rec, prec, use_07_metric=True)
            return rec, prec, ap
        else:
            return [], [], 0

    # === Original Method 1
    # def on_epoch1(harn, tag, loader):

    #     # Measure accumulated outputs
    #     num_images = len(loader.dataset)
    #     num_classes = loader.dataset.num_classes
    #     all_pred_boxes = [[[] for _ in range(num_images)]
    #                       for _ in range(num_classes)]
    #     all_true_boxes = [[[] for _ in range(num_images)]
    #                       for _ in range(num_classes)]

    #     # cx = 3
    #     # cx = 7
    #     print(ub.repr2([(gx, b) for gx, b in enumerate(all_true_boxes[cx]) if len(b)], nl=1))
    #     print(ub.repr2([(gx, b) for gx, b in enumerate(all_pred_boxes[cx]) if len(b)], nl=1))

    #     # Iterate over output from each batch
    #     for postout, labels in harn.accumulated:

    #         # Iterate over each item in the batch
    #         batch_pred_boxes, batch_pred_scores, batch_pred_cls_inds = postout
    #         batch_true_boxes, batch_true_cls_inds = labels[0:2]
    #         batch_orig_sz, batch_img_inds = labels[2:4]

    #         batch_size = len(labels[0])
    #         for bx in range(batch_size):
    #             gx = batch_img_inds[bx]

    #             true_boxes = batch_true_boxes[bx].data.cpu().numpy()
    #             true_cxs = batch_true_cls_inds[bx]

    #             pred_boxes  = batch_pred_boxes[bx]
    #             pred_scores = batch_pred_scores[bx]
    #             pred_cxs    = batch_pred_cls_inds[bx]

    #             for cx, boxes, score in zip(pred_cxs, pred_boxes, pred_scores):
    #                 all_pred_boxes[cx][gx].append(np.hstack([boxes, score]))

    #             for cx, boxes in zip(true_cxs, true_boxes):
    #                 all_true_boxes[cx][gx].append(boxes)

    #     all_boxes = all_true_boxes
    #     for cx, class_boxes in enumerate(all_boxes):
    #         for gx, boxes in enumerate(class_boxes):
    #             all_boxes[cx][gx] = np.array(boxes)
    #             if len(boxes):
    #                 boxes = np.array(boxes)
    #             else:
    #                 boxes = np.empty((0, 4))
    #             all_boxes[cx][gx] = boxes

    #     all_boxes = all_pred_boxes
    #     for cx, class_boxes in enumerate(all_boxes):
    #         for gx, boxes in enumerate(class_boxes):
    #             # Sort predictions by confidence
    #             if len(boxes):
    #                 boxes = np.array(boxes)
    #             else:
    #                 boxes = np.empty((0, 5))
    #             all_boxes[cx][gx] = boxes

    #     self = voc.EvaluateVOC(all_true_boxes, all_pred_boxes)
    #     ovthresh = 0.5
    #     mean_ap1 = self.compute(ovthresh)
    #     print('mean_ap1 = {!r}'.format(mean_ap1))

    #     num_classes = len(self.all_true_boxes)
    #     ap_list1 = []
    #     for cx in range(num_classes):
    #         rec, prec, ap = self.eval_class(cx, ovthresh)
    #         ap_list1.append(ap)
    #     print('ap_list1 = {!r}'.format(ap_list1))

    #     # reset accumulated for next epoch
    #     harn.accumulated.clear()


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m clab.data.voc
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
