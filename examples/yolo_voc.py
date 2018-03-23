"""
Need to compile yolo first.

Currently setup is a bit hacked.

pip install cffi
python setup.py build_ext --inplace
"""
from clab.util import profiler  # NOQA
import psutil
import torch
import cv2
import ubelt as ub
import numpy as np
import pandas as pd
import imgaug as ia
import imgaug.augmenters as iaa
from clab.models.yolo2.utils import yolo_utils as yolo_utils
from clab.models.yolo2 import multiscale_batch_sampler
from clab.data import voc
from clab import hyperparams
from clab import xpu_device
from clab import fit_harness
from clab import monitor
from clab import nninit
from clab.lr_scheduler import ListedLR
from clab.models.yolo2 import darknet


class YoloVOCDataset(voc.VOCDataset):
    """
    Extends VOC localization dataset (which simply loads the images in VOC2008
    with minimal processing) for multiscale training.

    Example:
        >>> assert len(YoloVOCDataset(split='train')) == 2501
        >>> assert len(YoloVOCDataset(split='test')) == 4952
        >>> assert len(YoloVOCDataset(split='val')) == 2510

    Example:
        >>> self = YoloVOCDataset()
        >>> for i in range(10):
        ...     a, bc = self[i]
        ...     #print(bc[0].shape)
        ...     print(bc[1].shape)
        ...     print(a.shape)
    """

    def __init__(self, devkit_dpath=None, split='train'):
        super(YoloVOCDataset, self).__init__(devkit_dpath, split=split)

        """
        From YOLO9000.pdf:
            With the addition of anchor boxes we changed the resolution to
            416×416.

            Since our model downsamples by a factor of 32, we pull from the
            following multiples of 32: {320, 352, ..., 608}.
        """

        self.factor = 32  # downsample factor of yolo grid
        self.base_wh = np.array([416, 416], dtype=np.int)
        assert np.all(self.base_wh % self.factor == 0)

        self.multi_scale_inp_size = np.array([
            self.base_wh + (self.factor * i) for i in range(-3, 7)])
        self.multi_scale_out_size = self.multi_scale_inp_size // self.factor

        self.anchors = np.asarray([(1.08, 1.19), (3.42, 4.41),
                                   (6.63, 11.38), (9.42, 5.11),
                                   (16.62, 10.52)],
                                  dtype=np.float)
        self.num_anchors = len(self.anchors)
        self.augmenter = None

        if split == 'train':

            # From YOLO-V1 paper:
            #     For data augmentation we introduce random scaling and
            #     translations of up to 20% of the original image size. We
            #     also randomly adjust the exposure and saturation of the image
            #     by up to a factor of 1.5 in the HSV color space.

            # Not sure if this changed in V2 yet.

            augmentors = [
                iaa.Fliplr(p=.5),
                iaa.Flipud(p=.5),
                iaa.Affine(
                    scale={"x": (1.0, 1.01), "y": (1.0, 1.01)},
                    translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                    rotate=(-15, 15),
                    shear=(-7, 7),
                    order=[0, 1, 3],
                    cval=(0, 255),
                    mode=ia.ALL,  # use any of scikit-image's warping modes (see 2nd image from the top for examples)
                    # Note: currently requires imgaug master version
                    backend='cv2',
                ),
                iaa.AddToHueAndSaturation((-20, 20)),  # change hue and saturation
                iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5),  # improve or worsen the contrast
            ]
            self.augmenter = iaa.Sequential(augmentors)

    def _find_anchors(self):
        """

        Example:
            >>> self = YoloVOCDataset(split='trainval')
            >>> anchors = self._find_anchors()
            >>> print('anchors = {}'.format(ub.repr2(anchors, precision=2)))
            >>> # xdoctest: +REQUIRES(--show)
            >>> xy = -anchors / 2
            >>> wh = anchors
            >>> show_boxes = np.hstack([xy, wh])
            >>> from clab.util import mplutil
            >>> mplutil.figure(doclf=True, fnum=1)
            >>> mplutil.qtensure()  # xdoc: +SKIP
            >>> mplutil.draw_boxes(show_boxes, box_format='xywh')
            >>> from matplotlib import pyplot as plt
            >>> plt.gca().set_xlim(xy.min() - 1, wh.max() / 2 + 1)
            >>> plt.gca().set_ylim(xy.min() - 1, wh.max() / 2 + 1)
            >>> plt.gca().set_aspect('equal')
        """
        all_norm_wh = []
        from PIL import Image
        for i in ub.ProgIter(range(len(self))):
            annots = self._load_annotation(i)
            img_wh = np.array(Image.open(self.gpaths[i]).size)
            boxes = np.array(annots['boxes'])
            box_wh = boxes[:, 2:4] - boxes[:, 0:2]
            # normalize to 0-1
            norm_wh = box_wh / img_wh
            all_norm_wh.extend(norm_wh.tolist())
        # Re-normalize to the size of the grid
        all_wh = np.array(all_norm_wh) * self.base_wh[0] / self.factor
        from sklearn import cluster
        algo = cluster.KMeans(
            n_clusters=5, n_init=20, max_iter=10000, tol=1e-6,
            algorithm='elkan', verbose=0)
        algo.fit(all_wh)
        anchors = algo.cluster_centers_
        return anchors

    # def __len__(self):
    #     return 16

    @profiler.profile
    def __getitem__(self, index):
        """
        Example:
            >>> import sys
            >>> sys.path.append('/home/joncrall/code/clab/examples')
            >>> from yolo_voc import *
            >>> self = YoloVOCDataset(split='train')
            >>> chw01, label = self[1]
            >>> hwc01 = chw01.numpy().transpose(1, 2, 0)
            >>> print(hwc01.shape)
            >>> boxes, class_idxs = label[0:2]
            >>> # xdoc: +REQUIRES(--show)
            >>> from clab.util import mplutil
            >>> mplutil.figure(doclf=True, fnum=1)
            >>> mplutil.qtensure()  # xdoc: +SKIP
            >>> mplutil.imshow(hwc01, colorspace='rgb')
            >>> mplutil.draw_boxes(boxes.numpy(), box_format='tlbr')
        """
        if isinstance(index, tuple):
            # Get size index from the batch loader
            index, size_index = index
            inp_size = self.multi_scale_inp_size[size_index]
        else:
            inp_size = self.base_size

        # load the raw data from VOC
        image = self._load_image(index)
        annot = self._load_annotation(index)

        # VOC loads annotations in tlbr, but yolo expects xywh
        tlbr = annot['boxes'].astype(np.float32)
        gt_classes = annot['gt_classes']

        # Weight samples so we dont care about difficult cases
        gt_weights = 1.0 - annot['gt_ishard'].astype(np.float32)

        # squish the bounding box and image into a standard size
        w, h = inp_size
        im_shape = image.shape[0:2]
        sx = float(w) / im_shape[1]
        sy = float(h) / im_shape[0]
        tlbr[:, 0::2] *= sx
        tlbr[:, 1::2] *= sy
        interpolation = cv2.INTER_AREA if (sx + sy) <= 2 else cv2.INTER_CUBIC
        hwc255 = cv2.resize(image, (w, h), interpolation=interpolation)

        if self.augmenter:
            # Ensure the same augmentor is used for bboxes and iamges
            seq_det = self.augmenter.to_deterministic()

            hwc255 = seq_det.augment_image(hwc255)

            bbs = ia.BoundingBoxesOnImage(
                [ia.BoundingBox(x1, y1, x2, y2)
                 for x1, y1, x2, y2 in tlbr], shape=hwc255.shape)
            bbs = seq_det.augment_bounding_boxes([bbs])[0]

            tlbr = np.array([[bb.x1, bb.y1, bb.x2, bb.y2]
                              for bb in bbs.bounding_boxes])
            tlbr = yolo_utils.clip_boxes(tlbr, hwc255.shape[0:2])

        chw01 = torch.FloatTensor(hwc255.transpose(2, 0, 1) / 255)
        gt_classes = torch.LongTensor(gt_classes)

        # The original YOLO-v2 works in xywh, but this implementation seems to
        # work in tlbr.
        # xywh = yolo_utils.to_xywh(tlbr)
        # But does this implementation expect tlbr? I think it does.
        boxes = torch.FloatTensor(tlbr)

        # Return index information in the label as well
        orig_sz = torch.LongTensor(im_shape)
        index = torch.LongTensor([index])
        gt_weights = torch.FloatTensor(gt_weights)
        label = (boxes, gt_classes, orig_sz, index, gt_weights)
        return chw01, label

    @ub.memoize_method
    def _load_image(self, index):
        return super(YoloVOCDataset, self)._load_image(index)

    @ub.memoize_method
    def _load_annotation(self, index):
        return super(YoloVOCDataset, self)._load_annotation(index)


def make_loaders(datasets, train_batch_size=16, other_batch_size=1, workers=0):
    """
    Example:
        >>> datasets = {'train': YoloVOCDataset(split='train'),
        >>>             'vali': YoloVOCDataset(split='val')}
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
        batch_size = train_batch_size if key == 'train' else other_batch_size
        # use custom sampler that does multiscale training
        batch_sampler = multiscale_batch_sampler.MultiScaleBatchSampler(
            dset, batch_size=batch_size, shuffle=(key == 'train')
        )
        loader = dset.make_loader(batch_sampler=batch_sampler,
                                  num_workers=workers)
        loader.batch_size = batch_size
        loaders[key] = loader
    return loaders


def test():
    """
    import sys
    sys.path.append('/home/joncrall/code/clab/examples')
    from yolo_voc import *
    """
    devkit_dpath = ub.truepath('~/data/VOC/VOCdevkit')
    dset = YoloVOCDataset(devkit_dpath, split='test')
    loader = dset.make_loader(batch_size=8, num_workers=4)

    xpu = xpu_device.XPU.cast('argv')
    model = darknet.Darknet19(**{
        'num_classes': dset.num_classes,
        'anchors': dset.anchors
    })
    model = xpu.mount(model)

    weights_fpath = darknet.demo_weights()
    state_dict = torch.load(weights_fpath)['model_state_dict']
    model.module.load_state_dict(state_dict)

    num_images = len(dset)
    # gx = 212
    # cx = 0

    cacher = ub.Cacher('all_boxes', cfgstr='v1', enabled=False)
    data = cacher.tryload()
    if data is None:
        all_pred_boxes = [
            [np.empty([0, 5], dtype=np.float32)
             for _ in range(num_images)]
            for _ in range(dset.num_classes)]

        all_true_boxes = [
            [np.empty([0, 5], dtype=np.float32)
             for _ in range(num_images)]
            for _ in range(dset.num_classes)]

        for batch in ub.ProgIter(loader, freq=1, adjust=False):
            im_data, labels = batch
            im_data = xpu.move(im_data)
            outputs = model(im_data)

            gt_boxes, gt_classes = labels[0:2]
            gt_boxes = [b.numpy() for b in gt_boxes]
            gt_classes = [c.numpy() for c in gt_classes]

            gt_weights = labels[4]
            im_shapes, indices = labels[2:4]
            inp_size = im_data.shape[-2:]
            postout = model.module.postprocess(outputs, inp_size, im_shapes)

            out_boxes, out_scores, out_cxs = postout

            for gx, boxes, cxs, weights, orig_shape in zip(indices, gt_boxes, gt_classes, gt_weights, im_shapes):
                cx_to_idxs = ub.group_items(range(len(cxs)), cxs)
                for cx, idxs in cx_to_idxs.items():
                    # hack weights (ishard) into the dataset
                    sbox = np.hstack([boxes[idxs], weights[idxs][:, None]])
                    # NOTE; Unnormalize the true bboxes back to orig coords
                    sf = np.array(orig_shape) / np.array(inp_size)
                    sbox[:, 0:4:2] *= sf[1]
                    sbox[:, 1:4:2] *= sf[0]
                    all_true_boxes[cx][gx] = sbox

            for gx, boxes, scores, cxs in zip(indices, out_boxes, out_scores, out_cxs):
                cx_to_idxs = ub.group_items(range(len(cxs)), cxs)
                for cx, idxs in cx_to_idxs.items():
                    sbox = np.hstack([boxes[idxs], scores[idxs][:, None]])
                    all_pred_boxes[cx][gx] = sbox

        data = (all_true_boxes, all_pred_boxes)
        cacher.save(data)
    all_true_boxes, all_pred_boxes = data

    # Test our scoring implementation
    voceval = voc.EvaluateVOC(all_true_boxes, all_pred_boxes)
    mean_ap, ap_list = voceval.compute()
    print('mean_ap = {!r}'.format(mean_ap))

    if False:
        # HACK IN THE ORIGINAL SCORING CODE
        import pickle
        import os
        output_dir = ub.ensuredir('test')
        det_file = os.path.join(output_dir, 'detections.pkl')
        with open(det_file, 'wb') as f:
            pickle.dump(all_pred_boxes, f, pickle.HIGHEST_PROTOCOL)

        print('Evaluating detections')
        import sys
        sys.path.append(ub.truepath('~/code/yolo2-pytorch'))
        from datasets import pascal_voc
        link = ub.symlink('/home/joncrall/data/VOC/VOCdevkit/',
                          '/home/joncrall/data/VOC/VOCdevkit2007/', verbose=3)
        imdb = pascal_voc.VOCDataset('voc_2007_test', os.path.dirname(link), 1, None)
        imdb.evaluate_detections(all_pred_boxes, output_dir)


def setup_harness(workers=None):
    """
    CommandLine:
        python ~/code/clab/examples/yolo_voc.py setup_harness
        python ~/code/clab/examples/yolo_voc.py setup_harness --profile

    Example:
        >>> harn = setup_harness(workers=0)
        >>> harn.initialize()
        >>> harn.dry = True
        >>> harn.run()
    """
    workdir = ub.truepath('~/work/VOC2007')
    devkit_dpath = ub.truepath('~/data/VOC/VOCdevkit')
    nice = ub.argval('--nice', default=None)

    pretrained_fpath = darknet.initial_weights()

    postproc_params = dict(
        conf_thresh=0.001,
        nms_thresh=0.45,
        ovthresh=0.5,
    )

    max_epoch = 160

    lr_step_points = {
        0: 0.001,
        # warmup learning rate
        # 0:  0.0001,
        # 1:  0.0001,
        # 2:  0.0002,
        # 3:  0.0003,
        # 4:  0.0004,
        # 5:  0.0005,
        # 6:  0.0006,
        # 7:  0.0007,
        # 8:  0.0008,
        # 9:  0.0009,
        # 10: 0.0010,
        # cooldown learning rate
        # 30: 0.0005,
        60: 0.0001,
        90: 0.00001,
    }
    batch_size = int(ub.argval('--batch_size', default=16))
    n_cpus = psutil.cpu_count(logical=True)
    workers = int(ub.argval('--workers', default=int(n_cpus / 2)))
    other_batch_size = batch_size // 4

    datasets = {
        # 'train': YoloVOCDataset(devkit_dpath, split='train'),
        # 'vali': YoloVOCDataset(devkit_dpath, split='val'),
        'test': YoloVOCDataset(devkit_dpath, split='test'),
        'train': YoloVOCDataset(devkit_dpath, split='trainval'),
    }

    # Check that the dataset exists
    # item = datasets['train'][0]
    # print('item = {!r}'.format(item))

    loaders = make_loaders(datasets,
                           train_batch_size=batch_size,
                           other_batch_size=other_batch_size,
                           workers=workers if workers is not None else workers)

    """
    Reference:
        Original YOLO9000 hyperparameters are defined here:
        https://github.com/pjreddie/darknet/blob/master/cfg/yolo-voc.2.0.cfg

        https://github.com/longcw/yolo2-pytorch/issues/1#issuecomment-286410772

        Notes:
            jitter is a translation / crop parameter
            https://groups.google.com/forum/#!topic/darknet/A-JJeXprvJU

            thresh in 2.0.cfg is iou_thresh here
    """

    hyper = hyperparams.HyperParams(

        model=(darknet.Darknet19, {
            'num_classes': datasets['train'].num_classes,
            'anchors': datasets['train'].anchors
        }),

        criterion=(darknet.DarknetLoss, {
            'anchors': datasets['train'].anchors,
            'object_scale': 5.0,
            'noobject_scale': 1.0,
            'class_scale': 1.0,
            'coord_scale': 1.0,
            'iou_thresh': 0.6,
        }),

        optimizer=(torch.optim.SGD, dict(
            lr=lr_step_points[0],
            momentum=0.9,
            weight_decay=0.0005
        )),

        # initializer=(nninit.KaimingNormal, {}),
        initializer=(nninit.Pretrained, {
            'fpath': pretrained_fpath,
        }),

        scheduler=(ListedLR, dict(
            step_points=lr_step_points
        )),

        other=ub.dict_union({
            'nice': str(nice),
            'batch_size': loaders['train'].batch_sampler.batch_size,
        }, postproc_params),
        centering=None,

        # centering=datasets['train'].centering,
        augment=datasets['train'].augmenter,
    )

    # NOTE: XPU implicitly supports DataParallel just pass --gpu=0,1,2,3
    xpu = xpu_device.XPU.cast('argv')
    print('xpu = {!r}'.format(xpu))
    harn = fit_harness.FitHarness(
        hyper=hyper, xpu=xpu, loaders=loaders, max_iter=max_epoch,
        workdir=workdir,
    )
    harn.postproc_params = postproc_params
    harn.nice = nice
    harn.monitor = monitor.Monitor(min_keys=['loss'],
                                   # max_keys=['global_acc', 'class_acc'],
                                   patience=max_epoch)

    @harn.set_batch_runner
    def batch_runner(harn, inputs, labels):
        """
        Custom function to compute the output of a batch and its loss.

        Example:
            >>> import sys
            >>> sys.path.append('/home/joncrall/code/clab/examples')
            >>> from yolo_voc import *
            >>> harn = setup_harness(workers=0)
            >>> harn.initialize()
            >>> batch = harn._demo_batch(0, 'train')
            >>> inputs, labels = batch
            >>> criterion = harn.criterion
            >>> weights_fpath = darknet.demo_weights()
            >>> state_dict = torch.load(weights_fpath)['model_state_dict']
            >>> harn.model.module.load_state_dict(state_dict)
            >>> outputs, loss = harn._custom_run_batch(harn, inputs, labels)
        """
        outputs = harn.model(*inputs)

        # darknet criterion needs to know the input image shape
        inp_size = tuple(inputs[0].shape[-2:])

        bbox_pred, iou_pred, prob_pred = outputs
        gt_boxes, gt_classes, orig_size, indices, gt_weights = labels

        loss = harn.criterion(bbox_pred, iou_pred, prob_pred, gt_boxes,
                              gt_classes, gt_weights=gt_weights,
                              inp_size=inp_size)
        return outputs, loss

    @harn.add_batch_metric_hook
    def custom_metrics(harn, output, labels):
        metrics_dict = ub.odict()
        criterion = harn.criterion
        metrics_dict['L_bbox'] = float(criterion.bbox_loss.data.cpu().numpy())
        metrics_dict['L_iou'] = float(criterion.iou_loss.data.cpu().numpy())
        metrics_dict['L_cls'] = float(criterion.cls_loss.data.cpu().numpy())
        return metrics_dict

    # Set as a harness attribute instead of using a closure
    harn.batch_confusions = []

    @harn.add_iter_callback
    def on_batch(harn, tag, loader, bx, inputs, labels, outputs, loss):
        """
        Custom hook to run on each batch (used to compute mAP on the fly)

        Example:
            >>> harn = setup_harness(workers=0)
            >>> harn.initialize()
            >>> batch = harn._demo_batch(0, 'train')
            >>> inputs, labels = batch
            >>> criterion = harn.criterion
            >>> loader = harn.loaders['train']
            >>> weights_fpath = darknet.demo_weights()
            >>> state_dict = torch.load(weights_fpath)['model_state_dict']
            >>> harn.model.module.load_state_dict(state_dict)
            >>> outputs, loss = harn._custom_run_batch(harn, inputs, labels)
            >>> tag = 'train'
            >>> on_batch(harn, tag, loader, bx, inputs, labels, outputs, loss)
        """
        # Accumulate relevant outputs to measure
        gt_boxes, gt_classes, orig_size, indices, gt_weights = labels
        # bbox_pred, iou_pred, prob_pred = outputs
        im_sizes = orig_size
        inp_size = inputs[0].shape[-2:]

        conf_thresh = harn.postproc_params['conf_thresh']
        nms_thresh = harn.postproc_params['nms_thresh']
        ovthresh = harn.postproc_params['ovthresh']

        postout = harn.model.module.postprocess(outputs, inp_size, im_sizes,
                                                conf_thresh, nms_thresh)
        # batch_pred_boxes, batch_pred_scores, batch_pred_cls_inds = postout
        # Compute: y_pred, y_true, and y_score for this batch
        batch_pred_boxes, batch_pred_scores, batch_pred_cls_inds = postout
        batch_true_boxes, batch_true_cls_inds = labels[0:2]
        batch_orig_sz, batch_img_inds = labels[2:4]

        y_batch = []
        for bx, index in enumerate(batch_img_inds.data.cpu().numpy().ravel()):
            pred_boxes  = batch_pred_boxes[bx]
            pred_scores = batch_pred_scores[bx]
            pred_cxs    = batch_pred_cls_inds[bx]

            # Group groundtruth boxes by class
            true_boxes_ = batch_true_boxes[bx].data.cpu().numpy()
            true_cxs = batch_true_cls_inds[bx].data.cpu().numpy()
            true_weights = gt_weights[bx].data.cpu().numpy()

            # NOTE; Unnormalize the true bboxes back to orig coords
            orig_shape = batch_orig_sz[bx]
            sf = np.array(orig_shape) / np.array(inp_size)
            if len(true_boxes_):
                true_boxes = np.hstack([true_boxes_, true_weights[:, None]])
                true_boxes[:, 0:4:2] *= sf[1]
                true_boxes[:, 1:4:2] *= sf[0]

            y = voc.EvaluateVOC.image_confusions(true_boxes, true_cxs,
                                                 pred_boxes, pred_scores,
                                                 pred_cxs, ovthresh=ovthresh)
            y['gx'] = index
            y_batch.append(y)

        harn.batch_confusions.extend(y_batch)

    @harn.add_epoch_callback
    def on_epoch(harn, tag, loader):
        y = pd.concat(harn.batch_confusions)
        num_classes = len(loader.dataset.label_names)

        mean_ap, ap_list = voc.EvaluateVOC.compute_map(y, num_classes)

        harn.log_value(tag + ' epoch mAP', mean_ap, harn.epoch)
        # max_ap = np.nanmax(ap_list)
        # harn.log_value(tag + ' epoch max-AP', max_ap, harn.epoch)
        harn.batch_confusions.clear()

    return harn


def train():
    """
    python ~/code/clab/examples/yolo_voc.py train --nice=baseline --workers=0
    python ~/code/clab/examples/yolo_voc.py train --nice=trainval2

    python ~/code/clab/examples/yolo_voc.py train --nice=med_batch --workers=6 --gpu=0,1 --batch_size=32
    python ~/code/clab/examples/yolo_voc.py train --nice=big_batch --workers=6 --gpu=0,1,2,3 --batch_size=64

    python ~/code/clab/examples/yolo_voc.py train --nice=basic --workers=0 --gpu=0 --batch_size=16
    python ~/code/clab/examples/yolo_voc.py train --nice=basic --workers=0 --batch_size=16
    """
    harn = setup_harness()
    with harn.xpu:
        harn.run()


if __name__ == '__main__':
    r"""
    CommandLine:
        export PYTHONPATH=$PYTHONPATH:/home/joncrall/code/clab/examples
        python ~/code/clab/examples/yolo_voc.py
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
