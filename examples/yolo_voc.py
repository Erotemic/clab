"""
Need to compile yolo first.

Currently setup is a bit hacked.

pip install cffi
python setup.py build_ext --inplace
"""
from os.path import exists
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
        >>> self = YoloVOC()
        >>> for i in range(10):
        ...     a, bc = self[i]
        ...     #print(bc[0].shape)
        ...     print(bc[1].shape)
        ...     print(a.shape)
    """

    def __init__(self, devkit_dpath=None, split='train'):
        super(YoloVOCDataset, self).__init__(devkit_dpath, split=split)
        base_wh = np.array([320, 320], dtype=np.int)
        self.multi_scale_inp_size = [base_wh + (32 * i) for i in range(8)]
        self.multi_scale_out_size = [s / 32 for s in self.multi_scale_inp_size]

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
            >>> boxes, class_idxs = label
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

        # load the raw data
        # hwc255, boxes, gt_classes = self._load_item(index, inp_size)
        image = self._load_image(index)
        annot = self._load_annotation(index)

        boxes = annot['boxes'].astype(np.float32)
        gt_classes = annot['gt_classes']

        # squish the bounding box and image into a standard size
        w, h = inp_size
        im_shape = image.shape[0:2]
        sx = float(w) / im_shape[1]
        sy = float(h) / im_shape[0]
        boxes[:, 0::2] *= sx
        boxes[:, 1::2] *= sy
        interpolation = cv2.INTER_AREA if (sx + sy) <= 2 else cv2.INTER_CUBIC
        hwc255 = cv2.resize(image, (w, h), interpolation=interpolation)

        if self.augmenter:
            # Ensure the same augmentor is used for bboxes and iamges
            seq_det = self.augmenter.to_deterministic()

            hwc255 = seq_det.augment_image(hwc255)

            bbs = ia.BoundingBoxesOnImage(
                [ia.BoundingBox(x1, y1, x2, y2)
                 for x1, y1, x2, y2 in boxes], shape=hwc255.shape)
            bbs = seq_det.augment_bounding_boxes([bbs])[0]

            boxes = np.array([[bb.x1, bb.y1, bb.x2, bb.y2]
                              for bb in bbs.bounding_boxes])
            boxes = yolo_utils.clip_boxes(boxes, hwc255.shape[0:2])

        chw01 = torch.FloatTensor(hwc255.transpose(2, 0, 1) / 255)
        gt_classes = torch.LongTensor(gt_classes)
        boxes = torch.LongTensor(boxes.astype(np.int32))

        # Return index information in the label as well
        label = (boxes, gt_classes, torch.LongTensor(im_shape),
                                    torch.LongTensor([index]))
        return chw01, label

    @ub.memoize_method
    def _load_image(self, index):
        return super(YoloVOCDataset, self)._load_image(index)

    @ub.memoize_method
    def _load_annotation(self, index):
        return super(YoloVOCDataset, self)._load_annotation(index)


def make_loaders(datasets, train_batch_size=16, vali_batch_size=1, workers=0):
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
        batch_size = train_batch_size if key == 'train' else vali_batch_size
        # use custom sampler that does multiscale training
        batch_sampler = multiscale_batch_sampler.MultiScaleBatchSampler(
            dset, batch_size=batch_size, shuffle=(key == 'train')
        )
        loader = dset.make_loader(batch_sampler=batch_sampler,
                                  num_workers=workers)
        loader.batch_size = batch_size
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
    n_cpus = psutil.cpu_count(logical=True)

    workers = int(n_cpus / 2)

    max_epoch = 160

    weight_decay = 0.0005
    momentum = 0.9
    init_learning_rate = 1e-3

    # dataset
    vali_batch_size = 4
    train_batch_size = 16

    # lr_decay = 1. / 10
    # lr_step_points = {
    #     0: init_learning_rate * lr_decay ** 0,
    #     60: init_learning_rate * lr_decay ** 1,
    #     90: init_learning_rate * lr_decay ** 2,
    # }
    lr_step_points = {
        # warmup learning rate
        0:  0.0001,
        10: 0.0010,
        # cooldown learning rate
        30: 0.0005,
        60: 0.0001,
        90: 0.00001,
    }

    workdir = ub.truepath('~/work/VOC2007')
    devkit_dpath = ub.truepath('~/data/VOC/VOCdevkit')


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
    cfg.pretrained_fpath = grab_darknet19_initial_weights()
    datasets = {
        'train': YoloVOCDataset(cfg.devkit_dpath, split='train'),
        'vali': YoloVOCDataset(cfg.devkit_dpath, split='val'),
        # 'train': YoloVOCDataset(cfg.devkit_dpath, split='trainval'),
        'test': YoloVOCDataset(cfg.devkit_dpath, split='test'),
    }

    loaders = make_loaders(datasets,
                           train_batch_size=cfg.train_batch_size,
                           vali_batch_size=cfg.vali_batch_size,
                           workers=workers if workers is not None else cfg.workers)

    """
    Reference:
        Original YOLO9000 hyperparameters are defined here:
        https://github.com/pjreddie/darknet/blob/master/cfg/yolo-voc.2.0.cfg
    """

    hyper = hyperparams.HyperParams(

        model=(darknet.Darknet19, {
            'num_classes': datasets['train'].num_classes,
            'anchors': datasets['train'].anchors
        }),

        # https://github.com/longcw/yolo2-pytorch/issues/1#issuecomment-286410772
        criterion=(darknet.DarknetLoss, {
            'anchors': datasets['train'].anchors,
            'object_scale': 5.0,
            'noobject_scale': 1.0,
            'class_scale': 1.0,
            'coord_scale': 1.0,
            'iou_thresh': 0.5,
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
            'batch_size': loaders['train'].batch_sampler.batch_size,
        },
        centering=None,

        # centering=datasets['train'].centering,
        augment=datasets['train'].augmenter,
    )

    xpu = xpu_device.XPU.cast('auto')
    harn = fit_harness.FitHarness(
        hyper=hyper, xpu=xpu, loaders=loaders, max_iter=160,
    )
    harn.monitor = monitor.Monitor(min_keys=['loss'],
                                   # max_keys=['global_acc', 'class_acc'],
                                   patience=160)

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

            >>> loader = harn.loaders['train']
            >>> weights_fpath = darknet.demo_weights()
            >>> state_dict = torch.load(weights_fpath)['model_state_dict']
            >>> harn.model.module.load_state_dict(state_dict)
            >>> outputs, loss = harn._custom_run_batch(harn, inputs, labels)
        """
        outputs = harn.model(*inputs)

        # darknet criterion needs to know the input image shape
        inp_size = tuple(inputs[0].shape[-2:])
        # assert np.sqrt(outputs[1].shape[1]) == inp_size[0] / 32

        bbox_pred, iou_pred, prob_pred = outputs
        gt_boxes, gt_classes, orig_size, indices = labels
        dontcare = np.array([[]] * len(gt_boxes))

        loss = harn.criterion(bbox_pred, iou_pred, prob_pred, gt_boxes,
                              gt_classes, dontcare=dontcare, inp_size=inp_size)
        return outputs, loss

    # Set as a harness attribute instead of using a closure
    harn.batch_confusions = []

    @harn.add_iter_callback
    def on_batch(harn, tag, loader, bx, inputs, labels, outputs, loss):
        # Accumulate relevant outputs to measure
        # if tag == 'train':
        #     return
        gt_boxes, gt_classes, orig_size, indices = labels
        # bbox_pred, iou_pred, prob_pred = outputs
        im_sizes = orig_size
        inp_size = inputs[0].shape[-2:]
        conf_thresh = 0.24
        nms_thresh = 0.5
        ovthresh = 0.5

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
            true_boxes = batch_true_boxes[bx].data.cpu().numpy()
            true_cxs = batch_true_cls_inds[bx].data.cpu().numpy()

            y = voc.EvaluateVOC.image_confusions(true_boxes, true_cxs,
                                                 pred_boxes, pred_scores,
                                                 pred_cxs, ovthresh=ovthresh)
            y['gx'] = index
            y_batch.append(y)

        harn.batch_confusions.extend(y_batch)

        # harn.accumulated2.append((y_pred_, y_true_, y_score_))
        # harn.accumulated.append((postout, labels))

    @harn.add_epoch_callback
    def on_epoch(harn, tag, loader):
        y = pd.concat(harn.batch_confusions)
        num_classes = len(loader.dataset.label_names)

        mean_ap, ap_list = voc.EvaluateVOC.compute_map(y, num_classes)

        harn.log_value(tag + ' epoch mAP', mean_ap, harn.epoch)
        harn.batch_confusions.clear()

    return harn


def train():
    """
    python ~/code/clab/examples/yolo_voc.py train
    """
    harn = setup_harness()
    harn.setup_dpath(ub.ensuredir(cfg.workdir))
    harn.run()


if __name__ == '__main__':
    r"""
    CommandLine:
        export PYTHONPATH=$PYTHONPATH:/home/joncrall/code/clab/examples
        python ~/code/clab/examples/yolo_voc.py
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
