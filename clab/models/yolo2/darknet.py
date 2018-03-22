"""
References:
    https://pjreddie.com/media/files/papers/yolo.pdf
    https://pjreddie.com/media/files/papers/YOLO9000.pdf

    https://github.com/ruiminshen/yolo2-pytorch/blob/master/model/yolo2.py
    https://github.com/starsmall-xiaoqq/yolo2keras/blob/master/backend.py

    https://github.com/pjreddie/darknet/blob/master/cfg/yolo-voc.2.0.cfg
    https://www.slideshare.net/JinwonLee9/pr12-yolo9000  # See Slide 16-18
"""
import ubelt as ub
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from .layers.reorg.reorg_layer import ReorgLayer
from .utils import yolo_utils
from functools import partial
from multiprocessing import Pool


class Conv2d_Noli(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 relu=True, same_padding=False):
        super(Conv2d_Noli, self).__init__()
        padding = int((kernel_size - 1) / 2) if same_padding else 0
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride, padding=padding)
        self.relu = nn.LeakyReLU(0.1, inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Conv2d_Norm_Noli(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 relu=True, same_padding=False):
        super(Conv2d_Norm_Noli, self).__init__()
        padding = int((kernel_size - 1) / 2) if same_padding else 0

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, momentum=0.01)
        self.relu = nn.LeakyReLU(0.1, inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


def np_to_variable(x, is_cuda=True, dtype=torch.FloatTensor):
    v = torch.autograd.Variable(torch.from_numpy(x).type(dtype))
    if is_cuda:
        v = v.cuda()
    return v


class BaseLossWithCudaState(torch.nn.modules.loss._Loss):
    """
    Keep track of if the module is in cpu or gpu mode
    """
    def __init__(self):
        super(BaseLossWithCudaState, self).__init__()
        self._iscuda = False
        self._device_num = None

    def cuda(self, device_num=None, **kwargs):
        self._iscuda = True
        self._device_num = device_num
        return super(BaseLossWithCudaState, self).cuda(device_num, **kwargs)

    def cpu(self):
        self._iscuda = False
        self._device_num = None
        return super(BaseLossWithCudaState, self).cpu()

    @property
    def is_cuda(self):
        return self._iscuda

    def get_device(self):
        return self._device_num


class DarknetLoss(BaseLossWithCudaState):
    """
    Example:
        >>> from clab.models.yolo2.darknet import *
        >>> model = Darknet19(num_classes=20)
        >>> criterion = DarknetLoss(model.anchors)
        >>> B = 1
        >>> inp_size = (100, 100)
        >>> im_data = torch.randn(B, 3, *inp_size)
        >>> output = model(im_data)
        >>> aoff_pred, iou_pred, prob_pred = output
        >>> rng = np.random.RandomState(0)
        >>> gt_classes = [yolo_utils.random_boxes(rng.randint(0, 4)) for _ in range(B)]
        >>> dontcare = [np.array([]) for _ in range(B)]
        >>> loss = criterion(aoff_pred, iou_pred, prob_pred, gt_boxes, gt_classes, dontcare, inp_size)
    """
    def __init__(criterion, anchors, object_scale=5.0, noobject_scale=1.0,
                 class_scale=1.0, coord_scale=1.0, iou_thresh=0.5,
                 workers=None):
        # train
        super(DarknetLoss, criterion).__init__()
        criterion.bbox_loss = None
        criterion.iou_loss = None
        criterion.cls_loss = None

        criterion.object_scale = object_scale
        criterion.noobject_scale = noobject_scale
        criterion.class_scale = class_scale
        criterion.coord_scale = coord_scale
        criterion.iou_thresh = iou_thresh

        if workers is None:
            criterion.pool = None
        else:
            criterion.pool = Pool(processes=workers)

        criterion.anchors = np.ascontiguousarray(anchors, dtype=np.float)
        criterion.mse = nn.MSELoss(size_average=False)

    def forward(criterion, aoff_pred, iou_pred, prob_pred, gt_boxes=None,
                gt_classes=None, dontcare=None, inp_size=None):
        aoff_pred_np = aoff_pred.data.cpu().numpy()
        iou_pred_np = iou_pred.data.cpu().numpy()

        num_classes = prob_pred.shape[-1]

        _tup = criterion._build_target(aoff_pred_np, gt_boxes, gt_classes,
                                       dontcare, iou_pred_np, inp_size,
                                       num_classes, criterion.anchors)
        _boxes, _ious, _classes, _box_mask, _iou_mask, _class_mask = _tup

        is_cuda = criterion.is_cuda

        _boxes = np_to_variable(_boxes, is_cuda)
        _ious = np_to_variable(_ious, is_cuda)
        _classes = np_to_variable(_classes, is_cuda)
        box_mask = np_to_variable(_box_mask, is_cuda, dtype=torch.FloatTensor)
        iou_mask = np_to_variable(_iou_mask, is_cuda, dtype=torch.FloatTensor)
        class_mask = np_to_variable(_class_mask, is_cuda,
                                    dtype=torch.FloatTensor)

        num_boxes = sum((len(boxes) for boxes in gt_boxes))

        # _boxes[:, :, :, 2:4] = torch.log(_boxes[:, :, :, 2:4])
        box_mask = box_mask.expand_as(_boxes)

        criterion.bbox_loss = criterion.mse(aoff_pred * box_mask,
                                            _boxes * box_mask) / num_boxes
        criterion.iou_loss = criterion.mse(iou_pred * iou_mask,
                                           _ious * iou_mask) / num_boxes

        class_mask = class_mask.expand_as(prob_pred)
        criterion.cls_loss = criterion.mse(prob_pred * class_mask,
                                           _classes * class_mask) / num_boxes

        total_loss = criterion.bbox_loss + criterion.iou_loss + criterion.cls_loss
        return total_loss

    def _build_target(criterion, aoff_pred_np, gt_boxes, gt_classes, dontcare,
                      iou_pred_np, inp_size, num_classes, anchors):
        """
        Determine which ground truths to compare against which predictions?

        Args:
            aoff_pred_np: [B, H x W, A, 4]:
                (sig(tx), sig(ty), exp(tw), exp(th))

        Example:
            >>> from clab.models.yolo2.darknet import *
            >>> model = Darknet19(num_classes=20)
            >>> criterion = DarknetLoss(model.anchors)
            >>> B = 4
            >>> inp_size = (100, 100)
            >>> im_data = torch.randn(B, 3, *inp_size)
            >>> output = model(im_data)
            >>> aoff_pred, iou_pred, prob_pred = output
            >>> gt_boxes = np.array([[[0, 0, 10, 10]]])
            >>> gt_classes = (torch.rand(B, 1) * 20).long()
            >>> dontcare = np.array([[]])
            >>> loss = criterion(aoff_pred, iou_pred, prob_pred, gt_boxes, gt_classes, dontcare, inp_size)
        """
        losskw = dict(object_scale=criterion.object_scale,
                      noobject_scale=criterion.noobject_scale,
                      class_scale=criterion.class_scale,
                      coord_scale=criterion.coord_scale,
                      iou_thresh=criterion.iou_thresh)

        func = partial(_process_batch, inp_size=inp_size,
                       num_classes=num_classes, anchors=anchors, **losskw)

        args = zip(aoff_pred_np, iou_pred_np, gt_boxes, gt_classes, dontcare)
        args = list(args)

        if criterion.pool:
            targets = criterion.pool.map(func, args)
        else:
            targets = list(map(func, args))

        _boxes = np.stack(tuple((row[0] for row in targets)))
        _ious = np.stack(tuple((row[1] for row in targets)))
        _classes = np.stack(tuple((row[2] for row in targets)))
        _box_mask = np.stack(tuple((row[3] for row in targets)))
        _iou_mask = np.stack(tuple((row[4] for row in targets)))
        _class_mask = np.stack(tuple((row[5] for row in targets)))

        return _boxes, _ious, _classes, _box_mask, _iou_mask, _class_mask


def _process_batch(data, inp_size, num_classes, anchors,
                   object_scale=5.0,
                   noobject_scale=1.0,
                   class_scale=1.0,
                   coord_scale=1.0,
                   iou_thresh=0.5):
    """
    Assign predicted boxes to groundtruth boxes and compute their IOU for the
    loss calculation.

    Example:
        >>> ntrue = 3  # number of gt boxes in one image
        >>> A, H, W = 5, 3, 3
        >>> inp_size = (96, 96)
        >>> num_classes = 20
        >>> gt_boxes = yolo_utils.random_boxes(ntrue).numpy()
        >>> gt_classes = (np.random.rand(ntrue,) * 20).astype(np.int)
        >>> weights = np.ones(ntrue)
        >>> aoff_pred_np = np.random.randn(H * W, A, 4)
        >>> iou_pred_np = np.random.rand(H * W, A, 1)
        >>> anchors = np.abs(np.random.randn(A, 2))
        >>> data = (aoff_pred_np, iou_pred_np, gt_boxes, gt_classes, weights)
        >>> object_scale = 5.0
        >>> noobject_scale = 1.0
        >>> class_scale = 1.0
        >>> coord_scale = 1.0
        >>> iou_thresh = 0.5
        >>> _tup = _process_batch(data, inp_size, num_classes, anchors)
        >>> _boxes, _ious, _classes, _box_mask, _iou_mask, _class_mask = _tup
    """
    # TODO: change dontcare to sample weights?
    aoff_pred_np, iou_pred_np, gt_boxes, gt_classes, weights = data

    out_size = [s // 32 for s in inp_size]
    W, H = out_size

    # net output
    hw, num_anchors, _ = aoff_pred_np.shape

    # PREALLOCATE OUTPUT
    # -----------------
    # gt
    _classes = np.zeros([hw, num_anchors, num_classes], dtype=np.float)
    _class_mask = np.zeros([hw, num_anchors, 1], dtype=np.float)

    _ious = np.zeros([hw, num_anchors, 1], dtype=np.float)
    _iou_mask = np.zeros([hw, num_anchors, 1], dtype=np.float)

    _boxes = np.zeros([hw, num_anchors, 4], dtype=np.float)
    _boxes[:, :, 0:2] = 0.5
    _boxes[:, :, 2:4] = 1.0
    _box_mask = np.zeros([hw, num_anchors, 1], dtype=np.float) + 0.01

    # APPLY OFFSETS TO EACH ANCHOR BOX
    # --------------------------------
    # get prediected boxes centered at each grid cell.
    anchors = np.ascontiguousarray(anchors, dtype=np.float)
    aoff_pred_np_ = np.expand_dims(aoff_pred_np, 0)
    aoff_pred_np_ = np.ascontiguousarray(aoff_pred_np_, dtype=np.float)
    bbox_abs_pred = yolo_utils.yolo_to_bbox(aoff_pred_np_, anchors, H, W)
    # bbox_abs_pred = [hw, num_anchors, [x1, y1, x2, y2]]   range: 0 ~ 1
    bbox_abs_pred = bbox_abs_pred[0]
    bbox_abs_pred[:, :, 0::2] *= float(inp_size[0])  # rescale x
    bbox_abs_pred[:, :, 1::2] *= float(inp_size[1])  # rescale y

    # FIND IOU BETWEEN ALL PAIRS OF (PRED x TRUE) BOXES
    # -------------------------------------------------
    # for each cell, compare predicted_bbox and gt_bbox
    gt_boxes_b = np.asarray(gt_boxes, dtype=np.float)
    pred_boxes_b = np.reshape(bbox_abs_pred, [-1, 4])
    ious = yolo_utils.bbox_ious(
        np.ascontiguousarray(pred_boxes_b, dtype=np.float),
        np.ascontiguousarray(gt_boxes_b, dtype=np.float)
    )
    # determine which iou is best
    best_ious = np.max(ious, axis=1).reshape(_iou_mask.shape)
    iou_penalty = 0 - iou_pred_np[best_ious < iou_thresh]
    _iou_mask[best_ious <= iou_thresh] = noobject_scale * iou_penalty

    # ASSIGN EACH TRUE BOX TO A GRID CELL
    # -----------------------------------
    # locate the cell of each gt_box
    # determine which cell each ground truth box belongs to
    cell_w = float(inp_size[0]) / W
    cell_h = float(inp_size[1]) / H
    cx = (gt_boxes_b[:, 0] + gt_boxes_b[:, 2]) * 0.5 / cell_w
    cy = (gt_boxes_b[:, 1] + gt_boxes_b[:, 3]) * 0.5 / cell_h
    gt_cell_inds = np.floor(cy) * W + np.floor(cx)
    gt_cell_inds = gt_cell_inds.astype(np.int)

    target_boxes = np.empty(gt_boxes_b.shape, dtype=np.float)
    target_boxes[:, 0] = cx - np.floor(cx)  # cx
    target_boxes[:, 1] = cy - np.floor(cy)  # cy
    target_boxes[:, 2] = ((gt_boxes_b[:, 2] - gt_boxes_b[:, 0]) /
                          inp_size[0] * out_size[0])  # tw
    target_boxes[:, 3] = ((gt_boxes_b[:, 3] - gt_boxes_b[:, 1]) /
                          inp_size[1] * out_size[1])  # th

    # ASSIGN EACH TRUE BOX TO AN ANCHOR
    # -----------------------------------
    # for each gt boxes, match the best anchor
    gt_boxes_resize = np.copy(gt_boxes_b)
    gt_boxes_resize[:, 0::2] *= (out_size[0] / float(inp_size[0]))
    gt_boxes_resize[:, 1::2] *= (out_size[1] / float(inp_size[1]))
    gt_boxes_resize = np.ascontiguousarray(gt_boxes_resize, dtype=np.float)

    anchor_ious = yolo_utils.anchor_intersections(anchors, gt_boxes_resize)
    anchor_inds = np.argmax(anchor_ious, axis=0)

    ious_reshaped = np.reshape(ious, [hw, num_anchors, len(gt_cell_inds)])
    for i, cell_ind in enumerate(gt_cell_inds):
        if cell_ind >= hw or cell_ind < 0:
            # print('cell inds size {}'.format(len(gt_cell_inds)))
            # print('cell over {} hw {}'.format(cell_ind, hw))
            continue
        a = anchor_inds[i]

        # 0 ~ 1, should be close to 1
        iou_pred_cell_anchor = iou_pred_np[cell_ind, a, :]
        _iou_mask[cell_ind, a, :] = object_scale * (1 - iou_pred_cell_anchor)  # noqa
        # _ious[cell_ind, a, :] = anchor_ious[a, i]
        _ious[cell_ind, a, :] = ious_reshaped[cell_ind, a, i]

        _box_mask[cell_ind, a, :] = coord_scale
        target_boxes[i, 2:4] /= anchors[a]
        _boxes[cell_ind, a, :] = target_boxes[i]

        _class_mask[cell_ind, a, :] = class_scale
        _classes[cell_ind, a, gt_classes[i]] = 1.

    # _boxes[:, :, 2:4] = np.maximum(_boxes[:, :, 2:4], 0.001)
    # _boxes[:, :, 2:4] = np.log(_boxes[:, :, 2:4])
    return _boxes, _ious, _classes, _box_mask, _iou_mask, _class_mask


class Darknet19(nn.Module):
    """
    Example:
        >>> from clab.models.yolo2.darknet import *
        >>> self = Darknet19(num_classes=20)
        >>> im_data = torch.randn(1, 3, 221, 221)
        >>> output = self(im_data)
        >>> bbox_pred, iou_pred, prob_pred = output
    """
    def __init__(self, num_classes=None, anchors=None):
        super(Darknet19, self).__init__()

        if anchors is None:
            # https://github.com/pjreddie/darknet/blob/master/cfg/yolo-voc.2.0.cfg#L228
            anchors = np.array([(1.08, 1.19), (3.42, 4.41), (6.63, 11.38),
                                (9.42, 5.11), (16.62, 10.52)], dtype=np.float)

        self.anchors = anchors
        self.num_classes = num_classes
        self.num_anchors = len(self.anchors)

        """
        Reduced form of original yolo-voc.2.0.cfg:

            # NOTE: conv1s
            [convolutional] filters=32 size=3

            [maxpool] size=2 stride=2
            [convolutional] filters=64 size=3

            [maxpool] size=2 stride=2
            [convolutional] filters=128 size=3
            [convolutional] filters=64 size=1
            [convolutional] filters=128 size=3

            [maxpool] size=2 stride=2
            [convolutional] filters=256 size=3
            [convolutional] filters=128 size=1
            [convolutional] filters=256 size=3

            [maxpool] size=2 stride=2
            [convolutional] filters=512 size=3
            [convolutional] filters=256 size=1
            [convolutional] filters=512 size=3
            [convolutional] filters=256 size=1
            [convolutional] filters=512 size=3   # -9 [route1] refers here

            # NOTE: conv2
            [maxpool] size=2 stride=2            # -8
            [convolutional] filters=1024 size=3  # -7
            [convolutional] filters=512 size=1   # -6
            [convolutional] filters=1024 size=3  # -5
            [convolutional] filters=512 size=1   # -4
            [convolutional] filters=1024 size=3  # -3

            #######

            # NOTE: conv3
            [convolutional] size=3 filters=1024  # -2
            [convolutional] size=3 filters=1024  # -1 / -3 [route2] refers here

            # NOTE: reorg (connects to output of conv1s)
            [route] layers=-9  # -2  # [route1]
            [reorg] stride=2   # -1

            # NOTE: skip connections (concat layer)
            [route] layers=-1,-3     # [route2]

            # NOTE: conv4
            [convolutional] size=3 filters=1024

            # NOTE: conv5
            [convolutional] size=1 filters=125 activation=linear
        """

        def _make_layers(in_channels, net_cfg):
            layers = []

            if len(net_cfg) > 0 and isinstance(net_cfg[0], list):
                for sub_cfg in net_cfg:
                    layer, in_channels = _make_layers(in_channels, sub_cfg)
                    layers.append(layer)
            else:
                for item in net_cfg:
                    if item == 'M':
                        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
                    else:
                        out_channels, ksize = item
                        layers.append(Conv2d_Norm_Noli(in_channels,
                                                       out_channels, ksize,
                                                       same_padding=True))
                        in_channels = out_channels
            return nn.Sequential(*layers), in_channels

        # darknet
        self.conv1s, c1 = _make_layers(3, [
            [(32, 3)],
            ['M', (64, 3)],
            ['M', (128, 3), (64, 1), (128, 3)],
            ['M', (256, 3), (128, 1), (256, 3)],
            ['M', (512, 3), (256, 1), (512, 3), (256, 1), (512, 3)],
        ])
        self.conv2, c2 = _make_layers(c1, [
            'M', (1024, 3), (512, 1), (1024, 3), (512, 1), (1024, 3)])
        # ---
        self.conv3, c3 = _make_layers(c2, [(1024, 3), (1024, 3)])

        self.reorg = ReorgLayer(in_channels=c1, stride=2)
        reorg_n_feat_out = self.reorg.out_channels

        # cat [conv1s, conv3]
        self.conv4, c4 = _make_layers((reorg_n_feat_out + c3), [(1024, 3)])

        # linear
        out_channels = self.num_anchors * (self.num_classes + 5)
        self.conv5 = Conv2d_Noli(c4, out_channels, 1, 1, relu=False)

    def forward(self, im_data):
        # Let B = batch_size
        # Let A = num_anchors
        # Let C = num_classes
        conv1s = self.conv1s(im_data)
        conv2 = self.conv2(conv1s)
        conv3 = self.conv3(conv2)

        # Reorganize conv1s to facilitate a skip connection
        conv1s_reorg = self.reorg(conv1s)
        cat_1_3 = torch.cat([conv1s_reorg, conv3], 1)  # [B,        3072, H, W]

        # Final 3x3 followed by a 1x1 convolution to obtain raw output
        conv4 = self.conv4(cat_1_3)                    # [B,        1024, H, W]
        conv5 = self.conv5(conv4)                      # [B, (C * A + 5), H, W]

        # for detection
        # (B, C, H, W) -> (B, H, W, C) -> (B, H * W, A, 5 + C)
        B, _, h, w = conv5.size()
        final = conv5.permute(0, 2, 3, 1).contiguous().view(
            B, -1, self.num_anchors, self.num_classes + 5)

        # Construct relative offsets for every anchor box
        # tx, ty, tw, th, to -> sig(tx), sig(ty), exp(tw), exp(th), sig(to)
        xy_offset_pred = F.sigmoid(final[:, :, :, 0:2])
        wh_offset_pred = torch.exp(final[:, :, :, 2:4])
        # Predict: Anchor Offsets (relative offsets to the anchors).
        aoff_pred = torch.cat([xy_offset_pred, wh_offset_pred], 3)

        # Predict IOU
        iou_pred   = F.sigmoid(final[:, :, :, 4:5])

        # TODO: do we do heirarchy stuff here?
        score_pred = final[:, :, :, 5:].contiguous()
        score_energy = score_pred.view(-1, score_pred.size()[-1])
        prob_pred = F.softmax(score_energy, dim=1).view_as(score_pred)

        output = (aoff_pred, iou_pred, prob_pred)
        return output

    def postprocess(self, output, inp_size, im_shapes, conf_thresh=0.24,
                    nms_thresh=0.5):
        """
        Postprocess the raw network output into usable bounding boxes

        Args:
            aoff_pred (ndarray): [B, HxW, A, 4]
                anchor offsets in the format (sig(x), sig(y), exp(w), exp(h))

            iou_pred (ndarray): [B, HxW, A, 1]
                predicted iou (is this the objectness score?)

            prob_pred (ndarray): [B, HxW, A, C]
                predicted class probability

            inp_size (tuple): size of input to network

            im_shapes (list): [B, 2]
                size of each in image before rescale

            conf_thresh (float): threshold for filtering bboxes. Keep only the
                detections above this confidence value.

            nms_thresh (float): nonmax supression iou threshold

        Notes:
            Let B = batch_size
            Let A = num_anchors
            Let C = num_classes
            Let (H, W) = shape of the output grid

            Original params for nms_thresh (iou_thresh) and conf_thresh
            (thresh) are here:
                https://github.com/pjreddie/darknet/blob/master/examples/yolo.c#L213

            On parameter settings:
                Remove the bounding boxes which have no object. Remove the
                bounding boxes that predict a confidence score less than a
                threshold of 0.24

                https://towardsdatascience.com/training-object-detection-yolov2-from-scratch-using-cyclic-learning-rates-b3364f7e4755

            Network Visualization:
                http://ethereon.github.io/netscope/#/gist/d08a41711e48cf111e330827b1279c31

        CommandLine:
            python -m clab.models.yolo2.darknet Darknet19.postprocess

        Example:
            >>> from clab.models.yolo2.darknet import *
            >>> inp_size = (288, 288)
            >>> self = Darknet19(num_classes=20)
            >>> state_dict = torch.load(demo_weights())['model_state_dict']
            >>> self.load_state_dict(state_dict)
            >>> im_data, rgb255 = demo_image(inp_size)
            >>> im_data = torch.cat([im_data, im_data])  # make a batch size of 2
            >>> output = self(im_data)
            >>> # Define remaining params
            >>> im_shapes = [rgb255.shape[0:2]] * len(im_data)
            >>> conf_thresh = 0.01
            >>> nms_thresh = 0.5
            >>> postout = self.postprocess(output, inp_size, im_shapes, conf_thresh, nms_thresh)
            >>> out_boxes, out_scores, out_cxs = postout
            >>> # xdoc: +REQUIRES(--show)
            >>> from clab.util import mplutil
            >>> mplutil.qtensure()  # xdoc: +SKIP
            >>> label_names = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
            >>>  'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
            >>>  'dog', 'horse', 'motorbike', 'person',
            >>>  'pottedplant', 'sheep', 'sofa', 'train',
            >>>  'tvmonitor')
            >>> import pandas as pd
            >>> cls_names = list(ub.take(label_names, out_cxs[0]))
            >>> print(pd.DataFrame({'name': cls_names, 'score': out_scores[0]}))
            >>> mplutil.figure(fnum=1, doclf=True)
            >>> mplutil.imshow(rgb255, colorspace='rgb')
            >>> mplutil.draw_boxes(out_boxes[0])
        """
        aoff_pred_, iou_pred_, prob_pred_ = output
        aoff_pred = aoff_pred_.data.cpu().numpy()
        iou_pred  = iou_pred_.data.cpu().numpy()
        prob_pred = prob_pred_.data.cpu().numpy()

        # num_classes, num_anchors = cfg.num_classes, cfg.num_anchors
        num_classes = self.num_classes
        anchors = self.anchors
        out_size = np.array(inp_size) // 32  # hacked
        W, H = out_size

        out_boxes = []
        out_scores = []
        out_cxs = []

        # For each image in the batch, postprocess the predicted boxes
        for bx in range(aoff_pred.shape[0]):
            aoffs = aoff_pred[bx]
            ious  = iou_pred[bx]
            probs = prob_pred[bx]
            im_shape = im_shapes[bx]

            # Convert anchored predictions to absolute bounding boxes
            aoffs = np.ascontiguousarray(aoffs, dtype=np.float)
            boxes = yolo_utils.yolo_to_bbox(aoffs[None, :], anchors, H, W)[0]
            # Scale the bounding boxes to the size of the original image.
            # and convert to integer representation.
            boxes[..., 0::2] *= float(im_shape[1])
            boxes[..., 1::2] *= float(im_shape[0])
            boxes = boxes.astype(np.int)

            # converts [1, W * H, A, 4] -> [W * H * A, 4]
            boxes = np.reshape(boxes, [-1, 4])
            ious = np.reshape(ious, [-1])
            probs = np.reshape(probs, [-1, num_classes])

            # Predict the class with maximum probability
            cls_inds = np.argmax(probs, axis=1)
            cls_probs = probs[(np.arange(probs.shape[0]), cls_inds)]

            """
            Reference: arXiv:1506.02640 [cs.CV] (Yolo 1):
                Formally we define confidence as $Pr(Object) ∗ IOU^truth_pred$.
                If no object exists in that cell, the confidence scores should
                be zero. Otherwise we want the confidence score to equal the
                intersection over union (IOU) between the predicted box and the
                ground truth
            """
            # Compute the final probabilities for the predicted class
            scores = ious * cls_probs

            # filter boxes based on confidence threshold
            keep_conf = np.where(scores >= conf_thresh)
            boxes = boxes[keep_conf]
            scores = scores[keep_conf]
            cls_inds = cls_inds[keep_conf]

            # nonmax supression (per-class)
            keep_flags = np.zeros(len(boxes), dtype=np.uint8)
            cx_to_inds = ub.group_items(range(len(cls_inds)), cls_inds)
            cx_to_inds = ub.map_vals(np.array, cx_to_inds)
            for cx, inds in cx_to_inds.items():
                # Do get predictions for each class
                c_bboxes = boxes[inds]
                c_scores = scores[inds]
                c_keep = yolo_utils.nms_detections(c_bboxes, c_scores,
                                                   nms_thresh)
                keep_flags[inds[c_keep]] = 1

            keep_nms = np.where(keep_flags > 0)
            boxes = boxes[keep_nms]
            scores = scores[keep_nms]
            cls_inds = cls_inds[keep_nms]

            # clip
            boxes = yolo_utils.clip_boxes(boxes, im_shape)

            out_boxes.append(boxes)
            out_scores.append(scores)
            out_cxs.append(cls_inds)

        postout = (out_boxes, out_scores, out_cxs)
        return postout

    def load_from_npz(self, fname, num_conv=None):
        dest_src = {'conv.weight': 'kernel', 'conv.bias': 'biases',
                    'bn.weight': 'gamma', 'bn.bias': 'biases',
                    'bn.running_mean': 'moving_mean',
                    'bn.running_var': 'moving_variance'}
        params = np.load(fname)
        own_dict = self.state_dict()
        keys = list(own_dict.keys())

        for i, start in enumerate(range(0, len(keys), 5)):
            if num_conv is not None and i >= num_conv:
                break
            end = min(start + 5, len(keys))
            for key in keys[start:end]:
                list_key = key.split('.')
                ptype = dest_src['{}.{}'.format(list_key[-2], list_key[-1])]
                src_key = '{}-convolutional/{}:0'.format(i, ptype)
                print((src_key, own_dict[key].size(), params[src_key].shape))
                param = torch.from_numpy(params[src_key])
                if ptype == 'kernel':
                    param = param.permute(3, 2, 0, 1)
                own_dict[key].copy_(param)

    def load_from_h5py(self, fname):
        """

        Ignore:
            # convert the h5 to a pt model and host it
            >>> fname = '/home/joncrall/Downloads/yolo-voc.weights.h5'
            >>> self = Darknet19(num_classes=20)
            >>> self.load_from_h5py(fname)
            >>> # from clab import xpu_device
            >>> # xpu = xpu_device.XPU()
            >>> # self = xpu.mount(self)
            >>> data = {
            ...     'model_state_dict': self.state_dict(),
            ...     'num_classes': self.num_classes,
            ... }
            >>> torch.save(data, 'yolo-voc.weights.pt')

        """
        import h5py
        h5f = h5py.File(fname, mode='r')
        for k, v in list(self.state_dict().items()):
            param = torch.from_numpy(np.asarray(h5f[k]))
            v.copy_(param)


def demo_weights():
    import ubelt as ub
    import os
    url = 'https://data.kitware.com/api/v1/item/5ab13b0e8d777f068578e251/download'
    dpath = ub.ensure_app_cache_dir('clab')
    fname = 'yolo-voc.weights.pt'
    dest = os.path.join(dpath, fname)
    if not os.path.exists(dest):
        command = 'curl -X GET {} > {}'.format(url, dest)
        ub.cmd(command, verbout=1, shell=True)
    return dest


def demo_image(inp_size):
    from clab import util
    import cv2
    rgb255 = util.grab_test_image('astro', 'rgb')
    rgb01 = cv2.resize(rgb255, inp_size).astype(np.float32) / 255
    im_data = torch.FloatTensor([rgb01.transpose(2, 0, 1)])
    return im_data, rgb255
