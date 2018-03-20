"""
References:
    https://pjreddie.com/media/files/papers/yolo.pdf
    https://pjreddie.com/media/files/papers/YOLO9000.pdf

    https://github.com/ruiminshen/yolo2-pytorch/blob/master/model/yolo2.py
    https://github.com/starsmall-xiaoqq/yolo2keras/blob/master/backend.py

    https://github.com/pjreddie/darknet/blob/master/cfg/yolo-voc.2.0.cfg
    https://www.slideshare.net/JinwonLee9/pr12-yolo9000  # See Slide 16-18
"""
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from .layers.reorg.reorg_layer import ReorgLayer
from .utils.cython_bbox import bbox_ious, anchor_intersections
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


class TrackCudaLoss(torch.nn.modules.loss._Loss):
    """
    Keep track of if the module is in cpu or gpu mode
    """
    def __init__(self):
        super(TrackCudaLoss, self).__init__()
        self._iscuda = False
        self._device_num = None

    def cuda(self, device_num=None, **kwargs):
        self._iscuda = True
        self._device_num = device_num
        return super(TrackCudaLoss, self).cuda(device_num, **kwargs)

    def cpu(self):
        self._iscuda = False
        self._device_num = None
        return super(TrackCudaLoss, self).cpu()

    @property
    def is_cuda(self):
        return self._iscuda

    def get_device(self):
        return self._device_num


class DarknetLoss(TrackCudaLoss):
    """
    Example:
        >>> from clab.models.yolo2.darknet import *
        >>> model = Darknet19(num_classes=20)
        >>> criterion = DarknetLoss(model.anchors)
        >>> im_data = torch.randn(1, 3, 100, 100)
        >>> output = model(im_data)
        >>> bbox_pred, iou_pred, prob_pred = output
        >>> gt_boxes = np.array([[[0, 0, 10, 10]]])
        >>> gt_classes = np.array([[1]])
        >>> dontcare = np.array([[]])
        >>> inp_size = (100, 100)
        >>> loss = criterion(bbox_pred, iou_pred, prob_pred, gt_boxes, gt_classes, dontcare, inp_size)
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

    def forward(criterion, bbox_pred, iou_pred, prob_pred, gt_boxes=None,
                gt_classes=None, dontcare=None, inp_size=None):
        bbox_pred_np = bbox_pred.data.cpu().numpy()
        iou_pred_np = iou_pred.data.cpu().numpy()

        num_classes = prob_pred.shape[-1]

        _tup = criterion._build_target(bbox_pred_np, gt_boxes, gt_classes,
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

        criterion.bbox_loss = criterion.mse(bbox_pred * box_mask,
                                            _boxes * box_mask) / num_boxes
        criterion.iou_loss = criterion.mse(iou_pred * iou_mask,
                                           _ious * iou_mask) / num_boxes

        class_mask = class_mask.expand_as(prob_pred)
        criterion.cls_loss = criterion.mse(prob_pred * class_mask,
                                           _classes * class_mask) / num_boxes

        total_loss = criterion.bbox_loss + criterion.iou_loss + criterion.cls_loss
        return total_loss

    def _build_target(criterion, bbox_pred_np, gt_boxes, gt_classes, dontcare,
                      iou_pred_np, inp_size, num_classes, anchors):
        """
        :param bbox_pred: shape: (B, h x w, num_anchors, 4) :
                          (sig(tx), sig(ty), exp(tw), exp(th))
        """

        B = bbox_pred_np.shape[0]
        losskw = dict(object_scale=criterion.object_scale,
                      noobject_scale=criterion.noobject_scale,
                      class_scale=criterion.class_scale,
                      coord_scale=criterion.coord_scale,
                      iou_thresh=criterion.iou_thresh)

        func = partial(_process_batch, inp_size=inp_size,
                       num_classes=num_classes, anchors=anchors, **losskw)

        args = ((bbox_pred_np[b], gt_boxes[b], gt_classes[b], dontcare[b],
                 iou_pred_np[b])
                for b in range(B))
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


def _process_batch(data, inp_size, num_classes, anchors, object_scale=5.0,
                   noobject_scale=1.0, class_scale=1.0, coord_scale=1.0,
                   iou_thresh=0.5):
    """
    Example:
        >>> bbox_pred_np = np.random.randn(9, 5, 4)
        >>> gt_boxes = np.random.randn(1, 4)
        >>> gt_classes = (np.random.randn(1,) * 20).astype(np.int)
        >>> dontcares = np.empty((0,))
        >>> iou_pred_np = np.random.randn(9, 5, 1)
        >>> data = (bbox_pred_np, gt_boxes, gt_classes, dontcares, iou_pred_np)
        >>> inp_size = (96, 96)
        >>> num_classes = 20
        >>> anchors = np.random.randn(5, 2)
        >>> _process_batch(data, inp_size, num_classes, anchors)
    """
    bbox_pred_np, gt_boxes, gt_classes, dontcares, iou_pred_np = data

    out_size = [s // 32 for s in inp_size]
    W, H = out_size

    # net output
    hw, num_anchors, _ = bbox_pred_np.shape

    # gt
    _classes = np.zeros([hw, num_anchors, num_classes], dtype=np.float)
    _class_mask = np.zeros([hw, num_anchors, 1], dtype=np.float)

    _ious = np.zeros([hw, num_anchors, 1], dtype=np.float)
    _iou_mask = np.zeros([hw, num_anchors, 1], dtype=np.float)

    _boxes = np.zeros([hw, num_anchors, 4], dtype=np.float)
    _boxes[:, :, 0:2] = 0.5
    _boxes[:, :, 2:4] = 1.0
    _box_mask = np.zeros([hw, num_anchors, 1], dtype=np.float) + 0.01

    # scale pred_bbox
    anchors = np.ascontiguousarray(anchors, dtype=np.float)
    bbox_pred_np_ = np.expand_dims(bbox_pred_np, 0)
    bbox_pred = np.ascontiguousarray(bbox_pred_np_, dtype=np.float)

    bbox_np = yolo_utils.yolo_to_bbox(bbox_pred, anchors, H, W)
    # bbox_np = (hw, num_anchors, (x1, y1, x2, y2))   range: 0 ~ 1
    bbox_np = bbox_np[0]
    bbox_np[:, :, 0::2] *= float(inp_size[0])  # rescale x
    bbox_np[:, :, 1::2] *= float(inp_size[1])  # rescale y

    # gt_boxes_b = np.asarray(gt_boxes[b], dtype=np.float)
    gt_boxes_b = np.asarray(gt_boxes, dtype=np.float)

    # for each cell, compare predicted_bbox and gt_bbox
    bbox_np_b = np.reshape(bbox_np, [-1, 4])
    ious = bbox_ious(
        np.ascontiguousarray(bbox_np_b, dtype=np.float),
        np.ascontiguousarray(gt_boxes_b, dtype=np.float)
    )
    best_ious = np.max(ious, axis=1).reshape(_iou_mask.shape)
    iou_penalty = 0 - iou_pred_np[best_ious < iou_thresh]
    _iou_mask[best_ious <= iou_thresh] = noobject_scale * iou_penalty

    # locate the cell of each gt_boxe
    cell_w = float(inp_size[0]) / W
    cell_h = float(inp_size[1]) / H
    cx = (gt_boxes_b[:, 0] + gt_boxes_b[:, 2]) * 0.5 / cell_w
    cy = (gt_boxes_b[:, 1] + gt_boxes_b[:, 3]) * 0.5 / cell_h
    cell_inds = np.floor(cy) * W + np.floor(cx)
    cell_inds = cell_inds.astype(np.int)

    target_boxes = np.empty(gt_boxes_b.shape, dtype=np.float)
    target_boxes[:, 0] = cx - np.floor(cx)  # cx
    target_boxes[:, 1] = cy - np.floor(cy)  # cy
    target_boxes[:, 2] = ((gt_boxes_b[:, 2] - gt_boxes_b[:, 0]) /
                          inp_size[0] * out_size[0])  # tw
    target_boxes[:, 3] = ((gt_boxes_b[:, 3] - gt_boxes_b[:, 1]) /
                          inp_size[1] * out_size[1])  # th

    # for each gt boxes, match the best anchor
    gt_boxes_resize = np.copy(gt_boxes_b)
    gt_boxes_resize[:, 0::2] *= (out_size[0] / float(inp_size[0]))
    gt_boxes_resize[:, 1::2] *= (out_size[1] / float(inp_size[1]))
    gt_boxes_resize = np.ascontiguousarray(gt_boxes_resize, dtype=np.float)

    anchor_ious = anchor_intersections(anchors, gt_boxes_resize)
    anchor_inds = np.argmax(anchor_ious, axis=0)

    ious_reshaped = np.reshape(ious, [hw, num_anchors, len(cell_inds)])
    for i, cell_ind in enumerate(cell_inds):
        if cell_ind >= hw or cell_ind < 0:
            print('cell inds size {}'.format(len(cell_inds)))
            print('cell over {} hw {}'.format(cell_ind, hw))
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

        # net_cfgs = [
        #     # conv1s
        #     [(32, 3)],
        #     ['M', (64, 3)],
        #     ['M', (128, 3), (64, 1), (128, 3)],
        #     ['M', (256, 3), (128, 1), (256, 3)],
        #     ['M', (512, 3), (256, 1), (512, 3), (256, 1), (512, 3)],
        #     # conv2
        #     ['M', (1024, 3), (512, 1), (1024, 3), (512, 1), (1024, 3)],
        #     # ------------
        #     # conv3
        #     [(1024, 3), (1024, 3)],
        #     # conv4
        #     [(1024, 3)]
        # ]
        # # darknet
        # self.conv1s, c1 = _make_layers(3, net_cfgs[0:5])
        # self.conv2, c2 = _make_layers(c1, net_cfgs[5])
        # # ---
        # self.conv3, c3 = _make_layers(c2, net_cfgs[6])

        # See Slide 16
        # https://www.slideshare.net/JinwonLee9/pr12-yolo9000

        """
        Reduced form of original yolo-voc.2.0.cfg

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
        final = conv5

        # for detection
        # (B, C, H, W) -> (B, H, W, C) -> (B, H * W, A, 5 + C)
        B, _, h, w = final.size()
        final_reshaped = final.permute(0, 2, 3, 1).contiguous().view(
            B, -1, self.num_anchors, self.num_classes + 5)

        # Construct relative offsets for every anchor box
        # tx, ty, tw, th, to -> sig(tx), sig(ty), exp(tw), exp(th), sig(to)
        xy_pred    = F.sigmoid(final_reshaped[:, :, :, 0:2])
        wh_pred    = torch.exp(final_reshaped[:, :, :, 2:4])
        bbox_pred  = torch.cat([xy_pred, wh_pred], 3)

        # Predict IOU
        iou_pred   = F.sigmoid(final_reshaped[:, :, :, 4:5])

        # TODO: do we do heirarchy stuff here?
        score_pred = final_reshaped[:, :, :, 5:].contiguous()
        score_energy = score_pred.view(-1, score_pred.size()[-1])
        prob_pred = F.softmax(score_energy, dim=1).view_as(score_pred)

        # Note: bbox_pred are relative offsets to the anchors.
        output = (bbox_pred, iou_pred, prob_pred)
        return output

    def postprocess(self, output, inp_size, im_shapes, conf_thresh=0.24,
                    nms_thresh=0.5):
        """
        Postprocess the raw network output into usable bounding boxes

        Args:
            bbox_pred: (B, HxW, A, 4)
                        ndarray of float (sig(tx), sig(ty), exp(tw), exp(th))

            iou_pred:  (B, HxW, A, 1)

            prob_pred: (B, HxW, A, C)

            inp_size (tuple): size of input to network

            im_shapes (list): size of each in image in the batch before rescale

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
            >>> output = self(im_data)
            >>> # Define remaining params
            >>> conf_thresh = 0.24
            >>> nms_thresh = 0.5
            >>> im_shapes = [rgb255.shape[0:2]]
            >>> postout = self.postprocess(output, inp_size, im_shapes, conf_thresh, nms_thresh)
            >>> bbox_abs, scores, cls_inds = postout
            >>> # xdoc: +REQUIRES(--show)
            >>> from clab.util import mplutil
            >>> mplutil.qtensure()  # xdoc: +SKIP
            >>> mplutil.figure(fnum=1, doclf=True)
            >>> mplutil.imshow(rgb255, colorspace='rgb')
            >>> mplutil.draw_boxes(bbox_abs)
        """
        bbox_pred_, iou_pred_, prob_pred_ = output

        assert bbox_pred_.shape[0] == 1, (
            'postprocess only support one image per batch')  # noqa

        # num_classes, num_anchors = cfg.num_classes, cfg.num_anchors
        num_classes = self.num_classes
        anchors = self.anchors
        out_size = np.array(inp_size) // 32  # hacked
        W, H = out_size

        bbox_pred = bbox_pred_.data.cpu().numpy()
        iou_pred  = iou_pred_.data.cpu().numpy()
        prob_pred = prob_pred_.data.cpu().numpy()
        bbox_pred = np.ascontiguousarray(bbox_pred, dtype=np.float)

        # Convert anchored predictions to absolute bounding boxes
        bbox_abs = yolo_utils.yolo_to_bbox(bbox_pred, anchors, H, W)

        # Scale the bounding boxes to the size of the original image.
        # and convert to integer representation.
        im_shape = im_shapes[0]
        bbox_abs[..., 0::2] *= float(im_shape[1])
        bbox_abs[..., 1::2] *= float(im_shape[0])
        bbox_abs = bbox_abs.astype(np.int)

        # converts (1, G, A, 4) to (G * A, 4), where G is the number of grid
        # cells and A is the number of anchor boxes.
        bbox_abs = np.reshape(bbox_abs, [-1, 4])
        iou_pred = np.reshape(iou_pred, [-1])
        prob_pred = np.reshape(prob_pred, [-1, num_classes])

        cls_inds = np.argmax(prob_pred, axis=1)
        prob_pred = prob_pred[(np.arange(prob_pred.shape[0]), cls_inds)]

        """
        Reference: arXiv:1506.02640 [cs.CV] (Yolo 1):
            Formally we define confidence as $Pr(Object) ∗ IOU^truth_pred$.
            If no object exists in that cell, the confidence scores should be
            zero. Otherwise we want the confidence score to equal the
            intersection over union (IOU) between the predicted box and the
            ground truth
        """
        scores = iou_pred * prob_pred

        # filter boxes based on confidence threshold
        keep = np.where(scores >= conf_thresh)
        bbox_abs = bbox_abs[keep]
        scores = scores[keep]
        cls_inds = cls_inds[keep]

        # nonmax supression (pre-class)
        keep = np.zeros(len(bbox_abs), dtype=np.int)
        for i in range(num_classes):
            inds = np.where(cls_inds == i)[0]
            if len(inds) == 0:
                continue
            c_bboxes = bbox_abs[inds]
            c_scores = scores[inds]
            c_keep = yolo_utils.nms_detections(c_bboxes, c_scores, nms_thresh)
            keep[inds[c_keep]] = 1

        keep = np.where(keep > 0)
        # keep = nms_detections(bbox_abs, scores, 0.3)
        bbox_abs = bbox_abs[keep]
        scores = scores[keep]
        cls_inds = cls_inds[keep]

        # clip
        bbox_abs = yolo_utils.clip_boxes(bbox_abs, im_shape)

        postout = bbox_abs, scores, cls_inds
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
