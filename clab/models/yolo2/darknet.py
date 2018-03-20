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


def _process_batch(data, inp_size, num_classes, anchors, object_scale=5.0,
                   noobject_scale=1.0, class_scale=1.0, coord_scale=1.0,
                   iou_thresh=0.6):
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
    anchor_ious = anchor_intersections(
        anchors,
        np.ascontiguousarray(gt_boxes_resize, dtype=np.float)
    )
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


class TrackCudaLoss(torch.nn.modules.loss._Loss):
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

        criterion.anchors = anchors

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
        :param bbox_pred: shape: (bsize, h x w, num_anchors, 4) :
                          (sig(tx), sig(ty), exp(tw), exp(th))
        """

        bsize = bbox_pred_np.shape[0]
        losskw = dict(object_scale=criterion.object_scale,
                      noobject_scale=criterion.noobject_scale,
                      class_scale=criterion.class_scale,
                      coord_scale=criterion.coord_scale,
                      iou_thresh=criterion.iou_thresh)

        func = partial(_process_batch, inp_size=inp_size,
                       num_classes=num_classes, anchors=anchors, **losskw)

        args = ((bbox_pred_np[b], gt_boxes[b], gt_classes[b], dontcare[b],
                 iou_pred_np[b])
                for b in range(bsize))
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
            anchors = np.array([(1.08, 1.19), (3.42, 4.41), (6.63, 11.38),
                                (9.42, 5.11), (16.62, 10.52)], dtype=np.float)

        net_cfgs = [
            # conv1s
            [(32, 3)],
            ['M', (64, 3)],
            ['M', (128, 3), (64, 1), (128, 3)],
            ['M', (256, 3), (128, 1), (256, 3)],
            ['M', (512, 3), (256, 1), (512, 3), (256, 1), (512, 3)],
            # conv2
            ['M', (1024, 3), (512, 1), (1024, 3), (512, 1), (1024, 3)],
            # ------------
            # conv3
            [(1024, 3), (1024, 3)],
            # conv4
            [(1024, 3)]
        ]

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
                        # layers.append(Conv2d_Noli(in_channels, out_channels,
                        #     ksize, same_padding=True))
                        in_channels = out_channels

            return nn.Sequential(*layers), in_channels

        # darknet
        self.conv1s, c1 = _make_layers(3, net_cfgs[0:5])
        self.conv2, c2 = _make_layers(c1, net_cfgs[5])
        # ---
        self.conv3, c3 = _make_layers(c2, net_cfgs[6])

        stride = 2
        # stride*stride times the channels of conv1s
        self.reorg = ReorgLayer(stride=2)
        # cat [conv1s, conv3]
        self.conv4, c4 = _make_layers(
            (c1 * (stride * stride) + c3), net_cfgs[7])

        self.anchors = anchors
        self.num_classes = num_classes
        self.num_anchors = len(self.anchors)

        # linear
        out_channels = self.num_anchors * (self.num_classes + 5)
        self.conv5 = Conv2d_Noli(c4, out_channels, 1, 1, relu=False)
        self.global_average_pool = nn.AvgPool2d((1, 1))

    def forward(self, im_data):
        conv1s = self.conv1s(im_data)
        conv2 = self.conv2(conv1s)
        conv3 = self.conv3(conv2)
        conv1s_reorg = self.reorg(conv1s)
        cat_1_3 = torch.cat([conv1s_reorg, conv3], 1)
        conv4 = self.conv4(cat_1_3)
        conv5 = self.conv5(conv4)   # batch_size, out_channels, h, w
        gapooled = self.global_average_pool(conv5)

        # for detection
        # bsize, c, h, w -> bsize, h, w, c ->
        #                   bsize, h x w, num_anchors, 5+num_classes
        bsize, _, h, w = gapooled.size()
        # assert bsize == 1, 'detection only support one image per batch'
        gapooled_reshaped = gapooled.permute(0, 2, 3, 1).contiguous().view(
            bsize, -1, self.num_anchors, self.num_classes + 5)

        # tx, ty, tw, th, to -> sig(tx), sig(ty), exp(tw), exp(th), sig(to)
        xy_pred = F.sigmoid(gapooled_reshaped[:, :, :, 0:2])
        wh_pred = torch.exp(gapooled_reshaped[:, :, :, 2:4])
        bbox_pred = torch.cat([xy_pred, wh_pred], 3)
        iou_pred = F.sigmoid(gapooled_reshaped[:, :, :, 4:5])

        score_pred = gapooled_reshaped[:, :, :, 5:].contiguous()
        # TODO: do we do heirarchy stuff here?
        score_energy = score_pred.view(-1, score_pred.size()[-1])
        prob_pred = F.softmax(score_energy, dim=1).view_as(score_pred)

        return bbox_pred, iou_pred, prob_pred

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
