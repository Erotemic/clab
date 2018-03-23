import numpy as np
import itertools as it
import torch
# import cv2
# from .im_transform import imcv2_affine_trans, imcv2_recolor
# from box import BoundBox, box_iou, prob_compare
from .nms_wrapper import nms
from .cython_yolo import yolo_to_bbox as yolo_to_bbox_c
from .cython_bbox import bbox_ious as bbox_ious_c
from .cython_bbox import anchor_intersections as anchor_intersections_c


def anchor_intersections(anchors, query_boxes):
    """
    For each query box compute the intersection ratio covered by anchors
    This assumes query boxes are normalized and centerated at the anchors.

    Args:
        anchors: (A, 2) aspect ratios of anchor boxes
        query_boxes: (K, 4) normalized query boxes centered at anchor posisions
            in tlbr format.

    Returns:
        overlaps: (A, K) ndarray of intersec between boxes and query_boxes

    Example:
        >>> A = 5
        >>> query_boxes = random_boxes(10, 'tlbr', 1.0).numpy().astype(np.float)
        >>> anchors = np.abs(np.random.randn(A, 2)).astype(np.float)
        >>> intersec1 = anchor_intersections_py(anchors, query_boxes)
        >>> intersec2 = anchor_intersections(anchors, query_boxes)
        >>> assert np.all(intersec1 == intersec2)
    """
    return anchor_intersections_c(anchors, query_boxes)


def anchor_intersections_py(anchors, query_boxes):
    N = anchors.shape[0]
    K = query_boxes.shape[0]
    intersec = np.zeros((N, K), dtype=np.float)
    for n in range(N):
        anchor_area = anchors[n, 0] * anchors[n, 1]
        for k in range(K):
            boxw = (query_boxes[k, 2] - query_boxes[k, 0] + 1)
            boxh = (query_boxes[k, 3] - query_boxes[k, 1] + 1)
            iw = min(anchors[n, 0], boxw)
            ih = min(anchors[n, 1], boxh)
            inter_area = iw * ih
            denom = (anchor_area + boxw * boxh - inter_area)
            intersec[n, k] = inter_area / denom
    return intersec


def bbox_ious(boxes, query_boxes):
    """
    For each query box compute the IOU covered by boxes

    Args:
        boxes: float[K, 4] in tlbr format
        query_boxes: float[K, 4] in tlbr format

    Returns:
        overlaps: (N, K) ndarray of intersec between boxes and query_boxes

    Example:
        >>> query_boxes = random_boxes(10, 'tlbr', 1.0).numpy().astype(np.float)
        >>> boxes = random_boxes(10, 'tlbr', 1.0).numpy().astype(np.float)
        >>> overlaps1 = bbox_ious_py(boxes, query_boxes)
        >>> overlaps2 = bbox_ious(boxes, query_boxes)
        >>> assert np.all(overlaps1 == overlaps2)
    """
    return bbox_ious_c(boxes, query_boxes)


def bbox_ious_py(boxes, query_boxes):
    N = boxes.shape[0]
    K = query_boxes.shape[0]
    intersec = np.zeros((N, K), dtype=np.float)
    for k in range(K):
        qbox_area = (
            (query_boxes[k, 2] - query_boxes[k, 0] + 1) *
            (query_boxes[k, 3] - query_boxes[k, 1] + 1)
        )
        for n in range(N):
            iw = (
                min(boxes[n, 2], query_boxes[k, 2]) -
                max(boxes[n, 0], query_boxes[k, 0]) + 1
            )
            if iw > 0:
                ih = (
                    min(boxes[n, 3], query_boxes[k, 3]) -
                    max(boxes[n, 1], query_boxes[k, 1]) + 1
                )
                if ih > 0:
                    box_area = (
                        (boxes[n, 2] - boxes[n, 0] + 1) *
                        (boxes[n, 3] - boxes[n, 1] + 1)
                    )
                    inter_area = iw * ih
                    denom = (qbox_area + box_area - inter_area)
                    intersec[n, k] = inter_area / denom
    return intersec


def bbox_overlaps_py(boxes, query_boxes):
    """
    Args:
        boxes: (N, 4) ndarray of float
        query_boxes: (K, 4) ndarray of float

    Returns:
        overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    N = boxes.shape[0]
    K = query_boxes.shape[0]
    overlaps = np.zeros((N, K), dtype=np.float)
    for k in range(K):
        box_area = (
            (query_boxes[k, 2] - query_boxes[k, 0] + 1) *
            (query_boxes[k, 3] - query_boxes[k, 1] + 1)
        )
        for n in range(N):
            iw = (
                min(boxes[n, 2], query_boxes[k, 2]) -
                max(boxes[n, 0], query_boxes[k, 0]) + 1
            )
            if iw > 0:
                ih = (
                    min(boxes[n, 3], query_boxes[k, 3]) -
                    max(boxes[n, 1], query_boxes[k, 1]) + 1
                )
                if ih > 0:
                    ua = float(
                        (boxes[n, 2] - boxes[n, 0] + 1) *
                        (boxes[n, 3] - boxes[n, 1] + 1) +
                        box_area - iw * ih
                    )
                    overlaps[n, k] = iw * ih / ua
    return overlaps


def yolo_to_bbox(aoff_pred, anchors, H, W):
    """
    Transform anchored predictions and predicted relative offsets into absolute
    bounding boxes. A cython version is available.

    Args:
        aoff_pred : [B, W * H, A, 4] predicted anchor offsets
        anchors : [A, 2] anchor box aspect ratios

    Returns:
        bbox_out : [B, W * H, A, 4] absolute bounding boxes independent of
            anchors, but scaled in the range 0 to 1.

    Example:
        >>> B, W, H, A = 1, 3, 3, 5
        >>> aoff_pred = np.random.randn(B, W * H, A, 4)
        >>> anchors = np.abs(np.random.randn(A, 2))
        >>> bbox_out1 = yolo_to_bbox_py(aoff_pred, anchors, H, W)
        >>> bbox_out2 = yolo_to_bbox(aoff_pred, anchors, H, W)
        >>> assert np.all(bbox_out1 == bbox_out2)

    Timeit:
       >>> %timeit yolo_to_bbox_py(aoff_pred, anchors, H, W)
       136 µs ± 2.17 µs per loop
       >>> %timeit yolo_to_bbox(aoff_pred, anchors, H, W)
       42.9 µs ± 1.43 µs per loop
    """
    return yolo_to_bbox_c(aoff_pred, anchors, H, W)


def yolo_to_bbox_py(aoff_pred, anchors, H, W):
    H = int(H)
    W = int(W)
    bsize = aoff_pred.shape[0]
    num_anchors = anchors.shape[0]
    bbox_out = np.zeros((bsize, H * W, num_anchors, 4), dtype=np.float)
    for b in range(bsize):
        for row, col in it.product(range(H), range(W)):
            ind = row * W + col
            for a in range(num_anchors):
                cx = (aoff_pred[b, ind, a, 0] + col) / W
                cy = (aoff_pred[b, ind, a, 1] + row) / H
                bw = aoff_pred[b, ind, a, 2] * anchors[a][0] / W * 0.5
                bh = aoff_pred[b, ind, a, 3] * anchors[a][1] / H * 0.5

                bbox_out[b, ind, a, 0] = cx - bw
                bbox_out[b, ind, a, 1] = cy - bh
                bbox_out[b, ind, a, 2] = cx + bw
                bbox_out[b, ind, a, 3] = cy + bh
    return bbox_out


def to_tlbr(xywh):
    """
    convert xywh bounding box to tlbr format

    Example:
        >>> xywh = np.array([0, 0, 10, 10])
        >>> tlbr = to_tlbr(xywh)
        >>> print('tlbr = {!r}'.format(tlbr))
        tlbr = array([ 0,  0, 10, 10])
    """
    tl = xy = xywh[..., 0:2]
    wh = xywh[..., 2:4]
    br = xy + wh
    if isinstance(xywh, np.ndarray):
        tlbr = np.concatenate([tl, br], axis=-1)
    else:
        tlbr = torch.cat([tl, br], dim=-1)
    return tlbr


def to_xywh(tlbr):
    """
    convert tlbr bounding box to xywh format

    Example:
        >>> tlbr = torch.FloatTensor([[5, 5, 11, 10]])
        >>> xywh = to_xywh(tlbr)
        >>> print('xywh = {!r}'.format(xywh))
        xywh =
         5  5  6  5
        [torch.FloatTensor of size (1,4)]
    """
    xy = tl = tlbr[..., 0:2]
    br = tlbr[..., 2:4]
    wh = br - tl
    if isinstance(tlbr, np.ndarray):
        xywh = np.concatenate([xy, wh], axis=-1)
    else:
        xywh = torch.cat([xy, wh], dim=-1)
    return xywh


def random_boxes(num, box_format='tlbr', scale=100):
    if num:
        xywh = (torch.rand(num, 4) * scale)
        if isinstance(scale, int):
            xywh = xywh.long()
        if box_format == 'xywh':
            return xywh
        elif box_format == 'tlbr':
            return to_tlbr(xywh)
        else:
            raise KeyError(box_format)
    else:
        if isinstance(scale, int):
            return torch.LongTensor()
        else:
            return torch.FloatTensor()


def clip_boxes(boxes, im_shape):
    """
    Clip boxes to image boundaries.
    """
    if boxes.shape[0] == 0:
        return boxes

    assert isinstance(boxes, (np.ndarray, list)), 'got boxes={}'.format(boxes)

    # x1 >= 0
    boxes[:, 0::4] = np.maximum(np.minimum(boxes[:, 0::4], im_shape[1] - 1), 0)
    # y1 >= 0
    boxes[:, 1::4] = np.maximum(np.minimum(boxes[:, 1::4], im_shape[0] - 1), 0)
    # x2 < im_shape[1]
    boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], im_shape[1] - 1), 0)
    # y2 < im_shape[0]
    boxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], im_shape[0] - 1), 0)
    return boxes


def nms_detections(pred_boxes, scores, nms_thresh):
    dets = np.hstack((pred_boxes,
                      scores[:, np.newaxis])).astype(np.float32)
    from clab import xpu_device
    device = xpu_device.XPU.default_gpu()
    keep = nms(dets, nms_thresh, device=device)
    return keep


# def _offset_boxes(boxes, im_shape, scale, offs, flip):
#     if len(boxes) == 0:
#         return boxes
#     boxes = np.asarray(boxes, dtype=np.float)
#     boxes *= scale
#     boxes[:, 0::2] -= offs[0]
#     boxes[:, 1::2] -= offs[1]
#     boxes = clip_boxes(boxes, im_shape)

#     if flip:
#         boxes_x = np.copy(boxes[:, 0])
#         boxes[:, 0] = im_shape[1] - boxes[:, 2]
#         boxes[:, 2] = im_shape[1] - boxes_x

#     return boxes


# def preprocess_train(data, size_index):
#     im_path, blob, inp_size = data
#     inp_size = inp_size[size_index]
#     boxes, gt_classes = blob['boxes'], blob['gt_classes']

#     im = cv2.imread(im_path)
#     ori_im = np.copy(im)

#     im, trans_param = imcv2_affine_trans(im)
#     scale, offs, flip = trans_param
#     boxes = _offset_boxes(boxes, im.shape, scale, offs, flip)

#     if inp_size is not None:
#         w, h = inp_size
#         boxes[:, 0::2] *= float(w) / im.shape[1]
#         boxes[:, 1::2] *= float(h) / im.shape[0]
#         im = cv2.resize(im, (w, h))
#     im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
#     im = imcv2_recolor(im)
#     # im /= 255.

#     # im = imcv2_recolor(im)
#     # h, w = inp_size
#     # im = cv2.resize(im, (w, h))
#     # im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
#     # im /= 255
#     boxes = np.asarray(boxes, dtype=np.int)
#     return im, boxes, gt_classes, [], ori_im


# def preprocess_test(data, size_index):
#     im, _, inp_size = data
#     inp_size = inp_size[size_index]
#     if isinstance(im, str):
#         im = cv2.imread(im)
#     ori_im = np.copy(im)

#     if inp_size is not None:
#         w, h = inp_size
#         im = cv2.resize(im, (w, h))
#     im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
#     im = im / 255.
#     return im, [], [], [], ori_im


# def postprocess(bbox_pred, iou_pred, prob_pred, im_shape, cfg, thresh=0.05,
#                 size_index=0):
#     """
#     bbox_pred: (bsize, HxW, num_anchors, 4)
#                ndarray of float (sig(tx), sig(ty), exp(tw), exp(th))
#     iou_pred: (bsize, HxW, num_anchors, 1)
#     prob_pred: (bsize, HxW, num_anchors, num_classes)
#     """

#     # num_classes, num_anchors = cfg.num_classes, cfg.num_anchors
#     num_classes = cfg.num_classes
#     anchors = cfg.anchors
#     W, H = cfg.multi_scale_out_size[size_index]
#     assert bbox_pred.shape[0] == 1, 'postprocess only support one image per batch'  # noqa

#     bbox_pred = yolo_to_bbox(
#         np.ascontiguousarray(bbox_pred, dtype=np.float),
#         np.ascontiguousarray(anchors, dtype=np.float),
#         H, W)
#     bbox_pred = np.reshape(bbox_pred, [-1, 4])
#     bbox_pred[:, 0::2] *= float(im_shape[1])
#     bbox_pred[:, 1::2] *= float(im_shape[0])
#     bbox_pred = bbox_pred.astype(np.int)

#     iou_pred = np.reshape(iou_pred, [-1])
#     prob_pred = np.reshape(prob_pred, [-1, num_classes])

#     cls_inds = np.argmax(prob_pred, axis=1)
#     prob_pred = prob_pred[(np.arange(prob_pred.shape[0]), cls_inds)]
#     scores = iou_pred * prob_pred
#     # scores = iou_pred

#     # threshold
#     keep = np.where(scores >= thresh)
#     bbox_pred = bbox_pred[keep]
#     scores = scores[keep]
#     cls_inds = cls_inds[keep]

#     # NMS
#     keep = np.zeros(len(bbox_pred), dtype=np.int)
#     for i in range(num_classes):
#         inds = np.where(cls_inds == i)[0]
#         if len(inds) == 0:
#             continue
#         c_bboxes = bbox_pred[inds]
#         c_scores = scores[inds]
#         c_keep = nms_detections(c_bboxes, c_scores, 0.3)
#         keep[inds[c_keep]] = 1

#     keep = np.where(keep > 0)
#     # keep = nms_detections(bbox_pred, scores, 0.3)
#     bbox_pred = bbox_pred[keep]
#     scores = scores[keep]
#     cls_inds = cls_inds[keep]

#     # clip
#     bbox_pred = clip_boxes(bbox_pred, im_shape)

#     return bbox_pred, scores, cls_inds


# def _bbox_targets_perimage(im_shape, gt_boxes, cls_inds, dontcare_areas, cfg):
#     # num_classes, num_anchors = cfg.num_classes, cfg.num_anchors
#     # anchors = cfg.anchors
#     H, W = cfg.out_size
#     gt_boxes = np.asarray(gt_boxes, dtype=np.float)
#     # TODO: dontcare areas
#     dontcare_areas = np.asarray(dontcare_areas, dtype=np.float)

#     # locate the cell of each gt_boxe
#     cell_w = float(im_shape[1]) / W
#     cell_h = float(im_shape[0]) / H
#     cx = (gt_boxes[:, 0] + gt_boxes[:, 2]) * 0.5 / cell_w
#     cy = (gt_boxes[:, 1] + gt_boxes[:, 3]) * 0.5 / cell_h
#     cell_inds = np.floor(cy) * W + np.floor(cx)
#     cell_inds = cell_inds.astype(np.int)

#     # [x1, y1, x2, y2],  [class]
#     # gt_boxes[:, 0::2] /= im_shape[1]
#     # gt_boxes[:, 1::2] /= im_shape[0]
#     # gt_boxes[:, 0] = cx - np.floor(cx)
#     # gt_boxes[:, 1] = cy - np.floor(cy)
#     # gt_boxes[:, 2] = (gt_boxes[:, 2] - gt_boxes[:, 0]) / im_shape[1]
#     # gt_boxes[:, 3] = (gt_boxes[:, 3] - gt_boxes[:, 1]) / im_shape[0]

#     bbox_target = [[] for _ in range(H * W)]
#     cls_target = [[] for _ in range(H * W)]
#     for i, ind in enumerate(cell_inds):
#         bbox_target[ind].append(gt_boxes[i])
#         cls_target[ind].append(cls_inds[i])
#     return bbox_target, cls_target


# def get_bbox_targets(images, gt_boxes, cls_inds, dontcares, cfg):
#     bbox_targets = []
#     cls_targets = []
#     for i, im in enumerate(images):
#         bbox_target, cls_target = _bbox_targets_perimage(im.shape,
#                                                          gt_boxes[i],
#                                                          cls_inds[i],
#                                                          dontcares[i],
#                                                          cfg)
#         bbox_targets.append(bbox_target)
#         cls_targets.append(cls_target)
#     return bbox_targets, cls_targets


# def draw_detection(im, bboxes, scores, cls_inds, cfg, thr=0.3):
#     # draw image
#     colors = cfg.colors
#     labels = cfg.label_names

#     imgcv = np.copy(im)
#     h, w, _ = imgcv.shape
#     for i, box in enumerate(bboxes):
#         if scores[i] < thr:
#             continue
#         cls_indx = cls_inds[i]

#         thick = int((h + w) / 300)
#         cv2.rectangle(imgcv,
#                       (box[0], box[1]), (box[2], box[3]),
#                       colors[cls_indx], thick)
#         mess = '%s: %.3f' % (labels[cls_indx], scores[i])
#         cv2.putText(imgcv, mess, (box[0], box[1] - 12),
#                     0, 1e-3 * h, colors[cls_indx], thick // 3)

#     return imgcv
