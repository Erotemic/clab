from functools import partial
from multiprocessing import Pool
import torch
import numpy as np
import torch.nn as nn
from .utils import yolo_utils
from . import darknet  # NOQA


def np_to_variable(x, device=None, dtype=torch.FloatTensor):
    v = torch.autograd.Variable(torch.from_numpy(x).type(dtype))
    if device is not None:
        v = v.cuda(device)
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

    From YoloV1:

        Our final layer predicts both class probabilities and bounding box
        coordinates.

        We normalize the bounding box width and height by the image width and
        height so that they fall between 0 and 1.

        We parametrize the bounding box x and y coordinates to be offsets of a
        particular grid cell location so they are also bounded between 0 and 1.

        YOLO predicts multiple bounding boxes per grid cell.
        At training time we only want one bounding box predictor to be
        responsible for each object.

        We assign one predictor to be “responsible” for predicting an object
        based on which prediction has the highest current IOU with the ground
        truth.

        [To avoid instability] we increase the loss from bounding box
        coordinate predictions and decrease the loss from confidence
        predictions for boxes that don’t contain objects.

        We use two parameters, λ_coord and λ_noobj to accomplish this.
        We set λ_coord = 5 and λ_noobj = .5.


    References:
        https://github.com/pjreddie/darknet/blob/56be49aa4854b81b855f6a9daffce4b4ad1fbb9e/src/network.c#L239
        https://github.com/pjreddie/darknet/blob/8215a8864d4ad07e058acafd75b2c6ff6600b9e8/src/detection_layer.c#L67

        https://github.com/pjreddie/darknet/blob/master/src/region_layer.c#L174

        https://stats.stackexchange.com/questions/287486/yolo-loss-function-explanation

    Example:
        >>> from clab.models.yolo2.darknet_loss import *
        >>> model = darknet.Darknet19(num_classes=20)
        >>> criterion = DarknetLoss(model.anchors)
        >>> inp_size = (96, 96)
        >>> inputs, labels = darknet.demo_batch(1, inp_size)
        >>> output = model(*inputs)
        >>> aoff_pred, iou_pred, prob_pred = output
        >>> gt_boxes, gt_classes, orig_size, indices, gt_weights = labels
        >>> loss = criterion(aoff_pred, iou_pred, prob_pred, gt_boxes,
        >>>                  gt_classes, gt_weights, inp_size)
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

        criterion.anchors = np.ascontiguousarray(anchors, dtype=np.float)
        criterion.cls_mse = nn.MSELoss(size_average=False)
        criterion.iou_mse = nn.MSELoss(size_average=False)
        criterion.box_mse = nn.MSELoss(size_average=False)

    def forward(criterion, aoff_pred, iou_pred, prob_pred,
                gt_boxes=None, gt_classes=None, gt_weights=None,
                inp_size=None):
        """
        Args:
            aoff_pred (torch.FloatTensor): anchor bounding box offsets
            iou_pred (torch.FloatTensor): objectness probability prediction
            prob_pred (torch.FloatTensor): conditional probability per class
            gt_boxes (list): groundtruth bounding boxes
            gt_classes (list): groundtruth class indexes
            gt_weights (list): example weights
            inp_size (tuple): WxH of image passed through the network
        """

        n_classes = prob_pred.shape[-1]

        # Transform groundtruth into formats comparable to predictions
        _tup = criterion._build_target(aoff_pred, iou_pred, gt_boxes,
                                       gt_classes, gt_weights, inp_size,
                                       n_classes, criterion.anchors)
        _aoffs, _ious, _classes, _aoff_mask, _iou_mask, _class_mask = _tup

        device = criterion.get_device()

        def logit(p):
            return -np.log((1 / p) - 1)

        logit(_aoffs[..., 0:2])

        aoff_true = np_to_variable(_aoffs, device)
        iou_true  = np_to_variable(_ious, device)
        onehot_class_true = np_to_variable(_classes, device)

        aoff_mask = np_to_variable(_aoff_mask, device, dtype=torch.FloatTensor)
        iou_mask = np_to_variable(_iou_mask, device, dtype=torch.FloatTensor)
        class_mask = np_to_variable(_class_mask, device,
                                    dtype=torch.FloatTensor)

        aoff_mask = aoff_mask.expand_as(aoff_true)
        class_mask = class_mask.expand_as(prob_pred)

        criterion.bbox_loss = criterion.box_mse(
            aoff_pred * aoff_mask, aoff_true * aoff_mask)

        criterion.iou_loss = criterion.iou_mse(
            iou_pred * iou_mask, iou_true * iou_mask)

        criterion.cls_loss = criterion.cls_mse(
            prob_pred * class_mask, onehot_class_true * class_mask)

        # Is this right? What if there are no boxes?
        # Shouldn't we divide by number of predictions or nothing?
        # num_boxes = sum(len(boxes) for boxes in gt_boxes)
        # criterion.bbox_loss /= num_boxes
        # criterion.iou_loss /= num_boxes
        # criterion.cls_loss /= num_boxes

        total_loss = criterion.bbox_loss + criterion.iou_loss + criterion.cls_loss
        return total_loss

    def _build_target(criterion, aoff_pred, iou_pred, gt_boxes, gt_classes,
                      gt_weights, inp_size, n_classes, anchors):
        """
        Determine which ground truths to compare against which predictions?

        Args:
            aoff_pred: [B, H x W, A, 4]:
                (sig(tx), sig(ty), exp(tw), exp(th))

        Example:
            >>> from clab.models.yolo2.darknet_loss import *
            >>> n_classes = 20
            >>> model = darknet.Darknet19(num_classes=n_classes)
            >>> criterion = DarknetLoss(model.anchors)
            >>> inp_size = (96, 96)
            >>> inputs, labels = darknet.demo_batch(10, inp_size)
            >>> output = model(*inputs)
            >>> aoff_pred, iou_pred, prob_pred = output
            >>> gt_boxes, gt_classes, orig_size, indices, gt_weights = labels
            >>> _tup = criterion._build_target(aoff_pred, iou_pred, gt_boxes,
            >>>                                gt_classes, gt_weights, inp_size,
            >>>                                n_classes, criterion.anchors)
        """
        losskw = dict(object_scale=criterion.object_scale,
                      noobject_scale=criterion.noobject_scale,
                      class_scale=criterion.class_scale,
                      coord_scale=criterion.coord_scale,
                      iou_thresh=criterion.iou_thresh)

        func = partial(build_target_item, inp_size=inp_size,
                       n_classes=n_classes, anchors=anchors, **losskw)

        # convert to numpy?
        aoff_pred_np = aoff_pred.data.cpu().numpy()
        iou_pred_np = iou_pred.data.cpu().numpy()

        gt_boxes_np = [item.data.cpu().numpy() for item in gt_boxes]
        gt_classes_np = [item.data.cpu().numpy() for item in gt_classes]
        gt_weights_np = [item.data.cpu().numpy() for item in gt_weights]

        args = zip(aoff_pred_np, iou_pred_np, gt_boxes_np, gt_classes_np,
                   gt_weights_np)
        # args = list(args)
        # data = args[0]

        targets = list(map(func, args))

        _aoffs = np.stack(tuple((row[0] for row in targets)))
        _ious = np.stack(tuple((row[1] for row in targets)))
        _classes = np.stack(tuple((row[2] for row in targets)))
        _aoff_mask = np.stack(tuple((row[3] for row in targets)))
        _iou_mask = np.stack(tuple((row[4] for row in targets)))
        _class_mask = np.stack(tuple((row[5] for row in targets)))

        return _aoffs, _ious, _classes, _aoff_mask, _iou_mask, _class_mask


def build_target_item(data, inp_size, n_classes, anchors, object_scale=5.0,
                      noobject_scale=1.0, class_scale=1.0, coord_scale=1.0,
                      iou_thresh=0.5):
    """

    Constructs the relevant ground truth terms of the YOLO loss function

    Assign predicted boxes to groundtruth boxes and compute their IOU for the
    loss calculation.o

    The output of this is used as the truth in the MSE calculation between
    this and the predictions.

    Notes:
        This function deals with several spaces:
           input coordinates: input size of the images fed to the network
           output coordinates: downsampled input coordinates (by factor of 32)
           norm coordinates: image coordinates where the bottom right is (1, 1)
        Each space can be absolute or relative to grid cell centers.

        groundtruth boxes must be specified in the abs input image space
            (input not orig)

    Example:
        >>> from clab.models.yolo2.darknet_loss import *
        >>> inp_size = (96, 96)
        >>> n_classes = 3
        >>> data, anchors = demo_npdata(inp_size=inp_size, C=n_classes)
        >>> object_scale = 5.0
        >>> noobject_scale = 1.0
        >>> class_scale = 1.0
        >>> coord_scale = 1.0
        >>> iou_thresh = 0.5
        >>> _tup = build_target_item(data, inp_size, n_classes, anchors)
        >>> _aoffs, _ious, _classes, _aoff_mask, _iou_mask, _class_mask = _tup

    Example:
        >>> from clab.models.yolo2.darknet_loss import *
        >>> inp_size = (96, 96)
        >>> n_classes = 20
        >>> data, anchors = demo_npdata(inp_size=inp_size, C=n_classes, n=0)
        >>> _tup = build_target_item(data, inp_size, n_classes, anchors)
        >>> _aoffs, _ious, _classes, _aoff_mask, _iou_mask, _class_mask = _tup
    """
    # gt_weights generalizes dontcare
    aoff_pred_np, iou_pred_np, gt_boxes_np, gt_classes_np, gt_weights_np = data

    # PREPARE INPUT
    # -------------
    aoff_pred_np = np.ascontiguousarray(aoff_pred_np, dtype=np.float)
    gt_boxes_np = np.ascontiguousarray(gt_boxes_np.reshape([-1, 4]),
                                       dtype=np.float)

    # Output size of the grid (which is a factor of 32 less than inp_size)
    out_size = [s // 32 for s in inp_size]
    # Input pixel w/h and output grid w/h
    in_w, in_h = map(float, inp_size)
    out_w, out_h = map(float, out_size)

    n_cells, n_anchors, _ = aoff_pred_np.shape
    n_real = len(gt_boxes_np)
    anchors = np.ascontiguousarray(anchors, dtype=np.float)
    # assert n_cells == out_w * out_h

    # APPLY OFFSETS TO EACH ANCHOR BOX
    # --------------------------------
    # get prediected boxes in absolute normalized coordinates, then
    # scale bounding boxes into absolute input image space (tlbr format)
    bbox_norm_pred = yolo_utils.yolo_to_bbox(
        aoff_pred_np[None, :], anchors, out_h, out_w)[0]
    bbox_abs_pred = scale_bbox(bbox_norm_pred, in_w, in_h)

    # FIND IOU BETWEEN ALL PAIRS OF (PRED x TRUE) BOXES
    # -------------------------------------------------
    # for each cell, compare predicted_bbox and gt_bbox
    ious, best_ious = _pred_true_overlap(bbox_abs_pred, gt_boxes_np)

    # ASSIGN EACH TRUE BOX TO A GRID CELL AND ANCHOR BOX
    # --------------------------------------------------
    # for each gt boxes, match the best anchor
    # This makes that anchor responsible for a particular groundtruth
    # Transform groundtruth from input coordinates to relative output
    # coordinates and locate the cell it should be relative to
    gt_aoff, gt_anchor_inds, gt_cell_inds = _bbox_to_yolo_flat(gt_boxes_np,
                                                               anchors,
                                                               inp_size,
                                                               out_size)

    # POPULATE OUTPUT
    # -----------------

    # original yolo options that we hard code in to help make a correspondence
    # between the original loss function and this one.
    RESCORE = True
    # Ignore BACKGROUND sections it is 0 for voc.cfg in darknet

    # Construct data corresponding to prediction shapes and populate items
    # corresponding with gt-assignments to positive labels and everything else
    # to negative labels.

    # For every possible anchor we either make it responsible for a groundtruth
    # label or make it responsible for predicting no-object (wrt to this item).

    # Indicator matrix (one-hot encoding of assigned groundtruth class)
    _classes = np.zeros([n_cells, n_anchors, n_classes], dtype=np.float)
    _class_mask = np.zeros([n_cells, n_anchors, 1], dtype=np.float)

    # iou of the assigned groundtruth box
    _ious = np.zeros([n_cells, n_anchors, 1], dtype=np.float)
    _iou_mask = np.zeros([n_cells, n_anchors, 1], dtype=np.float)

    # assigned groundtruth boxes in anchor offset format
    _aoffs = np.zeros([n_cells, n_anchors, 4], dtype=np.float)
    _aoff_mask = np.zeros([n_cells, n_anchors, 1], dtype=np.float)

    # HANDLE NO-OBJECT CASES
    # ----------------------
    # do we want to encourage no-objects to move predictions back to the
    # anchors? If so how badly?

    # From YoloV3:
    # If a bounding box prior is not assigned to a ground truth object it
    # incurs no loss for coordinate or class predictions, only objectness.
    # _aoff_mask += 0.01

    # NOTE: other code expects offsets to be from the top left
    # corner of a grid cell. Should (0, 0) be the center?
    _aoffs[:, :, 0:2] = 0.5  # cell center in relative output coordinates
    _aoffs[:, :, 2:4] = 1.0  # size of one cell in relative output coords

    # Flags that denotes if the prediction does not overlap a real object
    noobj_flags = best_ious < iou_thresh
    _iou_mask[noobj_flags] = noobject_scale

    # iou_penalty = 0 - iou_pred_np[noobj_flags]
    # _iou_mask[noobj_flags] = noobject_scale * iou_penalty

    # HANDLE ASSIGNED OBJECT CASES
    # ----------------------------
    # Place the flat groundtruth with cell and anchor locations
    # into the dense structure corresponding with the network predictions.

    # See:
    # https://github.com/pjreddie/darknet/blob/master/src/region_layer.c#L158

    # TODO / FIXME: What happens when more than one class is assigned to the
    # same anchor box? Does real YOLO do anything special there?

    ious_reshaped = ious.reshape([n_cells, n_anchors, n_real])
    for gt_idx, cell_idx in enumerate(gt_cell_inds):
        # Ignore groundtruth outside of image bounds
        if cell_idx >= n_cells or cell_idx < 0:
            continue

        ax = gt_anchor_inds[gt_idx]
        gt_weight = gt_weights_np[gt_idx]

        # Populate mask for assigned object loss
        # 0 ~ 1, should be close to 1
        # pred_iou = iou_pred_np[cell_idx, ax, :]

        # _iou_mask[cell_idx, ax, :] = object_scale * (1 - pred_iou) * gt_weight
        if RESCORE:
            _ious[cell_idx, ax, :] = ious_reshaped[cell_idx, ax, gt_idx]
        else:
            _ious[cell_idx, ax, :] = 1

        _iou_mask[cell_idx, ax, :] = object_scale * gt_weight

        _aoffs[cell_idx, ax, :] = gt_aoff[gt_idx]
        _aoff_mask[cell_idx, ax, :] = coord_scale * gt_weight

        _classes[cell_idx, ax, gt_classes_np[gt_idx]] = 1.
        _class_mask[cell_idx, ax, :] = class_scale * gt_weight

    return _aoffs, _ious, _classes, _aoff_mask, _iou_mask, _class_mask


def _pred_true_overlap(bbox_abs_pred, gt_boxes_np):
    """
    Find iou between all pairs of (pred x true) boxes

    Args:
        bbox_abs_pred (ndarray): pred boxes in absolute input space
        gt_boxes_np (ndarray): true boxes in absolute input space

    Returns:
        tuple:
            ious (n_pred x n_real) - matrix
            best_outs - maps each predicted bbox index to groundtruth iou

    Example:
        >>> from clab.models.yolo2.darknet_loss import *
        >>> A, H, W = 5, 3, 3
        >>> inp_size = (96, 96)
        >>> out_size = (H, W)
        >>> data, anchors = demo_npdata(A, H, W, inp_size)
        >>> gt_boxes_np = data[2]
        >>> aoff_pred_np = data[0]
        >>> bbox_norm_pred = yolo_utils.yolo_to_bbox(
        >>>     aoff_pred_np[None, :], anchors, H, W)[0]
        >>> bbox_abs_pred = scale_bbox(bbox_norm_pred, *inp_size)
        >>> ious, best_ious = _pred_true_overlap(
        >>>     bbox_abs_pred, gt_boxes_np)
        >>> print('ious = {!r}'.format(ious))
        >>> print('best_ious = {!r}'.format(best_ious))
    """
    # for each cell, compare predicted_bbox and gt_bbox
    n_cells, n_anchors, _ = bbox_abs_pred.shape

    # compute IOU matrix [n_pred x n_real]
    bbox_abs_pred_ = bbox_abs_pred.reshape([-1, 4])
    ious = yolo_utils.bbox_ious(
        np.ascontiguousarray(bbox_abs_pred_, dtype=np.float),
        np.ascontiguousarray(gt_boxes_np, dtype=np.float)
    )
    # determine which iou is best
    # TODO use argmax instead?
    if ious.size > 0:
        # For each predicted box, determine if it overlaps a real object
        best_ious = np.max(ious, axis=1).reshape([n_cells, n_anchors, 1])
    else:
        best_ious = np.empty(0)
    return ious, best_ious


def _bbox_to_yolo_flat(gt_boxes_np, anchors, inp_size, out_size):
    """
    Transforms tlbr groundtruth boxes into YOLO anchor box offsets.

    Args:
        gt_boxes_np (ndarray): [N, 4] tlbr groundtruth boxes in input
            coordinates for a single item in a batch.

    Returns:
        tuple:
            gt_cell_inds - cell indices corresponding to xywh bounding boxes
            gt_aoff - with xy positions relative to the center of their
                cells and wh relative to output coordinates.

    Example:
        >>> from clab.models.yolo2.darknet_loss import *
        >>> A, H, W = 5, 3, 3
        >>> inp_size = (96, 96)
        >>> out_size = (H, W)
        >>> data, anchors = demo_npdata(A, H, W, inp_size, rng=1)
        >>> gt_boxes_np = data[2]
        >>> _yf = _bbox_to_yolo_flat(gt_boxes_np, anchors, inp_size, out_size)
        >>> gt_aoff, gt_anchor_inds, gt_cell_inds = _yf
        >>> print('gt_cell_inds = {}'.format(gt_cell_inds))
        >>> print('gt_aoff = {}'.format(gt_aoff))
        >>> boxes_out = yolo_utils.flat_yolo_to_bbox_py(gt_aoff, gt_anchor_inds,
        >>>                                             gt_cell_inds, anchors,
        >>>                                             out_size, inp_size)
        >>> assert np.allclose(boxes_out, gt_boxes_np)
    """
    # locate the cell of each gt_box
    # determine which cell each ground truth box belongs to

    # Input pixel w/h and output grid w/h
    in_w, in_h = map(float, inp_size)
    out_w, out_h = map(float, out_size)

    # Number of pixels each cell corresponds to (should be 32)
    cell_w = in_w / out_w
    cell_h = in_h / out_h

    # Scale transform from input to output coordinates
    sf_x, sf_y = (out_w / in_w), (out_h / in_h)

    # Centers of the groundtruth boxes in output coordinates
    x1, y1 = gt_boxes_np[:, 0], gt_boxes_np[:, 1]
    x2, y2 = gt_boxes_np[:, 2], gt_boxes_np[:, 3]
    cx = (x1 + x2) * 0.5 / cell_w
    cy = (y1 + y2) * 0.5 / cell_h

    # Groundtruth width / height (in input coordinates)
    gt_width_in  = x2 - x1
    gt_height_in = y2 - y1

    # Groundtruth width / height (in output coordinates)
    gt_width_out = gt_width_in * sf_x
    gt_height_out = gt_height_in * sf_y

    # each anchor corresponds to one of $A$ predictors in the grid cell.
    # we want each predictor to be responsible for only one ground truth object
    gt_boxes_out = scale_bbox(gt_boxes_np, sf_x, sf_y)
    anchor_ious = yolo_utils.anchor_intersections(anchors, gt_boxes_out)
    gt_anchor_inds = np.argmax(anchor_ious, axis=0)

    # Lookup indices of which cell the groundtruth belongs to
    gt_cell_inds = np.floor(cy) * out_w + np.floor(cx)
    gt_cell_inds = gt_cell_inds.astype(np.int)

    # Translate each groundtruth box to be relative to its respective cell
    # Transform w/h to be a multiple of the assigned anchor size.
    gt_anchors = anchors[gt_anchor_inds]
    gt_exp_width = gt_width_out / gt_anchors.T[0]
    gt_exp_height = gt_height_out / gt_anchors.T[1]

    # Convert the centered output bounding box to yolo format (where w/h is
    # a multiple of the assigned anchor)
    gt_aoff = np.empty(gt_boxes_np.shape, dtype=np.float)
    gt_aoff[:, 0] = cx - np.floor(cx)  # cx
    gt_aoff[:, 1] = cy - np.floor(cy)  # cy
    gt_aoff[:, 2] = gt_exp_width  # tw
    gt_aoff[:, 3] = gt_exp_height  # th

    return gt_aoff, gt_anchor_inds, gt_cell_inds


def scale_bbox(bboxes, sf_x, sf_y):
    """ works with tlbr or xywh """
    bboxes = bboxes.copy()
    bboxes[..., 0:4:2] *= sf_x
    bboxes[..., 1:4:2] *= sf_y
    return bboxes


def demo_npdata(A=5, H=3, W=3, inp_size=(96, 96), C=20, factor=32, n=None,
                rng=None):
    from clab import util
    rng = util.ensure_rng(rng)
    B = 1
    out_size = np.array([W, H])
    n_classes = C

    inputs, labels = darknet.demo_batch(B, inp_size, n_classes=n_classes,
                                        rng=rng, n=n)
    aoff_pred, iou_pred = darknet.demo_predictions(1, H, W, A, rng=rng)

    gt_boxes, gt_classes, orig_size, indices, gt_weights = labels
    aoff_pred_np = aoff_pred.numpy()[0].astype(np.float)
    iou_pred_np = iou_pred.numpy()[0].astype(np.float)
    gt_boxes_np = [item.cpu().numpy().astype(np.float) for item in gt_boxes][0]
    gt_classes_np = [item.cpu().numpy() for item in gt_classes][0]
    gt_weights_np = [item.cpu().numpy() for item in gt_weights][0]

    gt_boxes_np = gt_boxes_np.reshape(-1, 4)

    # Random anchors specified w.r.t output shape
    anchors = np.abs(rng.randn(A, 2) * out_size).astype(np.float)

    data = (aoff_pred_np, iou_pred_np,
            gt_boxes_np, gt_classes_np, gt_weights_np)
    return data, anchors


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m clab.models.yolo2.darknet_loss all
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
