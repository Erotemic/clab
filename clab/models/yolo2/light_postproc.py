#
#   Lightnet related postprocessing
#   Thers are functions to transform the output of the network to brambox detection objects
#   Copyright EAVISE
#

import logging
import torch
from torch.autograd import Variable
from clab.util import profiler
import numpy as np
import ubelt as ub
from clab.models.yolo2.utils import yolo_utils

log = logging.getLogger(__name__)


class GetBoundingBoxes(object):
    """ Convert output from darknet networks to bounding box tensor.

    Args:
        network (lightnet.network.Darknet): Network the converter will be used with
        conf_thresh (Number [0-1]): Confidence threshold to filter detections
        nms_thresh(Number [0-1]): Overlapping threshold to filter detections with non-maxima suppresion

    Returns:
        (Batch x Boxes x 6 tensor): **[x_center, y_center, width, height, confidence, class_id]** for every bounding box

    Note:
        The output tensor uses relative values for its coordinates.
    """
    def __init__(self, network=None, conf_thresh=0.001, nms_thresh=0.4, anchors=None, num_classes=None):
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        if anchors is not None:
            self.num_classes = num_classes
            self.anchors = anchors['values']
            self.num_anchors = anchors['num']
            self.anchor_step = len(self.anchors) // self.num_anchors
        else:
            self.num_classes = network.num_classes
            self.anchors = network.anchors
            self.num_anchors = network.num_anchors
            self.anchor_step = len(self.anchors) // self.num_anchors

    def __call__(self, network_output):
        """ Compute bounding boxes after thresholding and nms

            network_output (torch.autograd.Variable): Output tensor from the lightnet network

        Examples:
            >>> import torch
            >>> torch.random.manual_seed(0)
            >>> anchors = dict(num=5, values=[1.3221,1.73145,3.19275,4.00944,5.05587,
            >>>                               8.09892,9.47112,4.84053,11.2364,10.0071])
            >>> self = GetBoundingBoxes(anchors=anchors, num_classes=20, conf_thresh=.01, nms_thresh=0.5)
            >>> output = torch.randn(8, 125, 9, 9)
            >>> boxes = self(output)
            >>> assert len(boxes) == 8
            >>> assert all(b.shape[1] == 6 for b in boxes)
        """
        boxes = self._get_boxes(network_output.data)
        boxes = [self._nms(box) for box in boxes]
        return boxes

    @classmethod
    def apply(cls, network_output, network, conf_thresh, nms_thresh):
        obj = cls(network, conf_thresh, nms_thresh)
        return obj(network_output)

    def _get_boxes(self, output):
        """
        Returns array of detections for every image in batch

        Examples:
            >>> import torch
            >>> torch.random.manual_seed(0)
            >>> anchors = dict(num=5, values=[1.3221,1.73145,3.19275,4.00944,5.05587,
            >>>                               8.09892,9.47112,4.84053,11.2364,10.0071])
            >>> self = GetBoundingBoxes(anchors=anchors, num_classes=20, conf_thresh=.01, nms_thresh=0.5)
            >>> output = torch.randn(8, 125, 9, 9)
            >>> boxes = self._get_boxes(output.data)
            >>> assert len(boxes) == 8
            >>> assert all(len(b[0]) == 6 for b in boxes)
        """

        # Check dimensions
        if output.dim() == 3:
            output.unsqueeze_(0)

        # Variables
        cuda = output.is_cuda
        batch = output.size(0)
        h = output.size(2)
        w = output.size(3)

        # Compute xc,yc, w,h, box_score on Tensor
        lin_x = torch.linspace(0, w-1, w).repeat(h,1).view(h*w)
        lin_y = torch.linspace(0, h-1, h).repeat(w,1).t().contiguous().view(h*w)
        anchor_w = torch.Tensor(self.anchors[::2]).view(1, self.num_anchors, 1)
        anchor_h = torch.Tensor(self.anchors[1::2]).view(1, self.num_anchors, 1)
        if cuda:
            lin_x = lin_x.cuda()
            lin_y = lin_y.cuda()
            anchor_w = anchor_w.cuda()
            anchor_h = anchor_h.cuda()

        output_ = output.view(batch, self.num_anchors, -1, h*w)  # -1 == 5+num_classes (we can drop feature maps if 1 class)
        output_[:,:,0,:].sigmoid_().add_(lin_x).div_(w)          # X center
        output_[:,:,1,:].sigmoid_().add_(lin_y).div_(h)          # Y center
        output_[:,:,2,:].exp_().mul_(anchor_w).div_(w)           # Width
        output_[:,:,3,:].exp_().mul_(anchor_h).div_(h)           # Height
        output_[:,:,4,:].sigmoid_()                              # Box score

        # Compute class_score
        if self.num_classes > 1:
            if torch.__version__.startswith('0.3'):
                cls_scores = torch.nn.functional.softmax(Variable(output_[:,:,5:,:], volatile=True), 2).data
            else:
                cls_scores = torch.nn.functional.softmax(output_[:,:,5:,:], 2)
            cls_max, cls_max_idx = torch.max(cls_scores, 2)
            cls_max.mul_(output_[:,:,4,:])
        else:
            cls_max = output_[:,:,4,:]
            cls_max_idx = torch.zeros_like(cls_max)

        # Save detection if conf*class_conf is higher than threshold
        output_ = output_.cpu()
        cls_max = cls_max.cpu()
        cls_max_idx = cls_max_idx.cpu()
        boxes = []
        for b in range(batch):
            box_batch = []
            for a in range(self.num_anchors):
                for i in range(h*w):
                    if cls_max[b,a,i] > self.conf_thresh:
                        box_batch.append([
                            output_[b,a,0,i],
                            output_[b,a,1,i],
                            output_[b,a,2,i],
                            output_[b,a,3,i],
                            cls_max[b,a,i],
                            cls_max_idx[b,a,i]
                            ])
            boxes.append(box_batch)

        return boxes

    @profiler.profile
    def _nms(self, boxes, mode=1):
        """ Non maximum suppression.
        Source: https://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/

        Args:
          boxes (tensor): Bounding boxes from get_detections

        Return:
          (tensor): Pruned boxes

        CommandLine:
            python -m clab.models.yolo2.light_postproc GetBoundingBoxes._nms --profile

        Examples:
            >>> import torch
            >>> torch.random.manual_seed(0)
            >>> anchors = dict(num=5, values=[1.3221,1.73145,3.19275,4.00944,5.05587,
            >>>                               8.09892,9.47112,4.84053,11.2364,10.0071])
            >>> self = GetBoundingBoxes(anchors=anchors, num_classes=20, conf_thresh=.01, nms_thresh=0.5)
            >>> output = torch.randn(8, 125, 9, 9)
            >>> boxes_ = self._get_boxes(output.data)
            >>> from clab import util
            >>> boxes = torch.Tensor(boxes_[0])
            >>> scores = boxes[..., 4:5]
            >>> classes = boxes[..., 5:6]
            >>> cxywh = util.Boxes(boxes[..., 0:4], 'cxywh')
            >>> tlbr = cxywh.as_tlbr()
            >>> from clab.models.yolo2.utils import yolo_utils
            >>> yolo_utils.nms_detections(tlbr.data.numpy(), scores.numpy().ravel(), self.nms_thresh)
            >>> self._nms(boxes, mode=0)
            >>> self._nms(boxes, mode=1)

            %timeit yolo_utils.nms_detections(tlbr.data.numpy(), scores.numpy().ravel(), self.nms_thresh)
            import ubelt
            for timer in ubelt.Timerit(100, bestof=10, label='time'):
                with timer:
                    self._nms(box, mode=0)
            for timer in ubelt.Timerit(100, bestof=10, label='time'):
                with timer:
                    self._nms(box, mode=1)
        """
        boxes = torch.Tensor(boxes)

        if boxes.numel() == 0:
            return boxes

        a = boxes[:,:2]
        b = boxes[:,2:4]
        bboxes = torch.cat([a-b/2,a+b/2], 1)
        scores = boxes[:,4]

        if mode == 1:
            bboxes = bboxes.cpu().numpy().astype(np.float32)
            scores = scores.cpu().numpy().astype(np.float32)
            classes = boxes[..., 5].cpu().numpy().astype(np.int)
            keep = []
            for idxs in ub.group_items(range(len(classes)), classes).values():
                cls_boxes = bboxes.take(idxs, axis=0)
                cls_scores = scores.take(idxs, axis=0)
                cls_keep = yolo_utils.nms_detections(cls_boxes, cls_scores, self.nms_thresh)
                keep.extend(list(ub.take(idxs, cls_keep)))
            keep = sorted(keep)
            return boxes[torch.LongTensor(keep)]

        # if torch.cuda.is_available:
        #     boxes = boxes.cuda()

        x1 = bboxes[:,0]
        y1 = bboxes[:,1]
        x2 = bboxes[:,2]
        y2 = bboxes[:,3]

        areas = ((x2-x1) * (y2-y1))
        _, order = scores.sort(0, descending=True)

        keep = []
        while order.numel() > 0:
            if torch.__version__.startswith('0.3'):
                i = order[0]
            else:
                i = order[0].item()
            keep.append(i)

            if order.numel() == 1:
                break

            xx1 = x1[order[1:]].clamp(min=x1[i])
            yy1 = y1[order[1:]].clamp(min=y1[i])
            xx2 = x2[order[1:]].clamp(max=x2[i])
            yy2 = y2[order[1:]].clamp(max=y2[i])

            w = (xx2-xx1).clamp(min=0)
            h = (yy2-yy1).clamp(min=0)
            inter = w*h

            iou = inter / (areas[i] + areas[order[1:]] - inter)

            ids = (iou<=self.nms_thresh).nonzero().squeeze()
            if ids.numel() == 0:
                break
            order = order[ids+1]
        return boxes[torch.LongTensor(keep)]

if __name__ == '__main__':
    """
    CommandLine:
        python -m clab.models.yolo2.light_postproc all
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
