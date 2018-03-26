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


class Darknet19(nn.Module):
    """
    Example:
        >>> from clab.models.yolo2.darknet import *
        >>> self = Darknet19(num_classes=20)
        >>> im_data = torch.randn(1, 3, 224, 224)
        >>> output = self(im_data)
        >>> aoff_pred, iou_pred, prob_pred = output
        >>> print('aoff_pred.shape = {!r}'.format(tuple(aoff_pred.shape)))
        aoff_pred.shape = (1, 49, 5, 4)
        >>> print('iou_pred.shape = {!r}'.format(tuple(iou_pred.shape)))
        iou_pred.shape = (1, 49, 5, 1)
        >>> print('prob_pred.shape = {!r}'.format(tuple(prob_pred.shape)))
        prob_pred.shape = (1, 49, 5, 20)
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

        # linear: add 5 extra outputs at the start for the 4 bounding box
        # offsets and the objectness score.
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
        # aoff = [sig(tx), sig(ty), exp(tw), exp(th), sig(to)]
        """
        From YOLO9000.pdf:

            Instead of predicting offsets we follow the approach of YOLO[v1]
            and predict location coordinates relative to the location of the
            grid cell. This bounds the ground truth to fall between 0 and 1. We
            use a logistic activation to constrain the network’s predictions to
            fall in this range.

            The network predicts 5 bounding boxes at each cell in the output
            feature map.

            The network predicts 5 coordinates for each bounding box,
            tx, ty, tw, th, and to.

            If the cell is offset from the top left corner of the image by (cx,
            cy) and the bounding box prior has width and height pw, ph, then
            the predictions correspond to:

                bx = σ(tx) + cx
                by = σ(ty) + cy
                bw = pw * exp(tw)
                bh = ph * exp(th)
                Pr(object) ∗ IOU(b, object) = σ(to)

            Note:
                bx, by is the center of the bounding box
                bw, bh are factors of the w/h of the associated anchor
        """

        # Compute real values of tx, ty, tw, th from paper
        raw_aoff_pred = final[:, :, :, 0:4]

        # Anchor xy offset predictions of the CENTER of the bbox are relative
        # to the corner of the grid cell for which they were predicted in
        # output coordinates.
        center_xy_sig_pred = F.sigmoid(raw_aoff_pred[:, :, :, 0:2])

        # Anchor wh preedictions are multiplicitive factors of the
        # anchor width / height.
        wh_exp_pred = torch.exp(raw_aoff_pred[:, :, :, 2:4])

        # Stack xy addative and wh multiplicative offsets into aoff format
        aoff_pred = torch.cat([center_xy_sig_pred, wh_exp_pred], dim=3)

        # Predict IOU: sigma(to)
        iou_pred = F.sigmoid(final[:, :, :, 4:5])

        # TODO: do we do heirarchy stuff here?
        score_pred = final[:, :, :, 5:].contiguous()
        score_energy = score_pred.view(-1, score_pred.size()[-1])

        # Prediction conditional class probability P(Class | object)
        prob_pred = F.softmax(score_energy, dim=1).view_as(score_pred)

        output = (aoff_pred, iou_pred, prob_pred)
        return output

    def postprocess(self, output, inp_size, orig_sizes, conf_thresh=0.24,
                    nms_thresh=0.5, max_per_image=300):
        """
        Postprocess the raw network output into usable bounding boxes

        Args:
            aoff_pred (ndarray): [B, HxW, A, 4]
                anchor offsets in the format (sig(x), sig(y), exp(w), exp(h))
                note: in the aoff format x and y are centers of the box
                and wh represenets multiples of the anchor w/h

            iou_pred (ndarray): [B, HxW, A, 1]
                predicted iou (is this the objectness score?)

            prob_pred (ndarray): [B, HxW, A, C]
                predicted class probability

            inp_size (tuple): size (W, H) of input to network

            orig_sizes (list): [B, 2]
                size (W, H) of each in image before rescale

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
            python -m clab.models.yolo2.darknet Darknet19.postprocess --show

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
            >>> orig_sizes = torch.LongTensor([rgb255.shape[0:2][::-1]] * len(im_data))
            >>> conf_thresh = 0.01
            >>> nms_thresh = 0.5
            >>> postout = self.postprocess(output, inp_size, orig_sizes, conf_thresh, nms_thresh)
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
            >>> mplutil.show_if_requested()
        """
        aoff_pred_, iou_pred_, prob_pred_ = output

        # convert to numpy
        aoff_pred = aoff_pred_.data.cpu().numpy()
        iou_pred  = iou_pred_.data.cpu().numpy()
        prob_pred = prob_pred_.data.cpu().numpy()

        orig_sizes = orig_sizes.data.cpu().numpy()

        # num_classes, num_anchors = cfg.num_classes, cfg.num_anchors
        num_classes = self.num_classes
        anchors = self.anchors
        out_size = np.array(inp_size) // 32  # hacked we know the factor is 32
        W, H = out_size

        out_boxes = []
        out_scores = []
        out_cxs = []

        # For each image in the batch, postprocess the predicted boxes
        for bx in range(aoff_pred.shape[0]):
            aoffs = aoff_pred[bx][None, :]
            ious  = iou_pred[bx]
            probs = prob_pred[bx]
            orig_w, orig_h = orig_sizes[bx]

            # Convert anchored predictions to absolute tlbr bounding boxes in
            # normalized space
            aoffs = np.ascontiguousarray(aoffs, dtype=np.float)
            norm_boxes = yolo_utils.yolo_to_bbox(aoffs, anchors, H, W)[0]

            # Scale the bounding boxes to the size of the original image.
            # and convert to integer representation.
            boxes = norm_boxes.copy()
            boxes[..., 0::2] *= float(orig_w)
            boxes[..., 1::2] *= float(orig_h)
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
                # get predictions for each class
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
            boxes = yolo_utils.clip_boxes(boxes, im_shape=(orig_h, orig_w))

            # sort boxes by descending score
            sortx = scores.argsort()[::-1]
            boxes = boxes[sortx]
            scores = scores[sortx]
            cls_inds = cls_inds[sortx]

            if max_per_image > 0 and len(boxes) > max_per_image:
                boxes = boxes[:max_per_image]
                scores = scores[:max_per_image]
                cls_inds = cls_inds[:max_per_image]

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


def demo_batch(batch_size=1, inp_size=(100, 100), n_classes=20, rng=None,
               n=None):
    from clab import util
    rng = util.ensure_rng(rng)
    # number of groundtruth boxes per image in the batch
    if n is None:
        ntrues = [rng.randint(0, 10) for _ in range(batch_size)]
    else:
        ntrues = [n for _ in range(batch_size)]
    scale = min(inp_size) / 2.0
    gt_boxes = [yolo_utils.random_boxes(n, 'tlbr', scale=scale).reshape(-1, 4)
                for n in ntrues]
    gt_classes = [rng.randint(0, n_classes, n) for n in ntrues]
    gt_weights = [np.ones(n) for n in ntrues]
    orig_size = [inp_size for _ in range(batch_size)]
    indices = np.arange(batch_size)

    im_data = torch.randn(batch_size, 3, *inp_size)

    im_data = torch.FloatTensor(im_data)
    im_data = torch.FloatTensor(im_data)

    indices = torch.LongTensor([indices])
    orig_size = torch.LongTensor(orig_size)
    gt_weights = [torch.FloatTensor(item) for item in gt_weights]
    gt_classes = [torch.LongTensor(item) for item in gt_classes]

    inputs = [im_data]
    labels = [gt_boxes, gt_classes, orig_size, indices, gt_weights]
    return inputs, labels


def demo_predictions(B, W, H, A, rng=None):
    """ dummy predictions in the same format as the output layer """
    from clab import util
    rng = util.ensure_rng(rng)
    # Simulate the final layers
    final0123 = torch.FloatTensor(np.random.rand(B, H * W, A, 4))
    final4 = torch.FloatTensor(np.random.rand(B, H * W, A, 1))
    # raw_aoff_pred_ = final0123
    xy_sig_pred = F.sigmoid(final0123[..., 0:2])
    wh_exp_pred = torch.exp(final0123[..., 2:4])
    aoff_pred = torch.cat([xy_sig_pred, wh_exp_pred], dim=3)
    iou_pred = F.sigmoid(final4)
    return aoff_pred, iou_pred


def demo_weights():
    """
    Weights trained on VOC by yolo9000-pytorch
    """
    import os
    url = 'https://data.kitware.com/api/v1/item/5ab13b0e8d777f068578e251/download'
    dpath = ub.ensure_app_cache_dir('clab/yolo_v2')
    fname = 'yolo-voc.weights.pt'
    dest = os.path.join(dpath, fname)
    if not os.path.exists(dest):
        command = 'curl -X GET {} > {}'.format(url, dest)
        ub.cmd(command, verbout=1, shell=True)
    return dest


def initial_weights():
    """
    Weights pretrained trained ImageNet by yolo9000-pytorch
    """
    import os
    url = 'https://data.kitware.com/api/v1/file/5ab513438d777f068578f1d0/download'
    dpath = ub.ensure_app_cache_dir('clab/yolo_v2')
    fname = 'darknet19.weights.npz'
    dest = os.path.join(dpath, fname)
    if not os.path.exists(dest):
        command = 'curl -X GET {} > {}'.format(url, dest)
        ub.cmd(command, verbout=1, shell=True)

    # url = 'http://acidalia.kitware.com:8000/weights/darknet19.weights.npz'
    # npz_fpath = ub.grabdata(url, dpath=ub.ensure_app_cache_dir('clab'))

    # convert to torch weights
    npz_fpath = dest
    torch_fpath = ub.augpath(npz_fpath, ext='.pt')
    if not os.path.exists(torch_fpath):
        # hack to transform initial state
        model = Darknet19(num_classes=20)
        model.load_from_npz(npz_fpath, num_conv=18)
        torch.save(model.state_dict(), torch_fpath)

    # from clab import xpu_device
    # xpu = xpu_device.XPU('gpu')
    # xpu.load(torch_fpath)
    # torch.load(torch_fpath)
    return torch_fpath


def demo_image(inp_size):
    from clab import util
    import cv2
    rgb255 = util.grab_test_image('astro', 'rgb')
    rgb01 = cv2.resize(rgb255, inp_size).astype(np.float32) / 255
    im_data = torch.FloatTensor([rgb01.transpose(2, 0, 1)])
    return im_data, rgb255


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m clab.models.yolo2.darknet all
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
