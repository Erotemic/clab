import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable  # NOQA

# from clab import getLogger
# logger = getLogger(__name__)
# print = logger.info


def testdata_siam_desc(num_data=128, desc_dim=8):
    import numpy as np
    import vtool as vt
    rng = np.random.RandomState(0)
    network_output = vt.normalize_rows(rng.rand(num_data, desc_dim))
    vecs1 = network_output[0::2]
    vecs2 = network_output[1::2]
    # roll vecs2 so it is essentially translated
    vecs2 = np.roll(vecs1, 1, axis=1)
    network_output[1::2] = vecs2
    # Every other pair is an imposter match
    network_output[::4, :] = vt.normalize_rows(rng.rand(32, desc_dim))
    #data_per_label = 2

    vecs1 = network_output[0::2].astype(np.float32)
    vecs2 = network_output[1::2].astype(np.float32)

    def true_dist_metric(vecs1, vecs2):
        g1_ = np.roll(vecs1, 1, axis=1)
        dist = vt.L2(g1_, vecs2)
        return dist
    #l2dist = vt.L2(vecs1, vecs2)
    true_dist = true_dist_metric(vecs1, vecs2)
    label = (true_dist > 0).astype(np.float32)
    vecs1 = torch.from_numpy(vecs1)
    vecs2 = torch.from_numpy(vecs2)
    label = torch.from_numpy(label)
    return vecs1, vecs2, label


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss function.

    References:
        https://github.com/delijati/pytorch-siamese/blob/master/contrastive.py

    LaTeX:
        Let D be the distance computed between the network layers or the direct
        distance output of the network.

        y is 0 if the pair should be labled as an imposter
        y is 1 if the pair should be labled as genuine

        ContrastiveLoss = ((y * D) ** 2 + ((1 - y) * max(m - D, 0) ** 2)) / 2

        $(y E)^2 + ((1 - y) max(m - E, 0)^2)$

    CommandLine:
        python -m clab.torch.criterions ContrastiveLoss --show


    Example:
        >>> from clab.torch.criterions import *
        >>> import utool as ut
        >>> import numpy as np
        >>> vecs1, vecs2, label = testdata_siam_desc()
        >>> output = torch.nn.PairwiseDistance(p=2)(vecs1, vecs2)
        >>> self = ContrastiveLoss()
        >>> ut.exec_func_src(self.forward, globals())
        >>> func = self.forward
        >>> loss2x, dist = ut.exec_func_src(self.forward, globals(), globals(), keys=['loss2x', 'dist'])
        >>> ut.quit_if_noshow()
        >>> loss2x, dist, label = map(np.array, [loss2x, dist, label])
        >>> label = label.astype(np.bool)
        >>> dist0_l2 = dist[~label]
        >>> dist1_l2 = dist[label]
        >>> loss0 = loss2x[~label] / 2
        >>> loss1 = loss2x[label] / 2
        >>> import plottool as pt
        >>> pt.plot2(dist0_l2, loss0, 'x', color=pt.FALSE_RED, label='imposter_loss', y_label='loss')
        >>> pt.plot2(dist1_l2, loss1, 'x', color=pt.TRUE_BLUE, label='genuine_loss', y_label='loss')
        >>> pt.gca().set_xlabel('l2-dist')
        >>> pt.legend()
        >>> ut.show_if_requested()

    Example:
        >>> from clab.torch.models import SiameseLP
        >>> from torch.autograd import Variable  # NOQA
        >>> imgs1 = Variable(torch.rand(3, 3, 224, 244))
        >>> imgs2 = Variable(torch.rand(3, 3, 224, 244))
        >>> label = (Variable(torch.rand(3)) * 2).long()

        >>> model = SiameseLP()
        >>> output = model(imgs1, imgs2)
        >>> self = ContrastiveLoss(margin=10)
        >>> self.forward(output, label)
    """

    def __init__(self, weight=None, margin=1.0):
        # ut.super2(ContrastiveLoss, self).__init__()
        super(ContrastiveLoss, self).__init__()
        self.weight = weight
        self.margin = margin

        self.neg_label = 0
        self.pos_label = 1

    def forward(self, output, label):
        # Output should be a Bx1 vector representing the predicted
        # distance between each pair in a patch of image pairs
        dist = torch.squeeze(output)

        # Build indicator vectors labeling which pairs are pos and neg
        is_genuine = label.float()
        is_imposter = (1 - is_genuine)

        # Negate and clamp the distance for imposter pairs so these pairs are
        # encouraged to predict a distance larger than the margin
        hinge_neg_dist = torch.clamp(self.margin - dist, min=0.0)
        # Apply this loss only to negative examples
        loss_imposter = is_imposter * torch.pow(hinge_neg_dist, 2)

        # THe loss for positive examples is simply the distance because we wish
        # to encourage the network to predict values close to zero
        loss_genuine = is_genuine * torch.pow(dist, 2)

        # Weight each class if desired
        if self.weight is not None:
            loss_imposter *= self.weight[0]
            loss_genuine *= self.weight[1]

        # Sum the loss together (actually there is a 2x factor here)
        loss2x = loss_genuine + loss_imposter

        # Divide by 2 after summing for efficiency
        ave_loss = torch.sum(loss2x) / 2.0 / label.size()[0]
        loss = ave_loss
        return loss


def testdata_sseg():
    import numpy as np
    rng = np.random.RandomState(0)
    batch_size = 4
    n_classes = 12
    width = 360
    height = 480
    output_np = rng.rand(batch_size, n_classes, width, height).astype(np.float32)
    label_np = (rng.rand(batch_size, width, height) * n_classes).astype(np.int64)
    output = Variable(torch.from_numpy(output_np))
    label = Variable(torch.from_numpy(label_np))
    return output, label


class CrossEntropyLoss2D(nn.Module):
    """
    https://github.com/ycszen/pytorch-seg/blob/master/loss.py
    https://discuss.pytorch.org/t/about-segmentation-loss-function/2906/8

    Example:
        >>> from clab.torch.criterions import *
        >>> #inputs, targets = testdata_sseg()
        >>> weight = Variable(torch.FloatTensor([1, 1, 0]))
        >>> size_average = True
        >>> inputs = Variable(torch.FloatTensor([[
        >>>     # class 0 (building)
        >>>     [[0.1, 0.1, 0.1],
        >>>      [0.6, 0.5, 0.1],
        >>>      [0.1, 0.1, 0.1],],
        >>>     # class 1 (non-building)
        >>>     [[0.1, 0.2, 0.1],
        >>>      [0.5, 0.4, 0.5],
        >>>      [0.3, 0.1, 0.4],],
        >>>     # class 2 (ignore)
        >>>     [[0.3, 0.1, 0.1],
        >>>      [0.1, 0.2, 0.1],
        >>>      [0.1, 0.1, 0.1],],
        >>> ]]))
        >>> targets = Variable(torch.LongTensor([[
        >>>     [2, 2, 2],
        >>>     [0, 1, 0],
        >>>     [0, 1, 0],
        >>> ]]))
        >>> self = CrossEntropyLoss2D(weight=weight, ignore_label=2)
        >>> probs = torch.exp(F.log_softmax(inputs, dim=1))
        >>> loss = self.forward(inputs, targets)
        >>> print('loss = {!r}'.format(loss))
    """
    def __init__(self, weight=None, size_average=True, ignore_label=-100):
        super(CrossEntropyLoss2D, self).__init__()
        self.nll_loss = torch.nn.NLLLoss2d(weight, size_average,
                                           ignore_index=ignore_label)

    def forward(self, inputs, targets):
        """
        Softmax followed by negative-log-likelihood loss (multiclass cross
        entropy)

        Note: softmax is only needed to put value in the proper range for cross
        entropy. Softmax preserves ordering, so if you are doing prediction,
        you don't need softmax because x.argsort() == softmax(x).argsort
        """
        return self.nll_loss(F.log_softmax(inputs, dim=1), targets)


# def cross_entropy2d(inputs, targets, weight=None, size_average=True):
#     """
#     https://github.com/ycszen/pytorch-seg/blob/master/loss.py
#     """
#     n, c, h, w = inputs.size()
#     inputs = inputs.transpose(1, 2).transpose(2, 3).contiguous()
#     inputs = inputs[targets.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0].view(-1, c)

#     target_mask = targets >= 0
#     targets = targets[target_mask]
#     #loss = F.nll_loss(F.log_softmax(input), targets, weight=weight, size_average=False)
#     loss = F.cross_entropy(input, targets, weight=weight, size_average=False)
#     if size_average:
#         loss /= target_mask.sum().data[0]
#     return loss

    # log_p = F.log_softmax(output, dim=1)
    # log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous()

    # # TODO: ignore any negative label
    # # for ignore in ignore_labels:
    # #     label[label == ignore] = -1

    # # Flatten Predictions
    # log_p = log_p[label.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0].view(-1, c)

    # # Flatten Labels
    # target_mask = label >= 0
    # target = label[target_mask]

    # # from clab import metrics
    # # confusion_matrix()
    # # loss = torch.nn.functional.nll_loss(log_p, target, weight=weight, size_average=False)
    # loss = F.cross_entropy(log_p, target, weight=weight, size_average=False)
    # if size_average:
        # loss /= target_mask.data.sum()
    # return loss


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m clab.torch.criterions
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
