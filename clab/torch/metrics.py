import torch  # NOQA
import pandas as pd
import numpy as np
import collections
import ubelt as ub
from .metrics import (confusion_matrix, jaccard_score_from_confusion,
                            pixel_accuracy_from_confusion,
                            perclass_accuracy_from_confusion)
import utool as ut
profile = ut.inject2(__name__)[-1]


def tpr(output, label, labels, ignore_label=-100):
    """
    true positive rate

    Example:
        >>> from clab.torch.sseg_train import *
        >>> from clab.torch.metrics import *
        >>> datasets = load_task_dataset(taskname='camvid')
        >>> train = datasets['train']
        >>> loader = torch.utils.data.DataLoader(train, batch_size=3)
        >>> inputs, label = map(torch.autograd.Variable, next(iter(loader)))
        >>> model = models.UNet(in_channels=train.n_channels, n_classes=train.n_classes)
        >>> output = model(inputs)
        >>> ignore_label = train.ignore_label
        >>> accuracy = tpr(output, label)
    """
    pred = output.data.max(dim=1)[1]
    true = label.data
    mask = (true != ignore_label) & (pred != ignore_label)
    is_tp = pred[mask] == true[mask]
    accuracy = is_tp.sum() / len(is_tp)
    return accuracy

    # pred == true
    # pred = output.data.max(dim=1)[1].cpu().numpy()
    # true = label.data.cpu().numpy()
    # is_tp = pred == true
    # tpr = is_tp.sum() / is_tp.size
    # return accuracy


@profile
def _sseg_metrics(output, label, labels, ignore_label=-100):
    """

    Ignore:
        >>> from clab.torch.sseg_train import *
        >>> from clab.torch.metrics import *
        >>> datasets = load_task_dataset(taskname='camvid')
        >>> train = datasets['train']
        >>> loader = torch.utils.data.DataLoader(train, batch_size=3)
        >>> inputs, label = map(torch.autograd.Variable, next(iter(loader)))
        >>> model = models.UNet(in_channels=train.n_channels, n_classes=train.n_classes)
        >>> output = model(inputs)
        >>> labels = train.task.labels
        >>> ignore_label = train.ignore_label
        >>> metrics = _sseg_metrics(output, label, labels, ignore_label)
    """
    pred = output.data.max(dim=1)[1]
    true = label.data
    mask = (true != ignore_label) & (pred != ignore_label)
    y_pred = pred[mask].cpu().numpy()
    y_true = true[mask].cpu().numpy()

    cfsn = confusion_matrix(y_pred, y_true, labels)
    cfsn = pd.DataFrame(cfsn, index=labels, columns=labels)
    cfsn = cfsn.drop(ignore_label, axis=0).drop(ignore_label, axis=1)

    ious = jaccard_score_from_confusion(cfsn)
    miou = ious.mean()

    # TODO: fix timetime warnings: Mean of empty slice, invalid value encoutered
    # in true_divide

    pixel_accuracy = pixel_accuracy_from_confusion(cfsn)  # same as tpr
    perclass_acc = perclass_accuracy_from_confusion(cfsn)
    perclass_acc = perclass_acc.fillna(0)
    class_accuracy = perclass_acc.mean()

    metrics_dict = ub.odict()
    metrics_dict['miou'] = miou
    metrics_dict['pixel_tpr'] = pixel_accuracy
    metrics_dict['class_tpr'] = class_accuracy
    # if len(perclass_acc) < 3:
    #     for k, acc in perclass_acc.to_dict().items():
    #         metrics_dict['class{}_tpr'.format(k)] = acc
    return metrics_dict


@profile
def _clf_metrics(output, label, labels, ignore_label=-100):
    """

    Ignore:
        >>> from clab.torch.sseg_train import *
        >>> from clab.torch.metrics import *
        >>> datasets = load_task_dataset(taskname='camvid')
        >>> train = datasets['train']
        >>> loader = torch.utils.data.DataLoader(train, batch_size=3)
        >>> inputs, label = map(torch.autograd.Variable, next(iter(loader)))
        >>> model = models.UNet(in_channels=train.n_channels, n_classes=train.n_classes)
        >>> output = model(inputs)
        >>> labels = train.task.labels
        >>> ignore_label = train.ignore_label
        >>> metrics = _sseg_metrics(output, label, labels, ignore_label)
    """
    pred = output.data.max(dim=1)[1]
    true = label.data
    mask = (true != ignore_label) & (pred != ignore_label)
    y_pred = pred[mask].cpu().numpy()
    y_true = true[mask].cpu().numpy()

    cfsn = confusion_matrix(y_pred, y_true, labels)
    cfsn = pd.DataFrame(cfsn, index=labels, columns=labels)

    if ignore_label >= 0:
        cfsn = cfsn.drop(ignore_label, axis=0)
        cfsn = cfsn.drop(ignore_label, axis=1)

    global_tpr = pixel_accuracy_from_confusion(cfsn)  # same as tpr
    perclass_acc = perclass_accuracy_from_confusion(cfsn)
    perclass_acc = perclass_acc.fillna(0)
    class_accuracy = perclass_acc.mean()

    metrics_dict = ub.odict()
    metrics_dict['global_tpr'] = global_tpr
    metrics_dict['class_tpr'] = class_accuracy
    return metrics_dict


def _siamese_metrics(output, label, margin=1):
    """

    Example:
        from .torch import models
        dim = 32
        model = models.SiameseLP(p=2, input_shape=(B, 3, dim, dim))
        input1 = Variable(torch.rand(B, 3, dim, dim))
        input2 = Variable(torch.rand(B, 3, dim, dim))
        outputs = model(input1, input2)
        label = Variable((torch.rand(B) + .5).long())
        margin = 1
        _siamese_metrics(output, label, margin)

        B = 42
        from torch.autograd import Variable
        output = Variable(torch.rand(B, 1)) * 2
        label = Variable((torch.rand(B) + .5).long())
        margin = 1
        _siamese_metrics(output, label, margin)

        output = Variable(torch.rand(B, 1)) * 2
        label = Variable((torch.rand(B) + 1).long())
        margin = 1
        _siamese_metrics(output, label, margin)
    """
    l2_dist_tensor = torch.squeeze(output.data.cpu())
    label_tensor = torch.squeeze(label.data.cpu())

    # Distance
    POS_LABEL = 1  # NOQA
    NEG_LABEL = 0  # NOQA
    is_pos = (label_tensor == POS_LABEL)

    pos_dists = l2_dist_tensor[is_pos]
    neg_dists = l2_dist_tensor[~is_pos]

    # Average positive / negative distances
    pos_dist = pos_dists.sum() / max(1, len(pos_dists))
    neg_dist = neg_dists.sum() / max(1, len(neg_dists))

    # accuracy
    pred_pos_flags = (l2_dist_tensor <= margin).long()

    # if NEG_LABEL == 0:
    pred = pred_pos_flags
    # else:
    #     pred = (pred_pos_flags * POS_LABEL) + (1 - pred_pos_flags) * NEG_LABEL

    n_correct = (pred == label_tensor).sum()
    fraction_correct = n_correct / len(label_tensor)
    # .size(0)

    # n_truepos = (pred_pos_flags == label_tensor.byte()).sum()
    # cur_score = torch.FloatTensor(label.size(0))
    # cur_score.fill_(NEG_LABEL)
    # cur_score[pred_pos_flags] = POS_LABEL
    # label_tensor_ = label_tensor.type(torch.FloatTensor)
    # accuracy = torch.eq(cur_score, label_tensor_).sum() / label_tensor.size(0)

    metrics = {
        'accuracy': fraction_correct,
        'pos_dist': pos_dist,
        'neg_dist': neg_dist,
    }
    return metrics


class MovingAve(ub.NiceRepr):

    def average(self):
        raise NotImplementedError()

    def update(self, other):
        raise NotImplementedError()

    def __nice__(self):
        return str(ub.repr2(self.average(), nl=0))


class CumMovingAve(MovingAve):
    """
    Cumulative moving average of dictionary values

    References:
        https://en.wikipedia.org/wiki/Moving_average

    Example:
        >>> from clab.torch.metrics import *
        >>> self = CumMovingAve()
        >>> str(self.update({'a': 10}))
        <CumMovingAve({'a': 10})>
        >>> str(self.update({'a': 0}))
        <CumMovingAve({'a': 5.0})>
        >>> str(self.update({'a': 2}))
        <CumMovingAve({'a': 4.0})>
    """
    def __init__(self):
        self.totals = ub.odict()
        self.n = 0

    def average(self):
        return {k: v / self.n for k, v in self.totals.items()}

    def update(self, other):
        self.n += 1
        for k, v in other.items():
            if np.isnan(v):
                v = 0
            if k not in self.totals:
                self.totals[k] = 0
            self.totals[k] += v
        return self


class WindowedMovingAve(MovingAve):
    """
    Windowed moving average of dictionary values

    Args:
        window (int): number of previous observations to consider

    Example:
        >>> from clab.torch.metrics import *
        >>> self = WindowedMovingAve(window=3)
        >>> str(self.update({'a': 10}))
        <CumMovingAve({'a': 10})>
        >>> str(self.update({'a': 0}))
        <CumMovingAve({'a': 5.0})>
        >>> str(self.update({'a': 2}))
        <CumMovingAve({'a': 1.0})>
    """
    def __init__(self, window=500):
        self.window = window
        self.totals = ub.odict()
        self.history = {}

    def average(self):
        return {k: v / len(self.history[k]) for k, v in self.totals.items()}

    def update(self, other):
        for k, v in other.items():
            if np.isnan(v):
                v = 0
            if k not in self.totals:
                self.history[k] = collections.deque()
                self.totals[k] = 0
            self.totals[k] += v
            self.history[k].append(v)
            if len(self.history[k]) == self.window:
                # Push out the oldest value
                self.totals[k] -= self.history[k].popleft()
        return self


class ExpMovingAve(MovingAve):
    """
    Exponentially weighted moving average of dictionary values

    Args:
        span (float): roughly corresponds to window size.
            http://greenteapress.com/thinkstats2/html/thinkstats2013.html

    Example:
        >>> from clab.torch.metrics import *
        >>> self = ExpMovingAve(span=3)
        >>> str(self.update({'a': 10}))
        <ExpMovingAve({'a': 10})>
        >>> str(self.update({'a': 0}))
        <ExpMovingAve({'a': 5.0})>
        >>> str(self.update({'a': 2}))
        <ExpMovingAve({'a': 3.5})>
    """
    def __init__(self, span=500):
        values = ub.odict()
        self.values = values
        self.alpha = 2 / (span + 1)

    def average(self):
        return self.values

    def update(self, other):
        alpha = self.alpha
        for k, v in other.items():
            if np.isnan(v):
                v = 0
            if k not in self.values:
                self.values[k] = v
            else:
                self.values[k] = (alpha * v) + (1 - alpha) * self.values[k]
        return self
