"""
Some functions adapated from https://github.com/martinkersner/py_img_seg_eval

There are n_cl different classes.

Let n_ij be the number of pixels of class i predicted to belong to class j

let t_i = sum_j(n_ij) be the total number of pixels of class i.
"""
import numpy as np
from scipy.sparse import coo_matrix
import torch  # NOQA
import pandas as pd
# import numpy as np
import collections
import ubelt as ub


def confusion_matrix(y_true, y_pred, n_labels=None, labels=None, sample_weight=None):
    """
    faster version of sklearn confusion matrix that avoids the
    expensive checks and label rectification

    Runs in about 0.7ms

    Returns:
        ndarray: matrix where rows represent real and cols represent pred

    Example:
        >>> y_true = np.array([0, 0, 0, 0, 1, 1, 1, 0,  0, 1])
        >>> y_pred = np.array([0, 0, 0, 0, 0, 0, 0, 1,  1, 1])
        >>> confusion_matrix(y_true, y_pred, 2)
        array([[4, 2],
               [3, 1]])
        >>> confusion_matrix(y_true, y_pred, 2).ravel()
        array([4, 2, 3, 1])
    """
    if sample_weight is None:
        sample_weight = [1] * len(y_true)
    if n_labels is None:
        n_labels = len(labels)
    CM = coo_matrix((sample_weight, (y_true, y_pred)),
                    shape=(n_labels, n_labels),
                    dtype=np.int64).toarray()
    return CM


def confusion_matrix2(y_true, y_pred, true_labels=None, pred_labels=None,
                      sample_weight=None):
    """
    faster version of sklearn confusion matrix that avoids the
    expensive checks and label rectification

    Runs in about 0.7ms
    """
    n_true = len(true_labels)
    n_pred = len(pred_labels)
    cfsn = coo_matrix((sample_weight, (y_true, y_pred)),
                      shape=(n_true, n_pred),
                      dtype=np.int64).toarray()
    return cfsn


def jaccard_score_from_confusion(cfsn):
    """ Calculate IoU for each class (jaccard score) """
    tp = np.diag(cfsn)
    # real is rows, pred is columns
    fp = cfsn.sum(axis=0) - tp
    fn = cfsn.sum(axis=1) - tp
    denom = (tp + fp + fn)
    ious = tp / (denom + (denom == 0) * 1e-9)
    ious[denom == 0] = 0
    return ious


def pixel_accuracy_from_confusion(cfsn):
    # real is rows, pred is columns
    n_ii = np.diag(cfsn)
    # sum over pred = columns = axis1
    t_i = cfsn.sum(axis=1)
    global_acc = n_ii.sum() / t_i.sum()
    return global_acc


def class_accuracy_from_confusion(cfsn):
    # real is rows, pred is columns
    n_ii = np.diag(cfsn)
    # sum over pred = columns = axis1
    t_i = cfsn.sum(axis=1)
    class_acc = (n_ii / t_i).mean()
    return class_acc


def perclass_accuracy_from_confusion(cfsn):
    # real is rows, pred is columns
    n_ii = np.diag(cfsn)
    # sum over pred = columns = axis1
    t_i = cfsn.sum(axis=1)
    perclass_acc = (n_ii / t_i)
    return perclass_acc


def pixel_accuracy(y_true, y_pred, labels=None, ignore_label=None):
    """
    (Global accuracy)

    sum_i(n_ii) / sum_i(t_i)
    """
    check_size(y_true, y_pred)

    n_cl = len(labels)
    pred_mask, true_mask = extract_both_masks(y_true, y_pred, labels, n_cl)

    sum_n_ii = 0
    sum_t_i  = 0

    for i, c in enumerate(labels):
        curr_eval_mask = pred_mask[i, :, :]
        curr_gt_mask = true_mask[i, :, :]

        sum_n_ii += np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
        sum_t_i  += np.sum(curr_gt_mask)

    if (sum_t_i == 0):
        pixel_accuracy_ = 0
    else:
        pixel_accuracy_ = sum_n_ii / sum_t_i

    return pixel_accuracy_


def mean_accuracy(y_true, y_pred):
    """
    (1/n_cl) sum_i(n_ii/t_i)
    """

    check_size(y_true, y_pred)

    labels, n_cl = extract_classes(y_true)
    pred_mask, true_mask = extract_both_masks(y_true, y_pred, labels, n_cl)

    accuracy = list([0]) * n_cl

    for i, c in enumerate(labels):
        curr_eval_mask = pred_mask[i, :, :]
        curr_gt_mask = true_mask[i, :, :]

        n_ii = np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
        t_i  = np.sum(curr_gt_mask)

        if (t_i != 0):
            accuracy[i] = n_ii / t_i

    mean_accuracy_ = np.mean(accuracy)
    return mean_accuracy_


def preclass_confusions(y_true, y_pred, labels=None, ignore_label=None):
    """
    (1/n_cl) * sum_i(n_ii / (t_i + sum_j(n_ji) - n_ii))
    """
    check_size(y_true, y_pred)
    # n_ignore = int(ignore_label is not None)

    n_cl = max(labels) + 1
    pred_mask, true_mask = extract_both_masks(y_true, y_pred, labels, n_cl)
    # IU = list([0]) * n_cl

    label_to_confusion = {}
    for i in range(n_cl):
        curr_eval_mask = pred_mask[i, :, :]
        curr_gt_mask = true_mask[i, :, :]

        if (np.sum(curr_eval_mask) == 0) or (np.sum(curr_gt_mask) == 0):
            continue

        n_ii = np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
        t_i  = np.sum(curr_gt_mask)
        n_ij = np.sum(curr_eval_mask)
        n_ii / (t_i + n_ij - n_ii)

        # Alternative confusion matrix formulation
        tp = n_ii
        fp = n_ij - n_ii
        fn = t_i - n_ii
        tp / (tp + fp + fn)

        tmp = {'tp': tp, 'fn': fn, 'fp': fp}
        label_to_confusion[i] = tmp
    return label_to_confusion
    # labels, n_cl = extract_classes(y_true)
    # _, n_cl_gt = extract_classes(y_true)
    # mean_IoU_ = np.sum(IU) / n_cl_gt
    # return mean_IoU_


def mean_IoU(y_true, y_pred, ignore_label=None):
    """
    (1/n_cl) * sum_i(n_ii / (t_i + sum_j(n_ji) - n_ii))
    """

    check_size(y_true, y_pred)

    cl, n_cl   = union_classes(y_true, y_pred, ignore_label=ignore_label)
    _, n_cl_gt = extract_classes(y_true, ignore_label=ignore_label)
    eval_mask, gt_mask = extract_both_masks(y_true, y_pred, cl, n_cl)

    IU = list([0]) * n_cl

    for i, c in enumerate(cl):
        curr_eval_mask = eval_mask[i, :, :]
        curr_gt_mask = gt_mask[i, :, :]

        if (np.sum(curr_eval_mask) == 0) or (np.sum(curr_gt_mask) == 0):
            continue

        n_ii = np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
        t_i  = np.sum(curr_gt_mask)
        n_ij = np.sum(curr_eval_mask)

        IU[i] = n_ii / (t_i + n_ij - n_ii)

    mean_IU_ = np.sum(IU) / n_cl_gt
    return mean_IU_


def frequency_weighted_IoU(y_true, y_pred):
    """
    sum_k(t_k)^(-1) * sum_i((t_i*n_ii)/(t_i + sum_j(n_ji) - n_ii))
    """

    check_size(y_true, y_pred)

    labels, n_cl = union_classes(y_true, y_pred)
    pred_mask, true_mask = extract_both_masks(y_true, y_pred, labels, n_cl)

    frequency_weighted_IU_ = list([0]) * n_cl

    for i, c in enumerate(labels):
        curr_eval_mask = pred_mask[i, :, :]
        curr_gt_mask = true_mask[i, :, :]

        if (np.sum(curr_eval_mask) == 0) or (np.sum(curr_gt_mask) == 0):
            continue

        n_ii = np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
        t_i  = np.sum(curr_gt_mask)
        n_ij = np.sum(curr_eval_mask)

        frequency_weighted_IU_[i] = (t_i * n_ii) / (t_i + n_ij - n_ii)

    sum_k_t_k = get_pixel_area(y_pred)

    frequency_weighted_IU_ = np.sum(frequency_weighted_IU_) / sum_k_t_k
    return frequency_weighted_IU_


# Auxiliary functions used during evaluation.
def get_pixel_area(segm):
    return segm.shape[0] * segm.shape[1]


def extract_both_masks(y_true, y_pred, labels, n_cl):
    pred_mask = extract_masks(y_pred, labels, n_cl)
    true_mask   = extract_masks(y_true, labels, n_cl)

    return pred_mask, true_mask


def extract_classes(segm, ignore_label=None):
    labels = np.unique(segm)
    n_cl = len(labels)
    if ignore_label:
        if ignore_label in labels:
            n_cl -= 1
            labels = np.setdiff1d(labels, [ignore_label])
    return labels, n_cl


def union_classes(y_true, y_pred, ignore_label=None):
    eval_cl, _ = extract_classes(y_pred, ignore_label=ignore_label)
    gt_cl, _   = extract_classes(y_true, ignore_label=ignore_label)

    labels = np.union1d(eval_cl, gt_cl)
    n_cl = len(labels)

    return labels, n_cl


def extract_masks(segm, labels, n_cl):
    h, w  = segm_size(segm)
    masks = np.zeros((n_cl, h, w), dtype=np.int)

    for i, c in enumerate(labels):
        masks[i, :, :] = segm == c

    return masks


def segm_size(segm):
    try:
        height = segm.shape[0]
        width  = segm.shape[1]
    except IndexError:
        raise

    return height, width


def check_size(y_true, y_pred):
    if segm_size(y_pred) != segm_size(y_true):
        raise EvalSegErr("DiffDim: Different dimensions of matrices!")


# Exceptions
class EvalSegErr(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


def tpr(output, label, labels, ignore_label=-100):
    """
    true positive rate

    Example:
        >>> from clab.sseg_train import *
        >>> from clab.metrics import *
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


def _sseg_metrics(output, label, labels, ignore_label=-100):
    """

    Ignore:
        >>> from clab.sseg_train import *
        >>> from clab.metrics import *
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


def _clf_metrics(output, label, all_labels, ignore_label=-100):
    """

    Ignore:
        >>> from clab.sseg_train import *
        >>> from clab.metrics import *
        >>> datasets = load_task_dataset(taskname='camvid')
        >>> train = datasets['train']
        >>> loader = torch.utils.data.DataLoader(train, batch_size=3)
        >>> inputs, label = map(torch.autograd.Variable, next(iter(loader)))
        >>> model = models.UNet(in_channels=train.n_channels, n_classes=train.n_classes)
        >>> output = model(inputs)
        >>> all_labels = train.task.labels
        >>> ignore_label = train.ignore_label
        >>> metrics = _sseg_metrics(output, label, all_labels, ignore_label)
    """
    pred = output.data.max(dim=1)[1]
    true = label.data
    mask = (true != ignore_label) & (pred != ignore_label)
    y_pred = pred[mask].cpu().numpy()
    y_true = true[mask].cpu().numpy()

    cfsn = confusion_matrix(y_pred, y_true, labels=all_labels)
    cfsn = pd.DataFrame(cfsn, index=all_labels, columns=all_labels)

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
        >>> from clab.metrics import *
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
            if pd.isnull(v):
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
        >>> from clab.metrics import *
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
            if pd.isnull(v):
                v = 0
            if k not in self.totals:
                self.history[k] = collections.deque()
                self.totals[k] = 0
            self.totals[k] += v
            self.history[k].append(v)
            if len(self.history[k]) > self.window:
                # Push out the oldest value
                self.totals[k] -= self.history[k].popleft()
        return self


class ExpMovingAve(MovingAve):
    """
    Exponentially weighted moving average of dictionary values

    Args:
        span (float): roughly corresponds to window size.
            equivalent to (2 / alpha) - 1
        alpha (float): roughly corresponds to window size.
            equivalent to 2 / (span + 1)

    References:
        http://greenteapress.com/thinkstats2/html/thinkstats2013.html

    Example:
        >>> from clab.metrics import *
        >>> self = ExpMovingAve(span=3)
        >>> str(self.update({'a': 10}))
        <ExpMovingAve({'a': 10})>
        >>> str(self.update({'a': 0}))
        <ExpMovingAve({'a': 5.0})>
        >>> str(self.update({'a': 2}))
        <ExpMovingAve({'a': 3.5})>
    """
    def __init__(self, span=None, alpha=None):
        values = ub.odict()
        self.values = values
        if not bool(span is None) ^ bool(alpha is None):
            raise ValueError('specify either alpha xor span')

        if alpha is not None:
            self.alpha = alpha
        elif span is not None:
            self.alpha = 2 / (span + 1)
        else:
            raise AssertionError('impossible state')

    def average(self):
        return self.values

    def update(self, other):
        alpha = self.alpha
        for k, v in other.items():
            if pd.isnull(v):
                v = 0
            if k not in self.values:
                self.values[k] = v
            else:
                self.values[k] = (alpha * v) + (1 - alpha) * self.values[k]
        return self
