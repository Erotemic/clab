"""
Segmentation metrics

Some functions adapated from https://github.com/martinkersner/py_img_seg_eval
by Martin Kersner, m.kersner@gmail.com (2015/11/30)

Note the Martin Kerner functions only work on a per-image basis, therefore we
arent using them. They are there for reference.

Evaluation metrics for image segmentation inspired by
paper Fully Convolutional Networks for Semantic Segmentation.


There are n_cl different classes.

Let n_ij be the number of pixels of class i predicted to belong to class j

let t_i = sum_j(n_ij) be the total number of pixels of class i.
"""
import numpy as np
from scipy.sparse import coo_matrix


def confusion_matrix(y_true, y_pred, labels=None, sample_weight=None):
    """
    faster version of sklearn confusion matrix that avoids the
    expensive checks and label rectification

    Runs in about 0.7ms
    """
    if sample_weight is None:
        sample_weight = [1] * len(y_true)
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
