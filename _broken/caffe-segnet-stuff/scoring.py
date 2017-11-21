# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import numpy as np
import pandas as pd
import ubelt as ub
import cv2
import six
from pysseg import metrics
from pysseg.util import fnameutil

from pysseg import getLogger
logger = getLogger(__name__)
print = logger.info


def ensureiterable(v):
    if isinstance(v, six.string_types):
        v = [v]
    return v


def rectify_confusions(raw_cfsn, to_true_alias_classnames,
                       pred_ignore_classnames, true_ignore_classnames,
                       true_ignore_classname):
    """
    Combine and consolidate rows and columns so the row and column index are
    the same. (probably should fixup the args to this func a bit)
    """
    total_cfsn = raw_cfsn.copy()
    # Convert pred to true labels
    # CONVERT PRED TO TRUE CLASSNAMES
    to_true_alias_classnames[tuple(pred_ignore_classnames)] = true_ignore_classname
    for pred_names, true_name in to_true_alias_classnames.items():
        pred_names = ensureiterable(pred_names)
        new_col = total_cfsn.loc[:, pred_names].sum(axis=1)
        total_cfsn.drop(list(pred_names), axis=1, inplace=True)
        if true_name in total_cfsn.columns:
            total_cfsn.loc[:, true_name] += new_col
        else:
            total_cfsn.loc[:, true_name] = new_col

    # consolodate all classes we cant handle into the null class
    true_ignore_classnames_ = true_ignore_classnames - {true_ignore_classname}
    if true_ignore_classnames_:
        new_col = total_cfsn.loc[true_ignore_classnames_, :].sum(axis=0)
        total_cfsn.drop(true_ignore_classnames_, axis=0, inplace=True)
        total_cfsn.loc[true_ignore_classname, :] += new_col
    return total_cfsn


def build_confusion(gt_paths, pred_paths, true_classnames, pred_classnames,
                    true_labels, pred_labels):
    im_size = cv2.imread(pred_paths[0], cv2.IMREAD_UNCHANGED).size
    sample_weight = np.ones(im_size, dtype=np.int64)

    raw_cfsn = pd.DataFrame(0, index=pd.Index(true_classnames, name='real'),
                              columns=pd.Index(pred_classnames, name='pred'))
    prog = ub.ProgIter(zip(pred_paths, gt_paths), length=len(pred_paths),
                       verbose=1, freq=1)
    for pred_path, gt_path in prog:
        y_pred = cv2.imread(pred_path, flags=cv2.IMREAD_UNCHANGED).ravel()
        y_true = cv2.imread(gt_path, flags=cv2.IMREAD_UNCHANGED).ravel()
        cfsn = metrics.confusion_matrix2(y_true, y_pred, true_labels,
                                         pred_labels,
                                         sample_weight=sample_weight)
        raw_cfsn += cfsn
    return raw_cfsn


def score_predictions(gt_paths, pred_paths, true_task=None, pred_task=None):
    """
    Logic for scoring a list of predictions against a list of groundtruth semantic
    segs, from the same or similar tasks.
    """
    # Ensure the predictions and groundtruth images are aligned
    pred_paths = fnameutil.align_paths(gt_paths, pred_paths)

    pred_labels = pred_task.labels
    pred_classnames = pred_task.classnames

    if true_task is not None:
        true_labels = true_task.labels
        true_classnames = true_task.classnames
        ignore_classnames = true_task.ignore_classnames
        true_ignore_classname = true_task.null_classname

        from_other = true_task.convert_classnames_from(pred_task.name)
        to_true_alias_classnames = from_other
        pred_ignore_classnames = (set(pred_task.classnames) -
                                  set(to_true_alias_classnames.keys()))
        true_ignore_classnames = (set(true_task.classnames) -
                                  set(to_true_alias_classnames.values()))
    else:
        true_labels = pred_labels
        true_classnames = pred_classnames
        ignore_classnames = pred_task.ignore_classnames

    raw_cfsn = build_confusion(gt_paths, pred_paths, true_classnames,
                               pred_classnames, true_labels, pred_labels)

    if true_task is not None:
        # If the prediction and target tasks are different we need to combine
        # some labels
        total_cfsn = rectify_confusions(raw_cfsn, to_true_alias_classnames,
                                        pred_ignore_classnames,
                                        true_ignore_classnames,
                                        true_ignore_classname)
    else:
        total_cfsn = raw_cfsn.copy()

    # Ensure rows and cols have the same order
    total_cfsn = total_cfsn.sort_index(axis=0).sort_index(axis=1)
    assert np.all(total_cfsn.index == total_cfsn.columns)

    ious = metrics.jaccard_score_from_confusion(total_cfsn)
    miou = ious.drop(ignore_classnames).mean(skipna=True)

    cfsn2 = total_cfsn.drop(ignore_classnames, axis=0).drop(ignore_classnames, axis=1)
    global_acc = metrics.pixel_accuracy_from_confusion(cfsn2)
    class_acc = metrics.class_accuracy_from_confusion(cfsn2)

    results = {
        'raw_cfsn': raw_cfsn.to_dict(),
        'total_cfsn': total_cfsn.to_dict(),
        'ious': ious.to_dict(),
        'global_miou': miou,
        'global_acc': global_acc,
        'class_acc': class_acc,
        'ignore_classnames': ignore_classnames,
    }
    return results
