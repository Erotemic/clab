# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
from collections import OrderedDict
import ubelt as ub
import numpy as np
from clab.util import colorutil
from clab.util import imutil


class SemanticSegmentationTask(object):
    """
    Doctest:
        >>> from clab.tasks import *
        >>> classnames = ['spam',  'NULL', 'eggs', 'ham', 'jam','bannana']
        >>> null_classname = 'NULL'
        >>> alias = {'jam': 'spam', 'ham': 'spam', 'bannana': 'NULL'}
        >>> task = SemanticSegmentationTask(classnames, null_classname, alias)
        >>> assert task.classnames == ['spam', 'NULL', 'eggs']
        >>> assert np.all(task.labels == [0, 1, 2])
        >>> assert np.all(task.relevant_labels == [0, 2])
        >>> assert task.ignore_label == 1
        >>> harn.assert_self()
    """
    def __init__(task, classnames=None, null_classname=None, alias={}):
        if classnames is not None:
            task.set_classnames(classnames, null_classname, alias)

        task.extern_train_im_paths = []
        task.extern_train_gt_paths = []

    @property
    def name(self):
        return self.__class__.__name__

    def set_classnames(task, classnames, null_classname, alias={}):

        task.classnames = classnames
        task.classname_alias = alias
        task.null_classname = null_classname

        # Remove aliased classes
        for k in alias.keys():
            if k in task.classnames:
                task.classnames.remove(k)

        # Assign an integer label to each classname
        task.classname_to_id = {
            classname: id_
            for id_, classname in enumerate(task.classnames, start=0)
        }
        # Map aliased classes to a different label
        for k, v in alias.items():
            task.classname_to_id[k] = task.classname_to_id[v]

        task.ignore_label = task.classname_to_id[task.null_classname]
        task.ignore_classnames = [task.null_classname]
        task.ignore_labels = np.array([task.ignore_label])

        task.labels = np.arange(len(task.classnames))
        task.relevant_labels = np.setdiff1d(task.labels, task.ignore_labels)
        # NOTE: WE ASSUME COLORS ARE IN BGR
        distinct_colors = colorutil.make_distinct_bgr01_colors(len(task.classnames))
        task.class_colors = OrderedDict(zip(task.classnames, distinct_colors))

        if hasattr(task, 'customize_colors'):
            task.customize_colors()
        # lookup_bgr255 = colorutil.lookup_bgr255
        # task.class_colors[task.null_classname] = lookup_bgr255('black')

    def assert_self(task):
        n_classes = len(task.classnames)
        assert max(task.classname_to_id.values()) == n_classes - 1
        assert np.all(task.ignore_labels == [task.ignore_label])
        assert np.all(task.ignore_classnames == [task.null_classname])
        # assert len(task.class_colors) == n_classes

    def colorize(task, y_img, x_img=None, alpha=.6):
        labels = list(ub.take(task.classname_to_id, task.classnames))
        assert np.all(labels == task.labels), (
            'classname order assumption is invalid')
        label_to_color = np.array(list(ub.take(task.class_colors,
                                               task.classnames)))
        label_to_color = label_to_color.astype(np.uint8)
        # label_to_color = pd.Series(
        #     list(ub.take(task.class_colors, task.classnames)),
        #     index=list(ub.take(task.classname_to_id, task.classnames)),
        # )
        color_img = label_to_color[y_img.astype(np.int)]
        if x_img is not None:
            # blend the color image on top of the data
            blend = imutil.overlay_colorized(color_img, x_img, alpha=alpha)
            return blend
        else:
            return color_img

    def instance_colorize(task, y_img, x_img=None, seed=None):
        n = y_img.max() + 1
        distinct_colors = colorutil.make_distinct_bgr01_colors(n)

        import random
        rng = random.Random(seed)
        rng.shuffle(distinct_colors)

        distinct_colors[0] = [0, 0, 0]

        label_to_color = np.array(distinct_colors)
        label_to_color = label_to_color.astype(np.uint8)

        color_img = label_to_color[y_img.astype(np.int)]
        if x_img is not None:
            # blend the color image on top of the data
            blend = imutil.overlay_colorized(color_img, x_img)
            return blend
        else:
            return color_img

    def exptdir(task, *args):
        """
        use for making directories where we will run experiments
        """
        return ub.ensuredir((task.workdir, 'harness') + args)
