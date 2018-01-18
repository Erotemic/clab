from __future__ import division, print_function
import cv2
import numpy as np
from clab.util import imutil
from clab import util


class SSegAugmentor(object):
    """
    Basic data augmentation for semantic segmantation
    (used in caffe version)
    """
    def __init__(auger, ignore_label, rng=None):
        auger.params = {
            'axis_flips': [0, 1],
            'gammas':  [.5, 2.5],
            'blur':  [5, 20],
            'n_rand_zoom':  2,
            'n_rand_occl': 2,
        }
        auger.rng = rng
        assert ignore_label is not None, 'specify an ignore label'
        auger.ignore_label = ignore_label
        auger.aux_unknown = -32767

    def n_expected(auger):
        """
        Returns how many augmentations will be generated per-image
        """
        num = 0
        for v in auger.params.values():
            if util.isiterable(v):
                num += len(v)
            else:
                num += v
        return num

    def augment(auger, im, gt, aux={}):
        rng = auger.rng
        # Flips
        for axis in auger.params['axis_flips']:
            im_aug = cv2.flip(im, axis)
            gt_aug = cv2.flip(gt, axis)
            aux_aug = {key: cv2.flip(val, axis) for key, val in aux.items()}
            yield (im_aug, gt_aug, aux_aug)

        # Power law
        # for gamma in [.5, .8, 1.5, 2.0, 2.5]:
        for gamma in auger.params['gammas']:
            im_aug = imutil.adjust_gamma(im, gamma=gamma)
            gt_aug = gt
            aux_aug = aux
            yield (im_aug, gt_aug, aux_aug)

        # Blur1
        for k in auger.params['blur']:
            im_aug = cv2.blur(im, (k, k))
            gt_aug = gt
            aux_aug = {key: cv2.blur(val, (k, k)) for key, val in aux.items()}
            yield (im_aug, gt_aug, aux_aug)

        # Random zoom
        if rng is None:
            rng = np.random
        h, w = im.shape[0:2]
        for _ in range(auger.params['n_rand_zoom']):
            r1 = int((rng.rand() * .25) * h)
            r2 = int((rng.rand() * .25 + .75) * h)
            c1 = int((rng.rand() * .25) * w)
            c2 = int((rng.rand() * .25 + .75) * w)
            im_aug = cv2.resize(im[r1:r2, c1:c2], (w, h), interpolation=cv2.INTER_LANCZOS4)
            gt_aug = cv2.resize(gt[r1:r2, c1:c2], (w, h), interpolation=cv2.INTER_NEAREST)
            aux_aug = {}
            for key, val in aux.items():
                aux_aug[key] = cv2.resize(val[r1:r2, c1:c2], (w, h), interpolation=cv2.INTER_NEAREST)
            yield (im_aug, gt_aug, aux_aug)

        # Random occlusion
        max_h, min_h = int(h * .3), int(h * .1)
        max_w, min_w = int(w * .3), int(w * .1)
        for _ in range(auger.params['n_rand_occl']):
            sy = int(rng.rand() * (max_h - min_h) + min_h)
            sx = int(rng.rand() * (max_w - min_w) + min_w)
            r1 = int(rng.rand() * (h - sy) + sy)
            c1 = int(rng.rand() * (h - sx) + sx)
            r2 = r1 + sy
            c2 = c1 + sx
            im_aug = im.copy()
            gt_aug = gt.copy()
            im_aug[r1:r2, c1:c2, :] = 0
            # CAREFUL: cannot just set the groundtruth label to zero.  The
            # network might be using that as a class.
            gt_aug[r1:r2, c1:c2] = auger.ignore_label
            aux_aug = {}
            for key, val in aux.items():
                aux_aug[key] = val.copy()
                aux_aug[key][r1:r2, c1:c2] = auger.aux_unknown
            yield (im_aug, gt_aug, aux_aug)
