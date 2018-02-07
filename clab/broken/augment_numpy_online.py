"""
DEPRICATE
"""
from clab.util import imutil
from clab import util  # NOQA
import ubelt as ub
import cv2
import numpy as np
try:
    import skimage.transform
except ImportError:
    pass
from clab.augment.augment_common import random_affine_args, affine_around_mat2x3, PERTERB_AUG_KW


SKIMAGE_INTERP_LOOKUP = {
    'nearest'   : 0,
    'linear'    : 1,
    'quadradic' : 2,
    'cubic'     : 3,
    'lanczos': NotImplementedError,
}


def online_affine_perterb_np(np_images, rng, interp='cubic',
                             border_mode='reflect', **kw):
    """
    Args:
        np_images (list) list of images to receive the same transform

    Exception:
        >>> from clab.augment.augment_numpy_online import *
        >>> import ubelt as ub
        >>> import numpy as np
        >>> import plottool as pt
        >>> rng = np.random
        >>> fpath = ub.grabdata('https://i.imgur.com/oHGsmvF.png', fname='carl.png')
        >>> img = imutil.imread(fpath)
        >>> np_images = [img]
        >>> kw = {}
        >>> imaug, = online_affine_perterb_np([img], rng)
        >>> pt.imshow(np.array(imaug))

    Ignore:
        Aff = affine_around_mat2x3(0, 0)
        matrix = np.array(Aff + [[0, 0, 1]])
        skaff = skimage.transform.AffineTransform(matrix=matrix)

        # CONCLUSION: this can handle n-channel images
        img2 = np.random.rand(24, 24, 5)
        imaug2 = skimage.transform.warp(
            img2, skaff, output_shape=img2.shape, order=0, mode='reflect',
            clip=True, preserve_range=True)

    """
    augkw = PERTERB_AUG_KW.copy()
    augkw.update(kw)
    affine_args = random_affine_args(rng=rng, **augkw)

    if not ub.iterable(interp):
        interps = [interp] * len(np_images)
    else:
        interps = interp
    assert len(interps) == len(np_images)

    for img, interp_ in zip(np_images, interps):
        h1, w1 = img.shape[0:2]
        x, y = w1 / 2, h1 / 2

        Aff = affine_around_mat2x3(x, y, *affine_args)
        matrix = np.array(Aff + [[0, 0, 1]])
        skaff = skimage.transform.AffineTransform(matrix=matrix)

        order = SKIMAGE_INTERP_LOOKUP[interp_]

        imaug = skimage.transform.warp(
            img, skaff, output_shape=img.shape, order=order,
            mode=border_mode,
            # cval=0.0,
            clip=True, preserve_range=True)
        imaug = imaug.astype(img.dtype)

        # imaug = cv2.warpAffine(
        #     img, Aff,
        #     dsize=(w1, h1),
        #     flags=cv2.INTER_LANCZOS4,
        #     borderMode=cv2.BORDER_REFLECT
        # )
        yield imaug


def online_intensity_augment_np(img, rng):
    """
        >>> from clab.augment.augment_numpy_online import *
        >>> import ubelt as ub
        >>> fpath = ub.grabdata('https://i.imgur.com/oHGsmvF.png', fname='carl.png')
        >>> img = imutil.imread(fpath)
        >>> imaug = online_intensity_augment_np(img, rng)
        >>> pt.imshow(np.array(imaug))
    """
    if rng.rand() > .5:
        gamma = rng.rand() * 2 + .5
        img = imutil.adjust_gamma(img, gamma=gamma)

    if rng.rand() > .5:
        k = rng.randint(1, 2)
        img = cv2.blur(img, (k, k))
    return img
