"""
DEPRICATE
"""
from PIL import Image
from PIL import ImageFilter
# from PIL import ImageEnhance
import itertools as it
from clab.augment.augment_common import random_affine_args, affine_around_mat2x3, PERTERB_AUG_KW


def online_affine_perterb(pil_images, rng, **kw):
    """
    Args:
        pil_images (list) list of images to receive the same transform

    Exception:
        >>> from clab.augment import *
        >>> from PIL import Image
        >>> import ubelt as ub
        >>> import numpy as np
        >>> import plottool as pt
        >>> rng = np.random
        >>> fpath = ub.grabdata('http://i.imgur.com/JGrqMnV.png', fname='lena.png')
        >>> img = Image.open(fpath)
        >>> kw = {}
        >>> imaug, = online_affine_perterb([img], rng)
        >>> #imaug.show()
        >>> x = np.array(imaug)[:, :, ::-1]
        >>> y = np.array(imaug)[:, :, ::-1]
        >>> pt.imshow(np.array(imaug)[:, :, ::-1])
    """
    augkw = PERTERB_AUG_KW.copy()
    augkw.update(kw)
    affine_args = random_affine_args(rng=rng, **augkw)

    for img in pil_images:
        w1, h1 = img.size
        x, y = w1 / 2, h1 / 2
        Aff = affine_around_mat2x3(x, y, *affine_args)
        pil_aff_params = list(it.chain.from_iterable(Aff))
        imgaug = img.transform((w1, h1), Image.AFFINE, pil_aff_params, fill=-1)
        # Image.AFFINE
        # # Image.BICUBIC
        # Image.NEAREST
        # imaug = cv2.warpAffine(
        #     img, Aff,
        #     dsize=(w1, h1),
        #     flags=cv2.INTER_LANCZOS4,
        #     borderMode=cv2.BORDER_REFLECT
        # )
        yield imgaug


def online_intensity_augment(img, rng):
    """
        >>> from clab.augment import *
        >>> import ubelt as ub
        >>> fpath = ub.grabdata('http://i.imgur.com/JGrqMnV.png', fname='lena.png')
        >>> img = Image.open(fpath)
        >>> imaug = online_intensity_augment(img, rng)
        >>> pt.imshow(np.array(imaug)[:, :, ::-1])
    """
    # if rng.rand() > .5:
    #     gamma = rng.rand() * 2 + .5
    #     img = imutil.adjust_gamma(img, gamma=gamma)
    #     PIL.ImageEnhance.Brightness

    if rng.rand() > .5:
        k = rng.randint(1, 2)
        # img = cv2.blur(img, (k, k))
        img = img.filter(ImageFilter.GaussianBlur(k))
    return img
