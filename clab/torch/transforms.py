import numpy as np
import skimage
import six
import cv2
from clab.augment import augment_common
from clab.util import imutil


class NaNInputer(object):
    """
    Methods for removing nan or invalid values

    Example:
        >>> data = np.array([
        >>>     [-1, 20, 30, 40],
        >>>     [10, -1, 30, -1],
        >>>     [30, -1, 50, 90],
        >>>     [30, 11, 13, 90],
        >>> ])
        >>> self = NaNInputer(fill='median', nan_value=-1)
        >>> self(data)
        array([[30, 20, 30, 40],
               [10, 30, 30, 30],
               [30, 30, 50, 90],
               [30, 11, 13, 90]])

    Example:
        >>> channel = np.array([[-1, 20], [10, -1]])
        >>> data = np.dstack([channel, channel])
        >>> self = NaNInputer(fill='median', nan_value=-1)
        >>> self(data).tolist()
        [[[15, 15], [20, 20]], [[10, 10], [15, 15]]]
    """

    def __init__(self, fill='median', nan_value=np.nan):
        self.nan_value = nan_value

        if np.isnan(nan_value):
            self.isnan = np.isnan
        else:
            def isnan(data):
                return data == nan_value
            self.isnan = isnan

        if isinstance(fill, six.string_types):
            # Define a local method for inputing fill data
            if fill == 'median':
                self.getfill = np.median
            elif fill == 'mean':
                self.getfill = np.mean
            else:
                raise KeyError(fill)
        else:
            self.getfill = None
            self.fill_value = fill

    def __call__(self, data):
        mask = self.isnan(data)
        if self.getfill is None:
            fill_value = self.fill_value
        elif np.all(mask):
            fill_value = 0
        else:
            fill_value = self.getfill(data[~mask].ravel())
        data[mask] = fill_value
        return data


class LocalCenterGlobalScale(object):
    def __init__(self, std):
        self.std = std

    def __call__(self, data):
        center = np.median(data)
        data = (data - center) / self.std
        return data


class DTMCenterScale(object):
    """
    Overly specific class for urban3d mapper.

    Combines NaNInputer and LocalCenterGlobalScale because both use median for a
    tiny speed increase.
    """
    def __init__(self, std, fill='median', nan_value=-32767.0):
        # aux_std = 5.3757350869126723
        self.std = std
        self.nan_value = nan_value
        self.fill = fill

    def __call__(self, data):
        # zero the median on a per-chip basis, but use
        # the global internal_std to normalize extent
        mask = (data == self.nan_value)
        if np.all(mask):
            center = 0
        else:
            center = self._get_center(data[~mask].ravel())
        data[mask] = center
        data = (data - center) / self.std
        return data

    @property
    def fill(self):
        return self._fill

    @fill.setter
    def fill(self, fill):
        valid_center_funcs = {
            'median': np.median
        }
        self._fill = fill
        self._get_center = valid_center_funcs[fill]

    def __getstate__(self):
        return {'std': self.std, 'nan_value': self.nan_value, 'fill': self.fill}

    def __setstate__(self, state):
        for key, value in state.items():
            setattr(key, value)


class ImageCenterScale():
    def __init__(self, im_mean, im_scale):
        self.im_mean = im_mean
        self.im_scale = im_scale

    def __call__(self, data):
        return (data - self.im_mean) / self.im_scale

    def invert(self, data):
        return (data * self.im_scale) + self.im_mean

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        for key, value in state.items():
            setattr(key, value)


class ZipTransforms():
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, inputs):
        return [
            transform(input)
            for transform, input in zip(self.transforms, inputs)
        ]


class RandomWarpAffine(object):
    def __init__(self, rng=None, **kw):
        self.rng = rng
        self.augkw = augment_common.PERTERB_AUG_KW.copy()
        self.augkw.update(kw)
        self.skimage_interp_lookup = {
            'nearest'   : 0,
            'linear'    : 1,
            'quadradic' : 2,
            'cubic'     : 3,
            'lanczos': NotImplementedError,
        }
        # self.interp = 'nearest'
        # self.border_mode = 'reflect'

    def __call__(self, data):
        raise NotImplementedError(
            'Use explicit calls to tie params between instances')
        params = self.random_params()
        return self.warp(data, params, self.interp, self.border_mode)

    def random_params(self):
        affine_args = augment_common.random_affine_args(**self.augkw,
                                                        rng=self.rng)
        return affine_args

    def warp(self, img, params, interp='nearest', border_mode='reflect'):
        x, y = img.shape[1] / 2, img.shape[0] / 2
        Aff = augment_common.affine_around_mat2x3(x, y, *params)
        matrix = np.array(Aff + [[0, 0, 1]])
        skaff = skimage.transform.AffineTransform(matrix=matrix)

        order = self.skimage_interp_lookup[interp]

        imaug = skimage.transform.warp(
            img, skaff, output_shape=img.shape, order=order, mode=border_mode,
            clip=True, preserve_range=True
        )
        imaug = imaug.astype(img.dtype)
        return imaug

    def sseg_warp(self, im, aux_channels, gt):
        """
        Specialized warping for semantic segmentation problems
        """
        params = self.random_params()
        im_aug = self.warp(im, params, interp='cubic')
        aux_channels_aug = [
            self.warp(aux, params, interp='nearest')
            for aux in aux_channels
        ]
        gt_aug = self.warp(gt, params, interp='nearest')
        return im_aug, aux_channels_aug, gt_aug


# class RandomIntensity(object):
#     def __init__(self, rng):
#         self.rng = rng

#     def __call__(self, data):
#         if self.rng.rand() > .5:
#             gamma = self.rng.rand() * 2 + .5
#             img = imutil.adjust_gamma(data, gamma=gamma)

#         if self.rng.rand() > .5:
#             k = self.rng.randint(1, 2)
#             img = cv2.blur(data, (k, k))
#         return img


class RandomBlur(object):
    def __init__(self, k_pdf=(1, 2), freq=.5, rng=None):
        self.rng = rng
        self.freq = freq
        self.k_pdf = k_pdf

    def __call__(self, data):
        if self.rng.rand() > self.freq:
            k = self.rng.randint(*self.k_pdf)
            data = cv2.blur(data, (k, k))
        return data


class RandomGamma(object):
    def __init__(self, input_colorspace='RGB', gamma_pdf=(.5, 2.5), freq=.5,
                 rng=None):
        self.rng = rng
        self.gamma_pdf = gamma_pdf
        self.input_colorspace = input_colorspace
        self.freq = freq

    def __call__(self, data):
        if self.rng.rand() > self.freq:
            mn, mx = self.gamma_pdf
            gamma = self.rng.rand() * (mx - mn) + mn

            if self.input_colorspace != 'RGB':
                if self.input_colorspace == 'LAB':
                    data = cv2.cvtColor(data, cv2.COLOR_LAB2RGB)
                else:
                    raise KeyError(self.input_colorspace)

            data = imutil.adjust_gamma(data, gamma=gamma)

            if self.input_colorspace != 'RGB':
                if self.input_colorspace == 'LAB':
                    data = cv2.cvtColor(data, cv2.COLOR_RGB2LAB)
        return data
