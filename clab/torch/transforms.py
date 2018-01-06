import ubelt as ub
import numpy as np
import skimage
from PIL import Image
import six
import cv2
from clab.augment import augment_common
from clab.util import imutil
from clab import util


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


class ImageCenterScale(ub.NiceRepr):
    def __init__(self, im_mean, im_scale):
        self.im_mean = im_mean
        self.im_scale = im_scale

    def __nice__(self):
        try:
            return '{:g}, {:g}'.format(self.im_mean, self.im_scale)
        except:
            return '{}, {}'.format(self.im_mean, self.im_scale)

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


class AffineWarp(object):
    """
    Wrapper around three different methods for affine warping.  The fastest is
    cv2, followed by pil, followed by skimage.  The most flexible is skimage,
    and the least flexible is pil, but cv2 has lanczos resampling while the
    others do not. However, cv2.warpAffine has known bugs that sometimes occur
    with multiprocessing, so its not always 100% safe to use.
    """
    skimage_interp_lookup = {
        'nearest'   : 0,
        'linear'    : 1,
        'quadradic' : 2,
        'cubic'     : 3,
        'lanczos': NotImplemented,
    }
    pil_interp_lookup = {
        'nearest'   : Image.NEAREST,
        'linear'    : Image.BILINEAR,
        'quadradic' : NotImplemented,
        'cubic'     : Image.BICUBIC,
        'lanczos': NotImplemented,
    }
    cv2_interp_lookup = {
        'nearest'   : cv2.INTER_NEAREST,
        'linear'    : cv2.INTER_LINEAR,
        'quadradic' : NotImplemented,
        'cubic'     : cv2.INTER_CUBIC,
        'lanczos': cv2.INTER_LANCZOS4,
    }

    skimage_border_lookup = {
        'constant': 'constant',
        'reflect': 'reflect',
    }
    cv2_border_lookup = {
        'constant': cv2.BORDER_CONSTANT,
        'reflect': cv2.BORDER_REFLECT,
    }
    pil_border_lookup = {
        'constant': 'constant',
        'reflect': NotImplemented,
    }
    def __init__(self):
        pass

    def around_mat3x3(x, y, sx, sy, theta, shear, tx, ty):
        Aff = augment_common.affine_around_mat2x3(x, y, sx, sy, theta, shear,
                                                  tx, ty)
        matrix = np.array(Aff + [[0, 0, 1]])
        return matrix

    def make_warper(self, backend, shape, matrix, interp, border_mode):
        return {
            'cv2': self.cv2_warper,
            'pil': self.pil_warper,
            'skimage': self.skimage_warper,
        }[backend]
        pass

    def skimage_warper(self, shape, matrix, interp, border_mode):
        inv_matrix = np.linalg.inv(matrix)
        order = self.skimage_interp_lookup[interp]
        skaff = skimage.transform.AffineTransform(matrix=inv_matrix)

        def _warp(img):
            imaug = skimage.transform.warp(
                img, skaff, output_shape=img.shape, order=order,
                mode=border_mode, clip=True, preserve_range=True
            )
            imaug = imaug.astype(img.dtype)
            return imaug
        return _warp

    def pil_warper(self, shape, matrix, interp, border_mode):
        inv_matrix = np.linalg.inv(matrix)
        pil_aff_params = list(inv_matrix.ravel()[0:6])
        h1, w1 = shape[0:2]
        resample = self.pil_interp_lookup[interp]
        assert border_mode == 'constant'
        # from torchvision.transforms.functional import to_pil_image
        # n_chan = util.get_num_channels(img)
        # if n_chan == 3 and img.dtype != np.uint8:
        #     pass
        #     # need to convert to uint8
        # pil_img = to_pil_image(img)

        def _warp(img):
            pil_img = Image.fromarray(img)
            imaug = np.array(pil_img.transform((w1, h1), Image.AFFINE,
                                               pil_aff_params,
                                               resample=resample))
            return imaug
        return _warp

    def cv2_warper(self, shape, matrix, interp, border_mode):
        h1, w1 = shape[0:2]
        inv_matrix = np.linalg.inv(matrix)
        inv_matrix_2x3 = inv_matrix[0:2]
        borderMode = self.cv2_border_lookup[border_mode]
        cv2_interp = self.cv2_interp_lookup[interp]
        flags = cv2_interp | cv2.WARP_INVERSE_MAP
        def _warp(img):
            imaug = cv2.warpAffine(img, inv_matrix_2x3, dsize=(w1, h1),
                                   flags=flags, borderMode=borderMode)
            return imaug
        return _warp


class RandomWarpAffine(object):
    def __init__(self, rng=None, backend='skimage', **kw):
        self.rng = util.ensure_rng(rng)
        self.augkw = augment_common.PERTERB_AUG_KW.copy()
        self.augkw.update(kw)
        # self.interp = 'nearest'
        # self.border_mode = 'reflect'
        # self.backend = 'skimage'
        # self.backend = 'cv2'
        # self.backend = 'pil'
        self.border_mode = 'constant'
        self.interp = 'nearest'
        self.backend = backend

        # @ub.memoize
        # def _memo_empty_uint8(shape):
        #     data = np.empty(shape, dtype=np.uint8)
        #     return data

    def __call__(self, data):
        # raise NotImplementedError(
        #     'Use explicit calls to tie params between instances')
        params = self.random_params()
        return self.warp(data, params, interp=self.interp,
                         border_mode=self.border_mode, backend=self.backend)

    def random_params(self):
        affine_args = augment_common.random_affine_args(**self.augkw,
                                                        rng=self.rng)
        return affine_args

    def warp(self, img, params, interp='nearest', border_mode='constant', backend=None):
        """

            >>> from clab.torch.transforms import *
            >>> import plottool as pt
            >>> pt.qtensure()
            >>> img_orig = util.imread(ut.grab_test_imgpath('lena.png'))
            >>> def _test_backend(img, backend, fnum):
            >>>     try:
            >>>         pnum_ = pt.make_pnum_nextgen(nSubplots=20)
            >>>         self = RandomWarpAffine(0, backend=backend)
            >>>         for _ in range(20):
            >>>             imgaug = self.warp(img, self.random_params())
            >>>             #pt.imshow(imgaug, fnum=fnum, pnum=pnum_())
            >>>     except Exception as ex:
            >>>         print('{} failed {}'.format(backend, repr(ex)))
            >>>         pass
            >>> def _test_all_backends(img):
            >>>     _test_backend(img, backend='skimage', fnum=1)
            >>>     _test_backend(img, backend='pil', fnum=2)
            >>>     _test_backend(img, backend='cv2', fnum=3)
            >>> # --- RGB dtype uint8 ---
            >>> img =  img_orig.copy()
            >>> _test_all_backends(img)
            >>> # --- RGB dtype float32 --- (PIL FAILS)
            >>> img = img_orig.copy().astype(np.float32) / 255
            >>> _test_all_backends(img)
            >>> # --- GRAY dtype float32, 0-1 ---
            >>> img = (img_orig.astype(np.float32) / 255).mean(axis=2)
            >>> _test_all_backends(img)
            >>> # --- RGB dtype uint16 ---
            >>> img = ((img_orig.astype(np.float32) / 255) * (2 ** 16) - 1).astype(np.uint16)
            >>> _test_all_backends(img)
            >>> # --- GRAY dtype float32, -1000-1000 ---
            >>> img = (img_orig.astype(np.float32) / 255).mean(axis=2) * 2000 - 1000
            >>> print(ut.get_stats(RandomWarpAffine(0, backend='pil')(img)))
            >>> print(ut.get_stats(RandomWarpAffine(0, backend='skimage')(img)))
            >>> print(ut.get_stats(RandomWarpAffine(0, backend='cv2')(img)))
            >>> _test_all_backends(img)

        Ignore:
            >>> img = (img_orig.astype(np.float32) / 255).mean(axis=2) * 2000 - 1000
            >>> %timeit RandomWarpAffine(0, backend='skimage')(img)
            4.76 ms ± 86.8 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
            >>> %timeit RandomWarpAffine(0, backend='pil')(img)
            1.11 ms ± 9.85 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
            >>> %timeit RandomWarpAffine(0, backend='cv2')(img)
            25.6 ms ± 498 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)

            >>> img = (img_orig.astype(np.float32) / 255).mean(axis=2)
            >>> ub.Timerit(50, bestof=5).call(RandomWarpAffine(0, backend='skimage'), img)
            time per loop : 5.0 ms ± 0.26
            >>> ub.Timerit(50, bestof=5).call(RandomWarpAffine(0, backend='pil'), img)
            time per loop : 1.334 ms ± 0.19
            >>> ub.Timerit(50, bestof=5).call(RandomWarpAffine(0, backend='cv2'), img)
            time per loop : 20.24 ms ± 5.4

            >>> img = (img_orig.copy())[0:8, 0:8]
            >>> ub.Timerit(100, bestof=10).call(RandomWarpAffine(0, backend='skimage'), img)
            >>> ub.Timerit(100, bestof=10).call(RandomWarpAffine(0, backend='pil'), img)
            >>> ub.Timerit(100, bestof=10).call(RandomWarpAffine(0, backend='cv2'), img)

            xdata = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
            ydatas = ub.ddict(list)
            for size in xdata:
                img = cv2.resize(img_orig, (size, size))
                ydatas['ski'] += [ub.Timerit(100, bestof=10).call(RandomWarpAffine(0, backend='skimage'), img)]
                ydatas['pil'] += [ub.Timerit(100, bestof=10).call(RandomWarpAffine(0, backend='pil'), img)]
                ydatas['cv2'] += [ub.Timerit(100, bestof=10).call(RandomWarpAffine(0, backend='cv2'), img)]
            pt.multi_plot(xdata, ydatas, title='affine warp speed (RGB uint8)', xlabel='image size', ylabel='seconds', fnum=99, ymax=.1)
            pt.gca().set_yscale('log')
            pt.gca().set_ylim(5e-5, 1e-1)

            xdata = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
            ydatas_f32 = ub.ddict(list)
            for size in xdata:
                img = cv2.resize(img_orig, (size, size)).astype(np.float32).mean(axis=2)
                ydatas_f32['ski'] += [ub.Timerit(100, bestof=10).call(RandomWarpAffine(0, backend='skimage'), img)]
                ydatas_f32['pil'] += [ub.Timerit(100, bestof=10).call(RandomWarpAffine(0, backend='pil'), img)]
                ydatas_f32['cv2'] += [ub.Timerit(100, bestof=10).call(RandomWarpAffine(0, backend='cv2'), img)]
            pt.multi_plot(xdata, ydatas_f32, title='affine warp speed (GRAY float32)', xlabel='image size', ylabel='seconds', fnum=100, ymax=.1)
            pt.gca().set_yscale('log')
            pt.gca().set_ylim(5e-5, 1e-1)

            xdata = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
            ydatas_f64 = ub.ddict(list)
            for size in xdata:
                img = cv2.resize(img_orig, (size, size)).astype(np.float64).mean(axis=2)
                ydatas_f64['ski'] += [ub.Timerit(100, bestof=10).call(RandomWarpAffine(0, backend='skimage'), img)]
                ydatas_f64['pil'] += [ub.Timerit(100, bestof=10).call(RandomWarpAffine(0, backend='pil'), img)]
                ydatas_f64['cv2'] += [ub.Timerit(100, bestof=10).call(RandomWarpAffine(0, backend='cv2'), img)]
            pt.multi_plot(xdata, ydatas_f64, title='affine warp speed (GRAY float32)', xlabel='image size', ylabel='seconds', fnum=101, ymax=.1)
            pt.gca().set_yscale('log')
            pt.gca().set_ylim(5e-5, 1e-1)

            xdata = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
            ydatas_f64 = ub.ddict(list)
            for size in xdata:
                img = (cv2.resize(img_orig, (size, size)).astype(np.float64).mean(axis=2) * 255).astype(np.uint8)
                ydatas_f64['ski'] += [ub.Timerit(100, bestof=10).call(RandomWarpAffine(0, backend='skimage'), img)]
                ydatas_f64['pil'] += [ub.Timerit(100, bestof=10).call(RandomWarpAffine(0, backend='pil'), img)]
                ydatas_f64['cv2'] += [ub.Timerit(100, bestof=10).call(RandomWarpAffine(0, backend='cv2'), img)]
            pt.multi_plot(xdata, ydatas_f64, title='affine warp speed (GRAY uint8)', xlabel='image size', ylabel='seconds', fnum=102, ymax=.1)
            pt.gca().set_yscale('log')
            pt.gca().set_ylim(5e-5, 1e-1)


        """
        x, y = img.shape[1] / 2, img.shape[0] / 2

        if backend is None:
            backend = self.backend

        affwarp = AffineWarp()
        matrix = affwarp.around_mat3x3(x, y, *params)
        _warp = affwarp.make_warper(backend, img.shape, matrix, interp,
                                    border_mode)
        imaug = _warp(img)
        return imaug

    def sseg_warp(self, im, aux_channels, gt, border_mode='constant'):
        """
        Specialized warping for semantic segmentation problems
        """
        params = self.random_params()
        im_aug = self.warp(im, params, interp='cubic', border_mode=border_mode)
        aux_channels_aug = [
            self.warp(aux, params, interp='nearest', border_mode=border_mode)
            for aux in aux_channels
        ]
        gt_aug = self.warp(gt, params, interp='nearest', border_mode=border_mode)
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
    """
    TODO: use imgaug or something similar.

    k_pdf should be a class that samples from a distribution
    """
    def __init__(self, k_pdf=(2, 4), freq=.5, rng=None):
        self.rng = rng
        self.freq = freq
        self.k_pdf = k_pdf

    def __call__(self, data):
        if self.rng.rand() < self.freq:
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
        if self.rng.rand() < self.freq:
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
