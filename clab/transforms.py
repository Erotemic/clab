import ubelt as ub
import numpy as np
from PIL import Image
import six
import cv2
from clab.augment import augment_common
from clab.util import imutil
from clab import util
try:
    import skimage
    import skimage.transform
except ImportError:
    pass


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
            setattr(self, key, value)


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
            setattr(self, key, value)


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

    Benchmark:
        >>> from clab.transforms import *
        >>> import itertools as it
        >>> fpath = ub.grabdata('http://i.imgur.com/JGrqMnV.png', fname='lena.png')
        >>> img_orig = util.imread(fpath)
        >>> shape = img_orig.shape
        >>> aff = AffineWarp()
        >>> x, y = shape[1] // 2, shape[0] // 2
        >>> matrix = aff.around_mat3x3(x, y, sx=.8, theta=.3, shear=.001, tx=2.5)
        >>> N = 100

        >>> def _compare_results(results, scale, thresh):
        >>>     diffs = []
        >>>     for key1, key2 in it.combinations(results.keys(), 2):
        >>>         ave_diff = np.abs(results[key1].astype(np.float64) - results[key2].astype(np.float64)).mean() / scale
        >>>         #print('{} {} ave_diff = {!r}'.format(key1, key2, ave_diff))
        >>>         diffs.append(ave_diff)
        >>>     assert np.all(np.array(diffs) < thresh)

        >>> def _assert_range(results, min, max):
        >>>     for key, result in results.items():
        >>>         assert result.max() <= max, '{} {}'.format(key, result.max())
        >>>         assert result.min() >= min, '{} {}'.format(key, result.min())

        >>> # Results for rgb float01
        >>> img = img_orig.astype(np.float32) / 255
        >>> results = ub.odict()
        >>> for back in aff.backends:
        >>>     _warper = aff.make_warper(backend=back, shape=img.shape, matrix=matrix, mode='float01')
        >>>     results[back] = _warper(img)
        >>>     ub.Timerit(N, verbose=1, label=back).call(_warper, img)
        >>> _assert_range(results, 0, 1)
        >>> _compare_results(results, scale=1, thresh=.02)
        Timed ski for: 10 loops, best of 3
            time per loop : best=67.75 ms, mean=68.14 ± 0.35 ms
        Timed pil for: 10 loops, best of 3
            time per loop : best=23.44 ms, mean=23.7 ± 0.22 ms
        Timed cv2 for: 10 loops, best of 3
            time per loop : best=3.403 ms, mean=3.713 ± 0.2 ms

        >>> # Results for grayscale float01
        >>> img = (img_orig.astype(np.float32) / 255).mean(axis=2)
        >>> results = ub.odict()
        >>> for back in aff.backends:
        >>>     _warper = aff.make_warper(backend=back, shape=img.shape, matrix=matrix, mode='float01')
        >>>     results[back] = _warper(img)
        >>>     ub.Timerit(N, verbose=1, label=back).call(_warper, img)
        >>> _compare_results(results, scale=1, thresh=.02)
        >>> _assert_range(results, 0, 1)
        Timed ski for: 10 loops, best of 3
            time per loop : best=21.31 ms, mean=22.5 ± 1.9 ms
        Timed pil for: 10 loops, best of 3
            time per loop : best=9.877 ms, mean=10.24 ± 0.37 ms
        Timed cv2 for: 10 loops, best of 3
            time per loop : best=1.753 ms, mean=1.926 ± 0.15 ms

        >>> # Results for grayscale uint8
        >>> img = ((img_orig.astype(np.float32) / 255).mean(axis=2) * 255).astype(np.uint8)
        >>> results = ub.odict()
        >>> for back in aff.backends:
        >>>     _warper = aff.make_warper(backend=back, shape=img.shape, matrix=matrix, mode='uint8')
        >>>     results[back] = _warper(img)
        >>>     ub.Timerit(N, verbose=1, label=back).call(_warper, img)
        >>> _compare_results(results, scale=255, thresh=.02)
        Timed ski for: 10 loops, best of 3
            time per loop : best=21.46 ms, mean=21.56 ± 0.14 ms
        Timed pil for: 10 loops, best of 3
            time per loop : best=11.22 ms, mean=11.9 ± 0.76 ms
        Timed cv2 for: 10 loops, best of 3
            time per loop : best=1.518 ms, mean=1.596 ± 0.053 ms

        >>> # Results for rgb uint8
        >>> img = img_orig.copy()
        >>> results = ub.odict()
        >>> for back in aff.backends:
        >>>     _warper = aff.make_warper(shape=img.shape, matrix=matrix, backend=back, mode='uint8')
        >>>     results[back] = _warper(img)
        >>>     ub.Timerit(N, verbose=1, label=back).call(_warper, img)
        >>> _compare_results(results, scale=255, thresh=.02)
        Timed ski for: 10 loops, best of 3
            time per loop : best=67.65 ms, mean=67.9 ± 0.25 ms
        Timed pil for: 10 loops, best of 3
            time per loop : best=21.38 ms, mean=21.65 ± 0.39 ms
        Timed cv2 for: 10 loops, best of 3
            time per loop : best=2.927 ms, mean=3.318 ± 0.3 ms

    Ignore:
        >>> # +SKIP
        >>> import plottool as pt
        >>> pt.qtensure()
        >>> pnum_ = pt.make_pnum_nextgen(nCols=len(results) + 1)
        >>> pt.imshow(img, pnum=pnum_(), fnum=1)
        >>> for key, result in results.items():
        >>>     pt.imshow(result, pnum=pnum_(), fnum=1, title=key)

    Ignore:
        >>> # +SKIP
        >>> %timeit aff.make_warper(shape=img.shape, matrix=matrix, backend='ski')(img)
        >>> %timeit aff.make_warper(shape=img.shape, matrix=matrix, backend='pil')(img)
        >>> %timeit aff.make_warper(shape=img.shape, matrix=matrix, backend='cv2')(img)
        >>> %timeit aff.make_warper(shape=img.shape, matrix=matrix, backend='cv2_inv')(img)
        10 loops, best of 3: 22.6 ms per loop
        100 loops, best of 3: 7.97 ms per loop
        1000 loops, best of 3: 1.41 ms per loop
    """
    backends = ['cv2', 'pil', 'ski']

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
        self.defaults = {
            'backend': 'cv2',
            'interp': 'cubic',
            'border_mode': 'constant',
            'mode': 'float01',
            'clip': 'auto',
        }

    def warp(self, img, **kwargs):
        _warper = self.make_warper(**kwargs)
        return _warper(img)

    def around_mat3x3(self, x, y, sx=1.0, sy=1.0, theta=0.0, shear=0.0, tx=0.0,
                      ty=0.0, flip_ud=False, flip_lr=False):
        mat_2x3 = augment_common.affine_around_mat2x3(x, y, sx, sy, theta,
                                                      shear, tx, ty,
                                                      flip_ud=flip_ud,
                                                      flip_lr=flip_lr)
        mat_3x3 = np.array(mat_2x3 + [[0, 0, 1]])
        return mat_3x3

    def mat3x3(self, sx=1.0, sy=1.0, theta=0.0, shear=0.0, tx=0.0,
               ty=0.0, flip_ud=False, flip_lr=False):
        mat_2x3 = augment_common.affine_mat2x3(sx, sy, theta, shear, tx, ty,
                                               flip_ud=flip_ud,
                                               flip_lr=flip_lr)
        mat_3x3 = np.array(mat_2x3 + [[0, 0, 1]])
        return mat_3x3

    def make_warper(self, shape, matrix, interp=None, border_mode=None,
                    clip=None, mode=None, backend=None):

        if backend is None:
            backend = self.defaults['backend']
        if interp is None:
            interp = self.defaults['interp']
        if border_mode is None:
            border_mode = self.defaults['border_mode']
        if mode is None:
            mode = self.defaults['mode']
        if clip is None:
            clip = self.defaults['clip']

        if backend == 'skimage':
            backend = 'ski'
        if clip == 'auto':
            # Find a "good" value for clip
            if mode == 'float01':
                # We only need to clip for higher order interpolation methods
                if interp not in {'nearest', 'linear'}:
                    clip = True
                else:
                    clip = False

            elif mode == 'uint8':
                # only clip uint8 for skimage
                if backend == 'ski':
                    clip = True
                else:
                    clip = False
        return {
            'cv2': self.make_cv2_warper,
            'pil': self.make_pil_warper,
            'ski': self.make_skimage_warper,
        }[backend](shape, matrix, interp, border_mode, clip, mode)

    def make_skimage_warper(self, shape, matrix, interp, border_mode, clip, mode):
        inv_matrix = np.linalg.inv(matrix)
        order = self.skimage_interp_lookup[interp]
        skaff = skimage.transform.AffineTransform(matrix=inv_matrix)

        def _sk_warper(img):
            imaug = skimage.transform.warp(
                img, skaff, output_shape=img.shape, order=order,
                mode=border_mode, clip=clip, preserve_range=True
            )
            imaug = imaug.astype(img.dtype)
            return imaug
        return _sk_warper

    def make_pil_warper(self, shape, matrix, interp, border_mode, clip, mode):
        inv_matrix = np.linalg.inv(matrix)
        pil_aff_params = list(inv_matrix.ravel()[0:6])
        h, w = shape[0:2]
        resample = self.pil_interp_lookup[interp]
        assert border_mode == 'constant', 'pil can only do constant right now'

        need_convert_uint8 = False

        if len(shape) > 2 and shape[2] >= 3:
            # TODO: fixme when floating point values are not in the 0-1 range
            if mode == 'float01':
                # PIL cannot handle multi-channel float images.
                # Need to convert to uint8
                need_convert_uint8 = True
                clip = False

        def _pil_warper(img):
            if need_convert_uint8:
                orig_dtype = img.dtype
                img = (img * 255.0).astype(np.uint8)

            imaug = np.array(Image.fromarray(img).transform((w, h),
                                                            Image.AFFINE,
                                                            pil_aff_params,
                                                            resample=resample))
            if clip:
                np.clip(imaug, 0, 1, out=imaug)
            if need_convert_uint8:
                imaug = imaug.astype(orig_dtype) / 255.0
            return imaug
        return _pil_warper

    def make_cv2_warper(self, shape, matrix, interp, border_mode, clip, mode):
        h, w = shape[0:2]
        borderMode = self.cv2_border_lookup[border_mode]
        cv2_interp = self.cv2_interp_lookup[interp]
        # It is slightly faster (5-10%) to pass an inverted matrix to cv2
        inv_mat_2x3 = np.linalg.inv(matrix)[0:2]
        inv_flags = cv2_interp | cv2.WARP_INVERSE_MAP

        # TODO: only clip if we are in float01 space
        if clip:
            def _cv2_warper(img):
                imaug = cv2.warpAffine(img, inv_mat_2x3, dsize=(w, h),
                                       flags=inv_flags, borderMode=borderMode)
                np.clip(imaug, 0, 1, out=imaug)
                return imaug
        else:
            def _cv2_warper(img):
                imaug = cv2.warpAffine(img, inv_mat_2x3, dsize=(w, h),
                                       flags=inv_flags, borderMode=borderMode)
                return imaug
        return _cv2_warper


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
        affine_args = augment_common.random_affine_args(rng=self.rng,
                                                        **self.augkw)
        return affine_args

    def warp(self, img, params, interp='nearest', border_mode='constant', backend=None):
        """

            >>> from clab.transforms import *
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

        SpeedPlots:
            >>> from clab.transforms import *
            >>> import plottool as pt
            >>> pt.qtensure()
            >>> img_orig = util.imread(ut.grab_test_imgpath('lena.png'))
            >>> img = (img_orig.astype(np.float32) / 255).mean(axis=2) * 2000 - 1000
            >>> xdata = [2, 4, 8, 16, 32, 64, 80, 128, 192, 256, 384, 512, 768, 1024, 2048]
            >>> #
            >>> ydatas = ub.ddict(list)
            >>> N = 30
            >>> for size in xdata:
            >>>     img = cv2.resize(img_orig, (size, size))
            >>>     ydatas['ski'] += [ub.Timerit(N, bestof=3).call(RandomWarpAffine(0, backend='skimage'), img).min()]
            >>>     ydatas['pil'] += [ub.Timerit(N, bestof=3).call(RandomWarpAffine(0, backend='pil'), img).min()]
            >>>     ydatas['cv2'] += [ub.Timerit(N, bestof=3).call(RandomWarpAffine(0, backend='cv2'), img).min()]
            >>> pt.multi_plot(xdata, ydatas, title='affine warp speed (RGB uint8)', xlabel='image size', ylabel='seconds', fnum=99, ymax=.1)
            >>> pt.gca().set_yscale('log')
            >>> pt.gca().set_ylim(5e-5, 1e-1)
            >>> pt.gcf().savefig('rgb_uint8.png')
            >>> #
            >>> ydatas_f32 = ub.ddict(list)
            >>> for size in xdata:
            >>>     img = cv2.resize(img_orig, (size, size)).astype(np.float32).mean(axis=2)
            >>>     ydatas_f32['ski'] += [ub.Timerit(N, bestof=3).call(RandomWarpAffine(0, backend='skimage'), img).min()]
            >>>     ydatas_f32['pil'] += [ub.Timerit(N, bestof=3).call(RandomWarpAffine(0, backend='pil'), img).min()]
            >>>     ydatas_f32['cv2'] += [ub.Timerit(N, bestof=3).call(RandomWarpAffine(0, backend='cv2'), img).min()]
            >>> pt.multi_plot(xdata, ydatas_f32, title='affine warp speed (GRAY float32)', xlabel='image size', ylabel='seconds', fnum=100, ymax=.1)
            >>> pt.gca().set_yscale('log')
            >>> pt.gca().set_ylim(5e-5, 1e-1)
            >>> pt.gcf().savefig('gray_float32.png')
            >>> #
            >>> ydatas_f64 = ub.ddict(list)
            >>> for size in xdata:
            >>>     img = cv2.resize(img_orig, (size, size)).astype(np.float64).mean(axis=2)
            >>>     ydatas_f64['ski'] += [ub.Timerit(N, bestof=3).call(RandomWarpAffine(0, backend='skimage'), img).min()]
            >>>     ydatas_f64['pil'] += [ub.Timerit(N, bestof=3).call(RandomWarpAffine(0, backend='pil'), img).min()]
            >>>     ydatas_f64['cv2'] += [ub.Timerit(N, bestof=3).call(RandomWarpAffine(0, backend='cv2'), img).min()]
            >>> pt.multi_plot(xdata, ydatas_f64, title='affine warp speed (GRAY float64)', xlabel='image size', ylabel='seconds', fnum=101, ymax=.1)
            >>> pt.gca().set_yscale('log')
            >>> pt.gca().set_ylim(5e-5, 1e-1)
            >>> pt.gcf().savefig('gray_float64.png')
            >>> #
            >>> ydatas_f64 = ub.ddict(list)
            >>> for size in xdata:
            >>>     img = (cv2.resize(img_orig, (size, size)).astype(np.float64).mean(axis=2) * 255).astype(np.uint8)
            >>>     ydatas_f64['ski'] += [ub.Timerit(N, bestof=3).call(RandomWarpAffine(0, backend='skimage'), img).min()]
            >>>     ydatas_f64['pil'] += [ub.Timerit(N, bestof=3).call(RandomWarpAffine(0, backend='pil'), img).min()]
            >>>     ydatas_f64['cv2'] += [ub.Timerit(N, bestof=3).call(RandomWarpAffine(0, backend='cv2'), img).min()]
            >>> pt.multi_plot(xdata, ydatas_f64, title='affine warp speed (GRAY uint8)', xlabel='image size', ylabel='seconds', fnum=102, ymax=.1)
            >>> pt.gca().set_yscale('log')
            >>> pt.gca().set_ylim(5e-5, 1e-1)
            >>> pt.gcf().savefig('gray_uint8.png')

        Ignore:
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


        """
        x, y = img.shape[1] / 2, img.shape[0] / 2

        if backend is None:
            backend = self.backend

        affwarp = AffineWarp()
        matrix = affwarp.around_mat3x3(x, y, *params)
        _warp = affwarp.make_warper(shape=img.shape, matrix=matrix,
                                    interp=interp, backend=backend,
                                    border_mode=border_mode)
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
