# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import glob
import ubelt as ub
from os.path import expanduser, exists, join, basename
import warnings
import numpy as np
import cv2
import six
from os.path import splitext  # NOQA
import itertools as it  # NOQA
import skimage.io

from clab import getLogger
logger = getLogger(__name__)
print = logger.info


CV2_INTERPOLATION_TYPES = {
    'nearest': cv2.INTER_NEAREST,
    'linear':  cv2.INTER_LINEAR,
    'area':    cv2.INTER_AREA,
    'cubic':   cv2.INTER_CUBIC,
    'lanczos': cv2.INTER_LANCZOS4
}


def _rectify_interpolation(interp, default=cv2.INTER_LANCZOS4):
    """
    Converts interpolation into flags suitable cv2 functions

    Args:
        interp (int or str): string or cv2-style interpolation type
        default (int): cv2 interpolation flag to use if `interp` is None

    Returns:
        int: flag specifying interpolation type that can be passed to
           functions like cv2.resize, cv2.warpAffine, etc...
    """
    if interp is None:
        return default
    elif isinstance(interp, six.text_type):
        try:
            return CV2_INTERPOLATION_TYPES[interp]
        except KeyError:
            print('Valid values for interpolation are {}'.format(
                list(CV2_INTERPOLATION_TYPES.keys())))
            raise
    else:
        return interp


def imscale(img, scale, interpolation=None):
    """
    Resizes an image by a scale factor.

    Because the result image must have an integer number of pixels, the scale
    factor is rounded, and the rounded scale factor is returned.

    Args:
        dsize (ndarray): an image
        scale (float or tuple): desired floating point scale factor
    """
    dsize = img.shape[0:2][::-1]
    try:
        sx, sy = scale
    except TypeError:
        sx = sy = scale
    w, h = dsize
    new_w = int(round(w * sx))
    new_h = int(round(h * sy))
    new_scale = new_w / w, new_h / h
    new_dsize = (new_w, new_h)

    interpolation = _rectify_interpolation(interpolation)
    new_img = cv2.resize(img, new_dsize, interpolation=interpolation)
    return new_img, new_scale


def adjust_gamma(img, gamma=1.0):
    """
    gamma correction function

    References:
        http://www.pyimagesearch.com/2015/10/05/opencv-gamma-correction/

    Ignore:
        >>> from clab.util.imutil import *
        >>> import ubelt as ub
        >>> fpath = ub.grabdata('http://i.imgur.com/JGrqMnV.png', fname='lena.png')
        >>> img = imread(fpath)
        >>> gamma = .5
        >>> imgf = ensure_float01(img)
        >>> img2 = adjust_gamma(img, gamma)
        >>> img3 = adjust_gamma(imgf, gamma)
        >>> import plottool as pt
        >>> pt.qtensure()
        >>> pt.imshow(img, pnum=(3, 3, 1), fnum=1)
        >>> pt.imshow(img2, pnum=(3, 3, 2), fnum=1)
        >>> pt.imshow(img3, pnum=(3, 3, 3), fnum=1)
        >>> pt.imshow(adjust_gamma(img, 1), pnum=(3, 3, 5), fnum=1)
        >>> pt.imshow(adjust_gamma(imgf, 1), pnum=(3, 3, 6), fnum=1)
        >>> pt.imshow(adjust_gamma(img, 2), pnum=(3, 3, 8), fnum=1)
        >>> pt.imshow(adjust_gamma(imgf, 2), pnum=(3, 3, 9), fnum=1)
    """
    if img.dtype.kind in ('i', 'u'):
        # build a lookup table mapping the pixel values [0, 255] to
        # their adjusted gamma values
        table = (((np.arange(0, 256) / 255.0) ** (1 / gamma)) * 255).astype(np.uint8)
        invGamma = 1.0 / gamma
        table = np.array([
            ((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)
        ]).astype("uint8")
        # apply gamma correction using the lookup table
        return cv2.LUT(img, table)
    else:
        np_img = ensure_float01(img)
        gain = 1
        np_img = gain * (np_img ** (1 / gamma))
        np_img = np.clip(np_img, 0, 1)
        return np_img


def ensure_float01(img, dtype=np.float32):
    """ Ensure that an image is encoded using a float properly """
    if img.dtype.kind in ('i', 'u'):
        assert img.max() <= 255
        img_ = img.astype(dtype) / 255.0
    else:
        img_ = img.astype(dtype)
    return img_


def make_channels_comparable(img1, img2):
    """
    Broadcasts image arrays so they can have elementwise operations applied

    CommandLine:
        python -m vtool.image make_channels_comparable

    Example:
        >>> # DISABLE_DOCTEST
        >>> wh_basis = [(5, 5), (3, 5), (5, 3), (1, 1), (1, 3), (3, 1)]
        >>> for w, h in wh_basis:
        >>>     shape_basis = [(w, h), (w, h, 1), (w, h, 3)]
        >>>     # Test all permutations of shap inputs
        >>>     for shape1, shape2 in ut.product(shape_basis, shape_basis):
        >>>         print('*    input shapes: %r, %r' % (shape1, shape2))
        >>>         img1 = np.empty(shape1)
        >>>         img2 = np.empty(shape2)
        >>>         img1, img2 = make_channels_comparable(img1, img2)
        >>>         print('... output shapes: %r, %r' % (img1.shape, img2.shape))
        >>>         elem = (img1 + img2)
        >>>         print('... elem(+) shape: %r' % (elem.shape,))
        >>>         assert elem.size == img1.size, 'outputs should have same size'
        >>>         assert img1.size == img2.size, 'new imgs should have same size'
        >>>         print('--------')
    """
    if img1.shape != img2.shape:
        c1 = get_num_channels(img1)
        c2 = get_num_channels(img2)
        if len(img1.shape) == 2 and len(img2.shape) == 2:
            raise AssertionError('UNREACHABLE: Both are 2-grayscale')
        elif len(img1.shape) == 3 and len(img2.shape) == 2:
            # Image 2 is grayscale
            if c1 == 3:
                img2 = np.tile(img2[..., None], 3)
            else:
                img2 = img2[..., None]
        elif len(img1.shape) == 2 and len(img2.shape) == 3:
            # Image 1 is grayscale
            if c2 == 3:
                img1 = np.tile(img1[..., None], 3)
            else:
                img1 = img1[..., None]
        elif len(img1.shape) == 3 and len(img2.shape) == 3:
            # Both images have 3 dims.
            # Check if either have color, then check for alpha
            if c1 == 1 and c2 == 1:
                raise AssertionError('UNREACHABLE: Both are 3-grayscale')
            elif c1 == 3 and c2 == 3:
                raise AssertionError('UNREACHABLE: Both are 3-color')
            elif c1 == 1 and c2 == 3:
                img1 = np.tile(img1, 3)
            elif c1 == 3 and c2 == 1:
                img2 = np.tile(img2, 3)
            elif c1 == 1 and c2  == 4:
                img1 = np.dstack((np.tile(img1, 3), np.ones(img1.shape[0:2])))
            elif c1 == 4 and c2  == 1:
                img2 = np.dstack((np.tile(img2, 3), np.ones(img2.shape[0:2])))
            elif c1 == 3 and c2  == 4:
                img1 = np.dstack((img1, np.ones(img1.shape[0:2])))
            elif c1 == 4 and c2  == 3:
                img2 = np.dstack((img2, np.ones(img2.shape[0:2])))
            else:
                raise AssertionError('Unknown shape case: %r, %r' % (img1.shape, img2.shape))
        else:
            raise AssertionError('Unknown shape case: %r, %r' % (img1.shape, img2.shape))
    return img1, img2


def get_num_channels(img):
    """ Returns the number of color channels """
    ndims = len(img.shape)
    if ndims == 2:
        nChannels = 1
    elif ndims == 3 and img.shape[2] == 3:
        nChannels = 3
    elif ndims == 3 and img.shape[2] == 4:
        nChannels = 4
    elif ndims == 3 and img.shape[2] == 1:
        nChannels = 1
    else:
        raise ValueError('Cannot determine number of channels '
                         'for img.shape={}'.format(img.shape))
    return nChannels


def overlay_alpha_images(img1, img2, keepalpha=True):
    """
    places img1 on top of img2 respecting alpha channels

    References:
        http://stackoverflow.com/questions/25182421/overlay-numpy-alpha
        https://en.wikipedia.org/wiki/Alpha_compositing#Alpha_blending
    """
    img1 = ensure_float01(img1)
    img2 = ensure_float01(img2)

    img1, img2 = make_channels_comparable(img1, img2)

    c1 = get_num_channels(img1)
    c2 = get_num_channels(img2)
    if c1 == 4:
        alpha1 = img1[:, :, 3]
    else:
        alpha1 = np.ones(img1.shape[0:2], dtype=img1.dtype)

    if c2 == 4:
        alpha2 = img2[:, :, 3]
    else:
        alpha2 = np.ones(img2.shape[0:2], dtype=img2.dtype)

    rgb1 = img1[:, :, 0:3]
    rgb2 = img2[:, :, 0:3]

    alpha3 = alpha1 + alpha2 * (1 - alpha1)

    numer1 = (rgb1 * alpha1[..., None])
    numer2 = (rgb2 * alpha2[..., None] * (1.0 - alpha1[..., None]))
    rgb3 = (numer1 + numer2) / alpha3[..., None]
    rgb3 = np.nan_to_num(rgb3)

    if keepalpha:
        img3 = np.dstack([rgb3, alpha3[..., None]])
    else:
        img3 = rgb3
    return img3


def ensure_alpha_channel(img, alpha=1.0):
    img = ensure_float01(img)
    c = get_num_channels(img)
    if c == 4:
        return img
    else:
        alpha_channel = np.full(img.shape[0:2], fill_value=alpha, dtype=img.dtype)
        if c == 3:
            return np.dstack([img, alpha_channel])
        elif c == 1:
            return np.dstack([img, img, img, alpha_channel])
        else:
            raise ValueError('unknown dim')


def ensure_grayscale(img, colorspace_hint='BGR'):
    img = ensure_float01(img)
    c = get_num_channels(img)
    if c == 1:
        return img
    else:
        return convert_colorspace(img, 'gray', colorspace_hint)


def convert_colorspace(img, dst_space, src_space='BGR', copy=False):
    r"""
    Converts colorspace of img.
    Convinience function around cv2.cvtColor

    Args:
        img (ndarray[uint8_t, ndim=2]):  image data
        colorspace (str): RGB, LAB, etc
        dst_space (unicode): (default = u'BGR')

    Returns:
        ndarray[uint8_t, ndim=2]: img -  image data

    Example:
        >>> convert_colorspace(np.array([[[0, 0, 1]]], dtype=np.float32), 'LAB', src_space='RGB')
        array([[[  32.29370117,   79.1875    , -107.859375  ]]], dtype=float32)
        >>> convert_colorspace(np.array([[[0, 1, 0]]], dtype=np.float32), 'LAB', src_space='RGB')
        array([[[ 87.73803711, -86.1875    ,  83.171875  ]]], dtype=float32)
        >>> convert_colorspace(np.array([[[1, 0, 0]]], dtype=np.float32), 'LAB', src_space='RGB')
        array([[[ 53.2409668,  80.09375  ,  67.203125 ]]], dtype=float32)
        >>> convert_colorspace(np.array([[[1, 1, 1]]], dtype=np.float32), 'LAB', src_space='RGB')
        array([[[ 100.,    0.,    0.]]], dtype=float32)
        >>> convert_colorspace(np.array([[[0, 0, 1]]], dtype=np.float32), 'HSV', src_space='RGB')
        array([[[ 240.        ,    0.99999988,    1.        ]]], dtype=float32)
    """
    dst_space = dst_space.upper()
    src_space = src_space.upper()
    if src_space == dst_space:
        img2 = img
        if copy:
            img2 = img2.copy()
    else:
        code = _lookup_colorspace_code(dst_space, src_space)
        # Note the conversion to colorspaces like LAB and HSV in float form
        # do not go into the 0-1 range. Instead they go into
        # (0-100, -111-111hs, -111-111is) and (0-360, 0-1, 0-1) respectively
        img2 = cv2.cvtColor(img, code)
    return img2


def _lookup_colorspace_code(dst_space, src_space='BGR'):
    src = src_space.upper()
    dst = dst_space.upper()
    convert_attr = 'COLOR_{}2{}'.format(src, dst)
    if not hasattr(cv2, convert_attr):
        prefix = 'COLOR_{}2'.format(src)
        valid_dst_spaces = [
            key.replace(prefix, '')
            for key in cv2.__dict__.keys() if key.startswith(prefix)]
        raise KeyError(
            '{} does not exist, valid conversions from {} are to {}'.format(
                convert_attr, src_space, valid_dst_spaces))
    else:
        code = getattr(cv2, convert_attr)
    return code


def overlay_colorized(colorized, orig, alpha=.6):
    """
    Overlays a color segmentation mask on an original image
    """
    color_mask = ensure_alpha_channel(colorized, alpha=alpha)
    gray_orig = ensure_grayscale(orig)
    color_blend = overlay_alpha_images(color_mask, gray_orig)
    color_blend = (color_blend * 255).astype(np.uint8)
    return color_blend


def load_image_paths(dpath, ext=('.png', '.tiff', 'tif')):
    dpath = expanduser(dpath)
    if not exists(dpath):
        raise ValueError('dpath = {} does not exist'.format(dpath))
    if not isinstance(ext, (list, tuple)):
        ext = [ext]

    image_paths = []
    for ext_ in ext:
        image_paths.extend(list(glob.glob(join(dpath, '*' + ext_))))
    # potentially non-general
    # (utilfname solves this though)
    image_paths = sorted(image_paths, key=basename)
    return image_paths


def imread(fpath, **kw):
    """
    reads image data in BGR format

    Example:
        >>> from clab.util.imutil import *
        >>> import tempfile
        >>> fpath = ub.grabdata('http://i.imgur.com/JGrqMnV.png', fname='lena.png')
        >>> fpath = ub.grabdata('http://www.topcoder.com/contest/problem/UrbanMapper3D/JAX_Tile_043_DTM.tif')
        >>> ext = splitext(fpath)[1]
        >>> img1 = imread(fpath)
        >>> # Check that write + read preserves data
        >>> tmp = tempfile.NamedTemporaryFile(suffix=ext)
        >>> imwrite(tmp.name, img1)
        >>> img2 = imread(tmp.name)
        >>> assert np.all(img2 == img1)

    Example:
        >>> #img1 = (np.arange(0, 12 * 12 * 3).reshape(12, 12, 3) % 255).astype(np.uint8)
        >>> img1 = imread(ub.grabdata('http://i.imgur.com/iXNf4Me.png', fname='ada.png'))
        >>> tmp_tif = tempfile.NamedTemporaryFile(suffix='.tif')
        >>> tmp_png = tempfile.NamedTemporaryFile(suffix='.png')
        >>> imwrite(tmp_tif.name, img1)
        >>> imwrite(tmp_png.name, img1)
        >>> tif_im = imread(tmp_tif.name)
        >>> png_im = imread(tmp_png.name)
        >>> assert np.all(tif_im == png_im)

    Example:
        >>> from clab.util.imutil import *
        >>> import ubelt as ub
        >>> import tempfile
        >>> #img1 = (np.arange(0, 12 * 12 * 3).reshape(12, 12, 3) % 255).astype(np.uint8)
        >>> tif_fpath = ub.grabdata('https://ghostscript.com/doc/tiff/test/images/rgb-3c-16b.tiff')
        >>> img1 = imread(tif_fpath)
        >>> tmp_tif = tempfile.NamedTemporaryFile(suffix='.tif')
        >>> tmp_png = tempfile.NamedTemporaryFile(suffix='.png')
        >>> imwrite(tmp_tif.name, img1)
        >>> imwrite(tmp_png.name, img1)
        >>> tif_im = imread(tmp_tif.name)
        >>> png_im = imread(tmp_png.name)
        >>> assert np.all(tif_im == png_im)

        import plottool as pt
        pt.qtensure()
        pt.imshow(tif_im / 2 ** 16, pnum=(1, 2, 1), fnum=1)
        pt.imshow(png_im / 2 ** 16, pnum=(1, 2, 2), fnum=1)

    Ignore:
        from PIL import Image
        pil_img = Image.open(tif_fpath)
        assert int(Image.PILLOW_VERSION.split('.')[0]) > 4
    """
    try:
        if fpath.endswith(('.tif', '.tiff')):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # skimage reads in RGB, convert to BGR
                image = skimage.io.imread(fpath, **kw)
                if get_num_channels(image) == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                elif get_num_channels(image) == 4:
                    image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)
        else:
            image = cv2.imread(fpath, flags=cv2.IMREAD_UNCHANGED)
            if image is None:
                raise IOError('OpenCV cannot read this image')
        return image
    except Exception as ex:
        print('Error reading fpath = {!r}'.format(fpath))
        raise


def imwrite(fpath, image, **kw):
    """
    writes image data in BGR format
    """
    if fpath.endswith(('.tif', '.tiff')):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # skimage writes in RGB, convert from BGR
            if get_num_channels(image) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            elif get_num_channels(image) == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
            return skimage.io.imsave(fpath, image)
    else:
        return cv2.imwrite(fpath, image)


def wide_strides_1d(start, margin, stop, step, keepbound=True):
    """
    Helper for `image_slices`. Generates slices in a single dimension.

    Args:
        start (int): starting point (in most cases set this to 0)

        margin (int): the length of the slice

        stop (int): the length of the image dimension

        step (int): the length of each step / distance between slices

        keepbound (bool): if True, a non-uniform step will be taken to ensure
            that the right / bottom of the image is returned as a slice if
            needed. Such a slice will not obey the overlap constraints.
            (Defaults to False)

    Yields:
        slice : slice in one dimension of size (margin)

    Example:
        >>> import sys
        >>> from clab.util.imutil import *
        >>> start = 0
        >>> stop = 2000
        >>> margin = 360
        >>> step = 360
        >>> keepbound = True
        >>> strides = list(wide_strides_1d(start, margin, stop, step, keepbound))
        >>> assert all([(s.stop - s.start) == margin for s in strides])
    """
    pos = start
    # probably could be more efficient with numpy here
    while True:
        yield slice(pos, pos + margin)
        # Stop once we reached the end
        if pos + margin == stop:
            break
        pos += step
        if pos + margin > stop:
            if keepbound:
                # Ensure the boundary is always used even if steps
                # would overshoot Could do some other strategy here
                pos = stop - margin
            else:
                break


def image_slices(img_shape, target_shape, overlap=0, keepbound=False):
    """
    Generates slices to break a large image into smaller pieces.

    Args:
        img_shape (tuple): height and width of the image

        target_shape (tuple): (height, width) of the

        overlap (float): a number between 0 and 1 indicating the fraction of
            overlap that parts will have. Must be `0 <= overlap < 1`.

        keepbound (bool): if True, a non-uniform step will be taken to ensure
            that the right / bottom of the image is returned as a slice if
            needed. Such a slice will not obey the overlap constraints.
            (Defaults to False)

    Yields:
        tuple(slice, slice): row and column slices used for numpy indexing

    Example:
        >>> import sys
        >>> from clab.util.imutil import *
        >>> img_shape = (2000, 2000)
        >>> target_shape = (360, 480)
        >>> overlap = 0
        >>> keepbound = True
        >>> list(image_slices(img_shape, target_shape, overlap, keepbound))
    """
    if overlap < 0 or overlap >= 1:
        raise ValueError(('part overlap was {}, but it must be '
                          'in the range [0, 1)').format(overlap))
    ph, pw = target_shape
    sy = int(ph - ph * overlap)
    sx = int(pw - pw * overlap)
    orig_h, orig_w = img_shape
    for rslice in wide_strides_1d(0, ph, orig_h, sy, keepbound):
        for cslice in wide_strides_1d(0, pw, orig_w, sx, keepbound):
            yield rslice, cslice


def run_length_encoding(img):
    """
    Run length encoding.

    Parameters
    ----------
    img : 2D image

    Example:
        >>> from clab.util.imutil import *
        >>> import ubelt as ub
        >>> lines = ub.codeblock(
        >>>     '''
        >>>     ..........
        >>>     ......111.
        >>>     ..2...111.
        >>>     .222..111.
        >>>     22222.....
        >>>     .222......
        >>>     ..2.......
        >>>     ''').replace('.', '0').splitlines()
        >>> img = np.array([list(map(int, line)) for line in lines])
        >>> (w, h), runlen = rlencode(img)
        >>> target = np.array([0,16,1,3,0,3,2,1,0,3,1,3,0,2,2,3,0,2,1,3,0,1,2,5,0,6,2,3,0,8,2,1,0,7])
        >>> assert np.all(target == runlen)
    """
    flat = img.ravel()
    diff_idxs = np.flatnonzero(np.abs(np.diff(flat)) > 0)
    pos = np.hstack([[0], diff_idxs + 1])

    values = flat[pos]
    lengths = np.hstack([np.diff(pos), [len(flat) - pos[-1]]])

    runlen = np.hstack([values[:, None], lengths[:, None]]).ravel()

    h, w = img.shape[0:2]
    return (w, h), runlen


def absdev(x, ave=np.mean, central=np.median, axis=None):
    """
    Average absolute deviation from a point of central tendency

    The `ave` absolute deviation from the `central`.

    Args:
        x (np.ndarray): input data
        axis (tuple): summarize over
        central (np.ufunc): function to get measure the center
            defaults to np.median
        ave (np.ufunc): function to average deviation over.
            defaults to np.mean

    Returns:
        np.ndarray : average_deviations

    References:
        https://en.wikipedia.org/wiki/Average_absolute_deviation

    Example:
        >>> x = np.array([[[0, 1], [3, 4]],
        >>>               [[0, 0], [0, 0]]])
        >>> axis = (0, 1)
        >>> absdev(x, np.mean, np.median, axis=(0, 1))
        array([ 0.75,  1.25])
        >>> absdev(x, np.median, np.median, axis=(0, 1))
        array([ 0. ,  0.5])
        >>> absdev(x, np.mean, np.median)
        1.0
        >>> absdev(x, np.median, np.median)
        0.0
        >>> absdev(x, np.median, np.median, axis=0)
        array([[ 0. ,  0.5],
               [ 1.5,  2. ]])
    """
    point = central(x, axis=axis, keepdims=True)
    deviations = np.abs(x - point)
    average_deviations = ave(deviations, axis=axis)
    return average_deviations


class RunningStats(object):
    """
    Dynamically records per-element array statistics and can summarized them
    per-element, across channels, or globally.

    SeeAlso:
        InternalRunningStats

    Example:
        >>> from clab.util.imutil import *
        >>> run = RunningStats()
        >>> ch1 = np.array([[0, 1], [3, 4]])
        >>> ch2 = np.zeros((2, 2))
        >>> img = np.dstack([ch1, ch2])
        >>> run.update(np.dstack([ch1, ch2]))
        >>> run.update(np.dstack([ch1 + 1, ch2]))
        >>> run.update(np.dstack([ch1 + 2, ch2]))
        >>> # Scalar averages
        >>> print(ub.repr2(run.simple(), nobr=1, si=True))
        max: 6.0,
        mean: 1.5,
        min: 0.0,
        n: 24,
        squares: 146.0,
        std: 2.0,
        total: 36.0,
        >>> # Per channel averages
        >>> print(ub.repr2(ub.map_vals(lambda x: np.array(x).tolist(), run.simple()), nobr=1, si=True, nl=1))
        mean: [3.0, 0.0],
        min: [0.0, 0.0],
        n: 12,
        squares: [146.0, 0.0],
        std: [1.8586407545691703, 0.0],
        total: [36.0, 0.0],
        >>> # Per-pixel averages
        >>> print(ub.repr2(ub.map_vals(lambda x: np.array(x).tolist(), run.detail()), nobr=1, si=True, nl=1))
        max: [[[2.0, 0.0], [3.0, 0.0]], [[5.0, 0.0], [6.0, 0.0]]],
        mean: [[[1.0, 0.0], [2.0, 0.0]], [[4.0, 0.0], [5.0, 0.0]]],
        min: [[[0.0, 0.0], [1.0, 0.0]], [[3.0, 0.0], [4.0, 0.0]]],
        n: 3,
        squares: [[[5.0, 0.0], [14.0, 0.0]], [[50.0, 0.0], [77.0, 0.0]]],
        std: [[[1.0, 0.0], [1.0, 0.0]], [[1.0, 0.0], [1.0, 0.0]]],
        total: [[[3.0, 0.0], [6.0, 0.0]], [[12.0, 0.0], [15.0, 0.0]]],
        """

    def __init__(run):
        run.raw_max = -np.inf
        run.raw_min = np.inf
        run.raw_total = 0
        run.raw_squares = 0
        run.n = 0

    def update(run, img):
        run.n += 1
        # Update stats across images
        run.raw_max = np.maximum(run.raw_max, img)
        run.raw_min = np.minimum(run.raw_min, img)
        run.raw_total += img
        run.raw_squares += img ** 2

    def _sumsq_std(run, total, squares, n):
        """
        Sum of squares method to compute standard deviation
        """
        numer = (n * squares - total ** 2)
        denom = (n * (n - 1))
        std = np.sqrt(numer / denom)
        return std

    def simple(run, axis=None):
        assert run.n > 0, 'no stats exist'
        maxi    = run.raw_max.max(axis=axis, keepdims=True)
        mini    = run.raw_min.min(axis=axis, keepdims=True)
        total   = run.raw_total.sum(axis=axis, keepdims=True)
        squares = run.raw_squares.sum(axis=axis, keepdims=True)
        if not hasattr(run.raw_total, 'shape'):
            n = run.n
        elif axis is None:
            n = run.n * np.prod(run.raw_total.shape)
        else:
            n = run.n * np.prod(np.take(run.raw_total.shape, axis))
        info = ub.odict([
            ('n', n),
            ('max', maxi),
            ('min', mini),
            ('total', total),
            ('squares', squares),
            ('mean', total / n),
            ('std', run._sumsq_std(total, squares, n)),
        ])
        return info

    def detail(run):
        total = run.raw_total
        squares = run.raw_squares
        maxi = run.raw_max
        mini = run.raw_min
        n = run.n
        info = ub.odict([
            ('n', n),
            ('max', maxi),
            ('min', mini),
            ('total', total),
            ('squares', squares),
            ('mean', total / n),
            ('std', run._sumsq_std(total, squares, n)),
        ])
        return info


class InternalRunningStats():
    """
    Maintains an averages of average internal statistics across a dataset.

    The difference between `RunningStats` and this is that the former can keep
    track of the average value of pixel (x, y) or channel (c) across the
    dataset, whereas this class tracks the average pixel value within an image
    across the dataset. So, this is an average of averages.

    Example:
        >>> from clab.util.imutil import *
        >>> ch1 = np.array([[0, 1], [3, 4]])
        >>> ch2 = np.zeros((2, 2))
        >>> img = np.dstack([ch1, ch2])
        >>> irun = InternalRunningStats(axis=(0, 1))
        >>> irun.update(np.dstack([ch1, ch2]))
        >>> irun.update(np.dstack([ch1 + 1, ch2]))
        >>> irun.update(np.dstack([ch1 + 2, ch2]))
        >>> # Scalar averages
        >>> print(ub.repr2(irun.info(), nobr=1, si=True))
    """

    def __init__(irun, axis=None):
        from functools import partial
        irun.axis = axis
        # Define a running stats object for each as well as the function to
        # compute the internal statistic
        irun.runs = ub.odict([
            ('mean', (
                RunningStats(), np.mean)),
            ('std', (
                RunningStats(), np.std)),
            ('median', (
                RunningStats(), np.median)),
            # ('mean_absdev_from_mean', (
            #     RunningStats(),
            #     partial(absdev, ave=np.mean, central=np.mean))),
            ('mean_absdev_from_median', (
                RunningStats(),
                partial(absdev, ave=np.mean, central=np.median))),
            ('median_absdev_from_median', (
                RunningStats(),
                partial(absdev, ave=np.median, central=np.median))),
        ])

    def update(irun, img):
        axis = irun.axis
        for run, func in irun.runs.values():
            stat = func(img, axis=axis)
            run.update(stat)

    def info(irun):
        return {
            key: run.detail() for key, (run, _) in irun.runs.items()
        }
