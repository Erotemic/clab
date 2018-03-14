# -*- coding: utf-8 -*-
from six.moves import zip, reduce
import cv2
import numpy as np
from numpy.core.umath_tests import matrix_multiply  # NOQA
from clab.util import util_affine
from clab.util import nputil


def make_kpts_heatmask(kpts, chipsize, cmap='plasma'):
    """
    makes a heatmap overlay for keypoints

    Ignore:
        >>> img_fpath = util.grab_test_imgpath('carl.png')
        >>> (kpts, vecs) = detect_feats(img_fpath)
        >>> chip = vt.imread(img_fpath)
        >>> kpts = kpts[0:100]
        >>> chipsize = chip.shape[0:2][::-1]
        >>> heatmask = make_kpts_heatmask(kpts, chipsize)
        >>> img1 = heatmask
        >>> img2 = chip
        >>> # xdoc: +REQUIRES(--show)
        >>> from clab.util import mplutil
        >>> mplutil.qtensure()
        >>> img3 = vt.overlay_alpha_images(heatmask, chip)
        >>> mplutil.imshow(img3)
        >>> #mplutil.imshow(heatmask)
        >>> #mplutil.draw_kpts2(kpts)
        >>> mplutil.show_if_requested()
    """
    # use a disk instead of a gaussian
    import skimage.morphology
    from matplotlib import pyplot as plt
    cov_scale_factor = .25
    radius = min(int((min(chipsize) * cov_scale_factor) // 2) - 1, 50)
    patch = skimage.morphology.disk(radius)
    mask = make_kpts_coverage_mask(kpts, chipsize, resize=True,
                                   cov_size_penalty_on=False,
                                   patch=patch,
                                   cov_scale_factor=cov_scale_factor,
                                   cov_blur_sigma=1.5,
                                   cov_blur_on=True)
    # heatmask = np.ones(tuple(chipsize) + (4,)) * mplutil.RED
    heatmask = plt.get_cmap(cmap)(mask)
    # conver to bgr
    heatmask[:, :, 0:3] = heatmask[:, :, 0:3][:, :, ::-1]
    # apply alpha channel
    heatmask[:, :, 3] = mask * .5
    return heatmask


def make_kpts_coverage_mask(
        kpts, chipsize,
        weights=None,
        return_patch=False,
        patch=None,
        resize=False,
        out=None,
        cov_blur_on=True,
        cov_disk_hack=None,
        cov_blur_ksize=(17, 17),
        cov_blur_sigma=5.0,
        cov_gauss_shape=(19, 19),
        cov_gauss_sigma_frac=.3,
        cov_scale_factor=.2,
        cov_agg_mode='max',
        cov_remove_shape=False,
        cov_remove_scale=False,
        cov_size_penalty_on=True,
        cov_size_penalty_power=.5,
        cov_size_penalty_frac=.1):
    r"""
    Returns a intensity image denoting which pixels are covered by the input
    keypoints

    Args:
        kpts (ndarray[float32_t, ndim=2][ndims=2]):  keypoints
        chipsize (tuple): width height of the underlying image

    Returns:
        tuple (ndarray, ndarray): dstimg, patch

    Ignore:
        >>> from clab.util import mplutil
        >>> img_fpath = util.grab_test_imgpath('carl.png')
        >>> (kpts, vecs) = detect_feats(img_fpath)
        >>> kpts = kpts[::10]
        >>> chip = vt.imread(img_fpath)
        >>> chipsize = chip.shape[0:2][::-1]
        >>> # execute function
        >>> dstimg, patch = make_kpts_coverage_mask(kpts, chipsize, resize=True, return_patch=True, cov_size_penalty_on=False, cov_blur_on=False)
        >>> # show results
        >>> # xdoc: +REQUIRES(--show)
        >>> mask = dstimg
        >>> show_coverage_map(chip, mask, patch, kpts)
        >>> mplutil.show_if_requested()
    """
    if patch is None:
        patch = _get_gaussian_weight_patch(cov_gauss_shape, cov_gauss_sigma_frac)
    chipshape = chipsize[::-1]
    # Warp patches onto a scaled image
    dstimg = _warp_patch_onto_kpts(
        kpts, patch, chipshape, weights=weights, out=out,
        cov_scale_factor=cov_scale_factor,
        cov_agg_mode=cov_agg_mode,
        cov_remove_shape=cov_remove_shape,
        cov_remove_scale=cov_remove_scale,
        cov_size_penalty_on=cov_size_penalty_on,
        cov_size_penalty_power=cov_size_penalty_power,
        cov_size_penalty_frac=cov_size_penalty_frac
    )
    # Smooth weight of influence
    if cov_blur_on:
        cv2.GaussianBlur(dstimg, ksize=cov_blur_ksize, sigmaX=cov_blur_sigma,
                         sigmaY=cov_blur_sigma, dst=dstimg,
                         borderType=cv2.BORDER_CONSTANT)
    if resize:
        # Resize to original chpsize of requested
        dsize = chipsize
        dstimg = cv2.resize(dstimg, dsize)
    if return_patch:
        return dstimg, patch
    else:
        return dstimg


def _warp_patch_onto_kpts(
        kpts, patch, chipshape,
        weights=None,
        out=None,
        cov_scale_factor=.2,
        cov_agg_mode='max',
        cov_remove_shape=False,
        cov_remove_scale=False,
        cov_size_penalty_on=True,
        cov_size_penalty_power=.5,
        cov_size_penalty_frac=.1):
    r"""
    Overlays the source image onto a destination image in each keypoint location

    Args:
        kpts (ndarray[float32_t, ndim=2]):  keypoints
        patch (ndarray): patch to warp (like gaussian)
        chipshape (tuple):
        weights (ndarray): score for every keypoint

    Kwargs:
        cov_scale_factor (float):

    Returns:
        ndarray: mask

    Ignore:
        >>> img_fpath    = util.grab_test_imgpath('carl.jpg')
        >>> (kpts, vecs) = detect_feats(img_fpath)
        >>> kpts = kpts[::15]
        >>> chip = vt.imread(img_fpath)
        >>> chipshape = chip.shape
        >>> weights = np.ones(len(kpts))
        >>> cov_scale_factor = 1.0
        >>> srcshape = (19, 19)
        >>> radius = srcshape[0] / 2.0
        >>> sigma = 0.4 * radius
        >>> SQUARE = ub.argflag('--square')
        >>> HOLE = ub.argflag('--hole')
        >>> if SQUARE:
        >>>     patch = np.ones(srcshape)
        >>> else:
        >>>     patch = gaussian_patch(shape=srcshape, sigma=sigma) #, norm_01=False)
        >>>     patch = patch / patch.max()
        >>> if HOLE:
        >>>     patch[int(patch.shape[0] / 2), int(patch.shape[1] / 2)] = 0
        >>> # execute function
        >>> dstimg = _warp_patch_onto_kpts(kpts, patch, chipshape, weights, cov_scale_factor=cov_scale_factor)
        >>> print('dstimg stats %r' % (ut.get_stats_str(dstimg, axis=None)),)
        >>> print('patch stats %r' % (ut.get_stats_str(patch, axis=None)),)
        >>> #print(patch.sum())
        >>> assert np.all(ut.inbounds(dstimg, 0, 1, eq=True))
        >>> # show results
        >>> # xdoc: +REQUIRES(--show)
        >>> from clab.util import mplutil
        >>> mask = dstimg
        >>> show_coverage_map(chip, mask, patch, kpts)
        >>> mplutil.show_if_requested()
    """
    chip_scale_h = int(np.ceil(chipshape[0] * cov_scale_factor))
    chip_scale_w = int(np.ceil(chipshape[1] * cov_scale_factor))
    if len(kpts) == 0:
        dstimg =  np.zeros((chip_scale_h, chip_scale_w))
        return dstimg
    if weights is None:
        weights = np.ones(len(kpts))
    dsize = (chip_scale_w, chip_scale_h)
    # Allocate destination image
    patch_shape = patch.shape
    # Scale keypoints into destination image
    # <HACK>
    if cov_remove_shape:
        # disregard affine information in keypoints
        # i still dont understand why we are trying this
        (patch_h, patch_w) = patch_shape
        half_width  = (patch_w / 2.0)  # - .5
        half_height = (patch_h / 2.0)  # - .5
        # Center src image
        T1 = util_affine.translation_mat3x3(-half_width + .5, -half_height + .5)
        # Scale src to the unit circle
        if not cov_remove_scale:
            S1 = util_affine.scale_mat3x3(1.0 / half_width, 1.0 / half_height)
        # Transform the source image to the keypoint ellipse
        kpts_T = np.array([util_affine.translation_mat3x3(x, y) for (x, y) in _get_xys(kpts).T])
        if not cov_remove_scale:
            kpts_S = np.array([util_affine.scale_mat3x3(np.sqrt(scale))
                               for scale in util_affine.get_scales(kpts).T])
        # Adjust for the requested scale factor
        S2 = util_affine.scale_mat3x3(cov_scale_factor, cov_scale_factor)
        #perspective_list = [S2.dot(A).dot(S1).dot(T1) for A in invVR_aff2Ds]
        if not cov_remove_scale:
            M_list = reduce(matrix_multiply, (S2, kpts_T, kpts_S, S1, T1))
        else:
            M_list = reduce(matrix_multiply, (S2, kpts_T, T1))
    # </HACK>
    else:
        M_list = _get_transforms_from_patch_image_kpts(kpts, patch_shape,
                                                       cov_scale_factor)
    affmat_list = M_list[:, 0:2, :]
    weight_list = weights
    # For each keypoint warp a gaussian scaled by the feature score into the image
    warped_patch_iter = _warped_patch_generator(
        patch, dsize, affmat_list, weight_list,
        cov_size_penalty_on=cov_size_penalty_on,
        cov_size_penalty_power=cov_size_penalty_power,
        cov_size_penalty_frac=cov_size_penalty_frac)
    # Either max or sum
    if cov_agg_mode == 'max':
        dstimg = nputil.iter_reduce_ufunc(np.maximum, warped_patch_iter, out=out)
    elif cov_agg_mode == 'sum':
        dstimg = nputil.iter_reduce_ufunc(np.add, warped_patch_iter, out=out)
        # HACK FOR SUM: DO NOT DO THIS FOR MAX
        dstimg[dstimg > 1.0] = 1.0
    else:
        raise AssertionError('Unknown cov_agg_mode=%r' % (cov_agg_mode,))
    return dstimg


def _get_transforms_from_patch_image_kpts(kpts, patch_shape, scale_factor=1.0):
    """
    Given some patch (like a gaussian patch) transforms a patch to be overlayed
    on top of each keypoint in the image (adjusted for a scale factor)

    Args:
        kpts (ndarray[float32_t, ndim=2]):  keypoints
        patch_shape (?):
        scale_factor (float):

    Returns:
        M_list: a list of 3x3 tranformation matricies for each keypoint

    Ignore:
        >>> kpts = vt.dummy.get_dummy_kpts()
        >>> patch_shape = (7, 7)
        >>> scale_factor = 1.0
        >>> M_list = _get_transforms_from_patch_image_kpts(kpts, patch_shape, scale_factor)

    Ignore:
        >>> kpts = vt.dummy.get_dummy_kpts()
        >>> invVR_aff2Ds = [np.array(((a, 0, x),
        >>>                           (c, d, y),
        >>>                           (0, 0, 1),))
        >>>                 for (x, y, a, c, d, ori) in kpts]
        >>> invVR_3x3 = vt._get_invVR_mats3x3(kpts)
        >>> invV_3x3 = vt.get_invV_mats3x3(kpts)
        >>> assert np.all(np.array(invVR_aff2Ds) == invVR_3x3)
        >>> assert np.all(np.array(invVR_aff2Ds) == invV_3x3)

    Timeit:
        %timeit [np.array(((a, 0, x), (c, d, y), (0, 0, 1),)) for (x, y, a, c, d, ori) in kpts]
        %timeit vt._get_invVR_mats3x3(kpts)
        %timeit vt.get_invV_mats3x3(kpts) <- THIS IS ACTUALLY MUCH FASTER

    Ignore::
        %pylab qt4
        from clab.util import mplutil
        mplutil.imshow(chip)
        mplutil.draw_kpts2(kpts)
        mplutil.update()

    Timeit:
        sa_list1 = np.array([S2.dot(A) for A in invVR_aff2Ds])
        sa_list2 = matrix_multiply(S2, invVR_aff2Ds)
        assert np.all(sa_list1 == sa_list2)
        %timeit np.array([S2.dot(A) for A in invVR_aff2Ds])
        %timeit matrix_multiply(S2, invVR_aff2Ds)

        from six.moves import reduce
        perspective_list2 = np.array([S2.dot(A).dot(S1).dot(T1) for A in invVR_aff2Ds])
        perspective_list = reduce(matrix_multiply, (S2, invVR_aff2Ds, S1, T1))
        assert np.all(perspective_list == perspective_list2)
        %timeit np.array([S2.dot(A).dot(S1).dot(T1) for A in invVR_aff2Ds])
        %timeit reduce(matrix_multiply, (S2, invVR_aff2Ds, S1, T1))
    """
    (patch_h, patch_w) = patch_shape
    half_width  = (patch_w / 2.0)  # - .5
    half_height = (patch_h / 2.0)  # - .5
    # Center src image
    T1 = util_affine.translation_mat3x3(-half_width + .5, -half_height + .5)
    # Scale src to the unit circle
    #S1 = util_affine.scale_mat3x3(1.0 / patch_w, 1.0 / patch_h)
    S1 = util_affine.scale_mat3x3(1.0 / half_width, 1.0 / half_height)
    # Transform the source image to the keypoint ellipse
    invVR_aff2Ds = _get_invVR_mats3x3(kpts)
    # Adjust for the requested scale factor
    S2 = util_affine.scale_mat3x3(scale_factor, scale_factor)
    #perspective_list = [S2.dot(A).dot(S1).dot(T1) for A in invVR_aff2Ds]
    M_list = reduce(matrix_multiply, (S2, invVR_aff2Ds, S1.dot(T1)))
    return M_list


def _get_invVR_mats3x3(kpts):
    r"""
    NEWER FUNCTION

    Returns full keypoint transform matricies from a unit circle to an
    ellipse that has been rotated, scaled, skewed, and translated. Into
    the image keypoint position.

    Args:
        kpts (ndarray[float32_t, ndim=2]):  keypoints

    Returns:
        ndarray[float32_t, ndim=3]: invVR_mats

    Example:
        >>> kpts = np.array([
        ...    [10, 20, 1, 2, 3, 0],
        ...    [30, 40, 1, 2, 3, np.pi / 2.0],
        ... ])
        >>> invVR_mats3x3 = _get_invVR_mats3x3(kpts)
    """
    #nKpts = len(kpts)
    invVR_mats2x2 = _get_invVR_mats2x2(kpts)
    invVR_mats3x3 = _augment_2x2_with_translation(kpts, invVR_mats2x2)
    # Unpack shape components
    #_iv11s = invVR_mats2x2.T[0, 0]
    #_iv12s = invVR_mats2x2.T[1, 0]
    #_iv21s = invVR_mats2x2.T[0, 1]
    #_iv22s = invVR_mats2x2.T[1, 1]
    ## Get translation components
    #_iv13s, _iv23s = _get_xys(kpts)
    ## Use homogenous coordinates
    #_zeros = np.zeros(nKpts)
    #_ones = np.ones(nKpts)
    #invVR_arrs =  np.array([[_iv11s, _iv12s, _iv13s],
    #                        [_iv21s, _iv22s, _iv23s],
    #                        [_zeros, _zeros,  _ones]])  # R x C x N
    #invVR_mats = np.rollaxis(invVR_arrs, 2)  # N x R x C
    return invVR_mats3x3


def _get_invVR_mats2x2(kpts):
    r"""
    Returns the keypoint shape+rotation matrix (from unit circle to ellipse)
    Ignores translation component

    Args:
        kpts (ndarray[float32_t, ndim=2][ndims=2]):  keypoints

    Returns:
        ndarray: invVR_mats

    Example:
        >>> kpts = np.array([
        ...    [0, 0, 1, 2, 3, 0],
        ...    [0, 0, 1, 2, 3, np.pi / 2.0],
        ... ])
        >>> invVR_mats2x2 = _get_invVR_mats2x2(kpts)

    Example:
        >>> kpts = np.empty((0, 6))
        >>> invVR_mats2x2 = _get_invVR_mats2x2(kpts)
        >>> assert invVR_mats2x2.shape == (0, 2, 2)
    """
    if len(kpts) == 0:
        return np.empty((0, 2, 2))
    invV_mats2x2 = _get_invV_mats2x2(kpts)
    # You must apply rotations before you apply shape
    # This is because we are dealing with \emph{inv}(V).
    # numpy operates with data on the right (operate right-to-left)
    R_mats2x2  = _get_ori_mats(kpts)
    invVR_mats2x2 = matrix_multiply(invV_mats2x2, R_mats2x2)
    return invVR_mats2x2


def _get_ori_mats(kpts):
    """ Returns keypoint orientation matrixes """
    _oris = _get_oris(kpts)
    R_mats = [util_affine.rotation_mat2x2(ori)
              for ori in _oris]
    return R_mats


def _get_invV_mats2x2(kpts):
    """
    Returns the keypoint shape (from unit circle to ellipse)
    Ignores translation and rotation component

    Args:
        kpts (ndarray[float32_t, ndim=2]):  keypoints

    Returns:
        ndarray[float32_t, ndim=3]: invV_mats

    Example:
        >>> kpts = np.array([
        ...    [0, 0, 1, 2, 3, 0],
        ...    [0, 0, 1, 2, 3, np.pi / 2.0],
        ... ])
        >>> invV_mats2x2 = _get_invV_mats2x2(kpts)
    """
    nKpts = len(kpts)
    _iv11s, _iv21s, _iv22s = _get_invVs(kpts)
    _zeros = np.zeros(nKpts)
    invV_arrs2x2 = np.array([[_iv11s, _zeros],
                             [_iv21s, _iv22s]])  # R x C x N
    invV_mats2x2 = np.rollaxis(invV_arrs2x2, 2)  # N x R x C
    return invV_mats2x2


def _augment_2x2_with_translation(kpts, _mat2x2):
    """
    helper function to augment shape matrix with a translation component.
    """
    nKpts = len(kpts)
    # Unpack shape components
    _11s = _mat2x2.T[0, 0]
    _12s = _mat2x2.T[1, 0]
    _21s = _mat2x2.T[0, 1]
    _22s = _mat2x2.T[1, 1]
    # Get translation components
    _13s, _23s = _get_xys(kpts)
    # Use homogenous coordinates
    _zeros = np.zeros(nKpts)
    _ones = np.ones(nKpts)
    _arrs3x3 =  np.array([[_11s, _12s, _13s],
                          [_21s, _22s, _23s],
                          [_zeros, _zeros,  _ones]])  # R x C x N
    _mats3x3 = np.rollaxis(_arrs3x3, 2)  # N x R x C
    return _mats3x3


def _warped_patch_generator(
        patch, dsize, affmat_list, weight_list,
        cov_size_penalty_on=True,
        cov_size_penalty_power=.5,
        cov_size_penalty_frac=.1):
    """
    generator that warps the patches (like gaussian) onto an image with dsize
    using constant memory.

    output must be used or copied on every iteration otherwise the next output
    will clobber the previous

    References:
        http://docs.opencv.org/modules/imgproc/doc/geometric_transformations.html#warpaffine
    """
    shape = dsize[::-1]
    #warpAffine is weird. If the shape of the dst is the same as src we can
    #use the dst outvar. I dont know why it needs that.  It seems that this
    #will not operate in place even if a destination array is passed in when
    #src.shape != dst.shape.
    patch_h, patch_w = patch.shape
    # If we pad the patch we can use dst
    padded_patch = np.zeros(shape, dtype=np.float32)
    # Prealloc output,
    warped = np.zeros(shape, dtype=np.float32)
    prepad_h, prepad_w = patch.shape[0:2]
    # each score is spread across its contributing pixels
    for (M, weight) in zip(affmat_list, weight_list):
        # inplace weighting of the patch
        np.multiply(patch, weight, out=padded_patch[:prepad_h, :prepad_w] )
        # inplace warping of the padded_patch
        cv2.warpAffine(padded_patch, M, dsize, dst=warped,
                       flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                       borderValue=0)
        if cov_size_penalty_on:
            # TODO: size penalty should be based of splitting number of
            # bins in a keypoint over the region that it covers
            total_weight = (warped.sum() ** cov_size_penalty_power) * cov_size_penalty_frac
            if total_weight > 1:
                # Whatever the size of the keypoint is it should
                # contribute a total of 1 score
                np.divide(warped, total_weight, out=warped)
        yield warped


def _get_gaussian_weight_patch(gauss_shape=(19, 19), gauss_sigma_frac=.3,
                               gauss_norm_01=True):
    r"""
    2d gaussian image useful for plotting

    Returns:
        ndarray: patch

    Example:
        >>> patch = _get_gaussian_weight_patch()
        >>> result = str(patch)
        >>> print(result)
    """
    # Perdoch uses roughly .95 of the radius
    radius = gauss_shape[0] / 2.0
    sigma = gauss_sigma_frac * radius
    # Similar to SIFT's computeCircularGaussMask in helpers.cpp
    # uses smmWindowSize=19 in hesaff for patch size. and 1.6 for sigma
    # Create gaussian image to warp
    patch = gaussian_patch(shape=gauss_shape, sigma=sigma)
    if gauss_norm_01:
        np.divide(patch, patch.max(), out=patch)
    return patch


# def get_coverage_kpts_gridsearch_configs():
#     """ testing function """
#     varied_dict = {
#         'cov_agg_mode'           : ['max', 'sum'],
#         #'cov_blur_ksize'         : [(19, 19), (5, 5)],
#         'cov_blur_ksize'         : [(5, 5)],
#         'cov_blur_on'            : [True, False],
#         'cov_blur_sigma'         : [5.0],
#         'cov_remove_scale'       : [True],
#         'cov_remove_shape'       : [False, True],
#         'cov_scale_factor'       : [.3],
#         'cov_size_penalty_frac'  : [.1],
#         'cov_size_penalty_on'    : [True],
#         'cov_size_penalty_power' : [.5],
#     }
#     slice_dict = {
#         'cov_scale_factor' : slice(0, 3),
#         'cov_agg_mode'     : slice(0, 2),
#         'cov_blur_ksize'   : slice(0, 2),
#         #'grid_sigma'        : slice(0, 4),
#     }
#     slice_dict = None
#     # Make configuration for every parameter setting
#     def constrain_func(cfgdict):
#         if cfgdict['cov_remove_shape']:
#             cfgdict['cov_remove_scale'] = False
#             cfgdict['cov_size_penalty_on'] = False
#         if not cfgdict['cov_size_penalty_on']:
#             cfgdict['cov_size_penalty_power'] = None
#             cfgdict['cov_size_penalty_frac'] = None
#         if not cfgdict['cov_blur_on']:
#             cfgdict['cov_blur_ksize'] = None
#             cfgdict['cov_blur_sigma'] = None
#         return cfgdict
#     cfgdict_list, cfglbl_list = ut.make_constrained_cfg_and_lbl_list(varied_dict, constrain_func, slice_dict)
#     return cfgdict_list, cfglbl_list


# def gridsearch_kpts_coverage_mask():
#     """
#     testing function

#     CommandLine:
#     Example:
#         >>> # DISABLE_DOCTEST
#         >>> from clab.util import mplutil
#         >>> gridsearch_kpts_coverage_mask()
#         >>> mplutil.show_if_requested()
#     """
#     from clab.util import mplutil
#     cfgdict_list, cfglbl_list = get_coverage_kpts_gridsearch_configs()
#     kpts, chipsize, weights = testdata_coverage('easy1.png')
#     imgmask_list = [
#         255 *  make_kpts_coverage_mask(kpts, chipsize, weights,
#                                        return_patch=False, **cfgdict)
#         for cfgdict in ub.Progter(cfgdict_list, lbl='coverage grid')
#     ]
#     #NORMHACK = True
#     #if NORMHACK:
#     #    imgmask_list = [
#     #        255 * (mask / mask.max()) for mask in imgmask_list
#     #    ]
#     fnum = mplutil.next_fnum()
    #     mplutil.imshow((mask * 255).astype(np.uint8), fnum=fnum, pnum=pnum_(1), title='mask')
    # else:
    #     mplutil.imshow((mask * 255).astype(np.uint8), fnum=fnum, pnum=(2, 1, 1), title='mask')
    # if show_mask_kpts:
    #     mplutil.draw_kpts2(kpts, rect=True, ell_alpha=ell_alpha)
    # mplutil.imshow(chip, fnum=fnum, pnum=pnum_(2), title='chip')
    # mplutil.draw_kpts2(kpts, rect=True, ell_alpha=ell_alpha)
    # masked_chip = (chip * mask[:, :, None]).astype(np.uint8)
    # mplutil.imshow(masked_chip, fnum=fnum, pnum=pnum_(3), title='masked chip')
    # #mplutil.draw_kpts2(kpts)


def gaussian_patch(shape=(7, 7), sigma=1.0):
    """
    another version of the guassian_patch function. hopefully better

    References:
        http://docs.opencv.org/modules/imgproc/doc/filtering.html#getgaussiankernel

    Args:
        shape (tuple):  array dimensions
        sigma (float):

    CommandLine:
        python -m clab.util.coverage_kpts gaussian_patch --show

    Example:
        >>> shape = (24, 24)
        >>> sigma = None  # 1.0
        >>> gausspatch = gaussian_patch(shape, sigma)
        >>> sum_ = gausspatch.sum()
        >>> assert np.all(np.isclose(sum_, 1.0))
        >>> # xdoc: +REQUIRES(--show)
        >>> from clab.util import mplutil
        >>> norm = (gausspatch - gausspatch.min()) / (gausspatch.max() - gausspatch.min())
        >>> mplutil.imshow(norm)
        >>> mplutil.show_if_requested()
    """
    if sigma is None:
        sigma = 0.3 * ((min(shape) - 1) * 0.5 - 1) + 0.8
    if isinstance(sigma, (float)):
        sigma1 = sigma2 = sigma
    else:
        sigma1, sigma2 = sigma
    # see hesaff/src/helpers.cpp : computeCircularGaussMask
    # HACK MAYBE: I think sigma is actually a sigma squared term?
    #sigma1 = np.sqrt(sigma1)
    #sigma2 = np.sqrt(sigma2)
    gauss_kernel_d0 = (cv2.getGaussianKernel(shape[0], sigma1))
    gauss_kernel_d1 = (cv2.getGaussianKernel(shape[1], sigma2))
    gausspatch = gauss_kernel_d0.dot(gauss_kernel_d1.T)
    return gausspatch


# --- raw keypoint components ---
def _get_xys(kpts):
    """ Keypoint locations in chip space """
    _xys = kpts.T[0:2]
    return _xys


def _get_invVs(kpts):
    """ Keypoint shapes (oriented with the gravity vector) """
    _invVs = kpts.T[2:5]
    return _invVs


def _get_oris(kpts):
    """ Extracts keypoint orientations for kpts array

    (in isotropic guassian space relative to the gravity vector)
    (in simpler words: the orientation is is taken from keypoints warped to the unit circle)

    Args:
        kpts (ndarray): (N x 6) [x, y, a, c, d, theta]

    Returns:
        (ndarray) theta
    """
    if kpts.shape[1] == 5:
        _oris = np.zeros(len(kpts), dtype=kpts.dtype)
    elif kpts.shape[1] == 6:
        _oris = kpts.T[5]
    else:
        raise AssertionError('[ktool] Invalid kpts.shape = %r' % (kpts.shape,))
    return _oris


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m clab.util.coverage_kpts
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
