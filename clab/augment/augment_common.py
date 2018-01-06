from __future__ import division, print_function
import numpy as np
from clab import util


# arguments good for small affine perterbations
PERTERB_AUG_KW = dict(zoom_pdf=(1 / 1.1, 1.1),
                      tx_pdf=(-12, 12),
                      ty_pdf=(-12, 12),
                      shear_pdf=(-np.pi / 32, np.pi / 32),
                      theta_pdf=(-np.pi / 16, np.pi / 16),
                      flip_lr_prob=.5,
                      flip_ud_prob=.5,
                      enable_stretch=True,
                      default_distribution='normal')


def random_affine_args(zoom_pdf=None,
                       tx_pdf=None,
                       ty_pdf=None,
                       shear_pdf=None,
                       theta_pdf=None,
                       flip_lr_prob=0,
                       flip_ud_prob=0,
                       enable_stretch=False,
                       default_distribution='uniform',
                       txy_pdf=None,
                       rng=np.random):
    r"""
    If pdfs are tuples it is interpreted as a default (uniform) distribution between the
    two points. A single scalar is a default distribution between -scalar and
    scalar.

    TODO: allow for a pdf of ranges for each dimension

    TODO: depricate for an imgaug type approach

    Args:
        zoom_pdf (tuple): (default = (1.0, 1.0))
        tx_pdf (tuple): (default = (0.0, 0.0))
        ty_pdf (tuple): (default = (0.0, 0.0))
        shear_pdf (tuple): in radians
        theta_pdf (tuple): in radians
        flip_lr_prob (float): probability of a left-right flip
        flip_ud_prob (float): probability of a up-down flip
        enable_stretch (bool): (default = False)
        rng (module):  random number generator(default = numpy.random)

    Returns:
        tuple: affine_args

    Example:
        >>> from clab.augment.augment_common import *
        >>> zoom_range = (0.9090909090909091, 1.1)
        >>> tx_pdf = (0.0, 4.0)
        >>> ty_pdf = (0.0, 4.0)
        >>> shear_pdf = (0, 0)
        >>> theta_pdf = (0, 0)
        >>> flip_lr_prob = .5
        >>> enable_stretch = False
        >>> rng = np.random.RandomState(0)
        >>> affine_args = random_affine_args(
        >>>     zoom_range, tx_pdf, ty_pdf, shear_pdf, theta_pdf,
        >>>     flip_lr_prob, enable_stretch, rng=rng)
        >>> import utool as ut
        >>> print('affine_args = %s' % (ut.repr2(affine_args),))
        >>> (sx, sy, theta, shear, tx, ty) = affine_args
        >>> Aff = affine_mat2x3(sx, sy, theta, shear, tx, ty)
        >>> result = ut.repr2(Aff)
        >>> print(result)
        np.array([[ 1.00934827, -0.        ,  1.6946192 ],
                  [ 0.        ,  1.0418724 ,  2.58357645]])

    Ignore:
        from clab.augment.augment_common import *
        import plottool as pt
        pt.qtensure()

        rng = np.random.RandomState(0)

        augkw = dict(zoom_pdf=(1 / 1.1, 1.1),
                     tx_pdf=(-1, 1),
                     ty_pdf=(-1, 1),
                     shear_pdf=(-np.pi / 32, np.pi / 32),
                     theta_pdf=(-np.pi, np.pi),
                     flip_lr_prob=.5,
                     flip_ud_prob=.5,
                     enable_stretch=True,
                     default_distribution='uniform')

        # augkw['zoom_pdf'] = None
        # augkw['shear_pdf'] = None
        # augkw['theta_pdf'] = None
        # augkw['tx_pdf'] = None
        # augkw['ty_pdf'] = None

        dx, dy = np.array([.5, -1]) * 100

        params = random_affine_args(**augkw)
        sx, sy, theta, shear, tx, ty = params

        print('sx, xy = {}, {}'.format(sx, sy))
        print('theta, shear = {}, {}'.format(theta, shear))
        print('tx, ty = {}, {}'.format(tx, ty))

        matrix = np.array(affine_around_mat2x3(cx, cy, *params) + [[0, 0, 1]])
        skaff = skimage.transform.AffineTransform(matrix=matrix)

        img = util.imread(ut.grab_test_imgpath('lena.png'))
        cx, cy = img.shape[1] / 2, img.shape[0] / 2
        img2 = skimage.transform.warp(
            img, skaff, output_shape=img.shape, order=3, mode='constant',
        )

        # dx_, dy_ = skaff.inverse([dx, dy])[0]
        vel_matrix = np.array(affine_mat2x3(*params) + [[0, 0, 1]])
        dx_, dy_ = np.linalg.inv(vel_matrix).dot([dx, dy, 1])[0:2]

        pt.imshow(img, pnum=(1, 2, 1), fnum=1)
        pt.plt.gca().arrow(*(cx, cy, dx, dy), width=5, length_includes_head=0)

        pt.imshow(img2, pnum=(1, 2, 2), fnum=1)
        pt.plt.gca().arrow(*(cx, cy, dx_, dy_), width=5, length_includes_head=0)


    """
    if zoom_pdf is None:
        sx = sy = 1.0
    else:
        if enable_stretch:
            sx = sy = rng.uniform(*zoom_pdf)
        else:
            sx = rng.uniform(*zoom_pdf)
            sy = rng.uniform(*zoom_pdf)

        if False:
            # why did I do this?
            log_zoom_range = [np.log(z) for z in zoom_pdf]

            if enable_stretch:
                sx = sy = np.exp(rng.uniform(*log_zoom_range))
            else:
                sx = np.exp(rng.uniform(*log_zoom_range))
                sy = np.exp(rng.uniform(*log_zoom_range))

    def param_distribution(param_pdf, rng=rng):
        if param_pdf is None:
            param = 0
        elif not util.isiterable(param_pdf):
            param = param_pdf
        elif isinstance(param_pdf, tuple):
            min_param, max_param = param_pdf
            if default_distribution == 'uniform':
                param = rng.uniform(min_param, max_param)
            elif default_distribution == 'normal':
                mean = (max_param + min_param) / 2
                std = (max_param - min_param) / 5
                param = np.clip(rng.normal(mean, std), min_param, max_param)
        else:
            assert False
        return param

    theta = param_distribution(theta_pdf)
    shear = param_distribution(shear_pdf)
    if txy_pdf is not None:
        assert tx_pdf is None, 'cannot specify both'
        assert ty_pdf is None, 'cannot specify both'
        xy_locs, xy_probs = txy_pdf
        tx = param_distribution(tx_pdf)
    else:
        tx = param_distribution(tx_pdf)
        ty = param_distribution(ty_pdf)

    def bernoulli_event(p):
        return p == 1 or (p != 0 and rng.rand() < p)

    """
    >>> from clab.augment.augment_common import *
    >>> theta = np.pi
    >>> shear = np.pi + np.pi
    >>> sx, sy, tx, ty = 1, 1, 0, 0
    >>> affine_args = (sx, sy, theta, shear, tx, ty)
    >>> M = affine_mat2x3(*affine_args)
    >>> print('M =\n{!r}'.format(M))
    affine_mat2x3(*random_affine_args(flip_ud_prob=1))
    affine_mat2x3(*random_affine_args(flip_lr_prob=1))
    affine_mat2x3(*random_affine_args(shear_pdf=.01))
    """

    # flip left-right some fraction of the time
    if bernoulli_event(flip_lr_prob):
        # shear 180 degrees + rotate 180 == lr-flip
        # print('FLIP LR')
        theta += np.pi
        shear += np.pi

    if bernoulli_event(flip_ud_prob):
        # print('FLIP UD')
        # shear 180 degrees == ud-flip
        shear += np.pi

    affine_args = (sx, sy, theta, shear, tx, ty)
    return affine_args


def affine_mat2x3(sx=1, sy=1, theta=0, shear=0, tx=0, ty=0, math=np):
    r"""
    Args:
        sx (float): x scale factor (default = 1)
        sy (float): y scale factor (default = 1)
        theta (float): rotation angle (radians) in counterclockwise direction
        shear (float): shear angle (radians) in counterclockwise directions
        tx (float): x-translation (default = 0)
        ty (float): y-translation (default = 0)

    References:
        https://github.com/scikit-image/scikit-image/blob/master/skimage/transform/_geometric.py
    """
    sin1_ = math.sin(theta)
    cos1_ = math.cos(theta)
    sin2_ = math.sin(theta + shear)
    cos2_ = math.cos(theta + shear)
    Aff = [
        [sx * cos1_, -sy * sin2_, tx],
        [sx * sin1_,  sy * cos2_, ty],
    ]
    return Aff


def affine_around_mat2x3(x, y, sx=1.0, sy=1.0, theta=0.0, shear=0.0, tx=0.0,
                         ty=0.0, flip_lr=False, flip_ud=False, x2=None,
                         y2=None, math=np):
    r"""
    Executes an affine transform around center point (x, y).
    Equivalent to translation.dot(affine).dot(inv(translation))

    Args:
        x (float): center x location in input space
        y (float):  center y location in input space
        sx (float): x scale factor (default = 1)
        sy (float): y scale factor (default = 1)
        theta (float): counter-clockwise rotation angle in radians(default = 0)
        shear (float): counter-clockwise shear angle in radians(default = 0)
        tx (float): x-translation (default = 0)
        ty (float): y-translation (default = 0)
        x2 (float, optional): center y location in output space (default = x)
        y2 (float, optional): center y location in output space (default = y)

    Example:
        >>> from vtool.linalg import *  # NOQA
        >>> import vtool as vt
        >>> orig_pts = np.array(vt.verts_from_bbox([10, 10, 20, 20]))
        >>> x, y = vt.bbox_center(vt.bbox_from_verts(orig_pts))
        >>> sx, sy = 0.5, 1.0
        >>> theta = 1 * np.pi / 4
        >>> shear = .1 * np.pi / 4
        >>> tx, ty = 5, 0
        >>> x2, y2 = None, None
        >>> Aff = affine_around_mat3x3(x, y, sx, sy, theta, shear,
        >>>                            tx, ty, x2, y2)
        >>> trans_pts = vt.transform_points_with_homography(Aff, orig_pts.T).T
        >>> import plottool as pt
        >>> pt.ensureqt()
        >>> pt.plt.plot(x, y, 'bx', label='center')
        >>> pt.plt.plot(orig_pts.T[0], orig_pts.T[1], 'b-', label='original')
        >>> pt.plt.plot(trans_pts.T[0], trans_pts.T[1], 'r-', label='transformed')
        >>> pt.plt.legend()
        >>> pt.plt.title('Demo of affine_around_mat3x3')
        >>> pt.plt.axis('equal')
        >>> pt.plt.xlim(0, 40)
        >>> pt.plt.ylim(0, 40)
        >>> ut.show_if_requested()

    Timeit:
        >>> from vtool.linalg import *  # NOQA
        >>> x, y, sx, sy, theta, shear, tx, ty, x2, y2 = (
        >>>     256.0, 256.0, 1.5, 1.0, 0.78, 0.2, 0, 100, 500.0, 500.0)
        >>> for timer in ut.Timerit(1000, 'old'):  # 19.0697 µs
        >>>     with timer:
        >>>         tr1_ = translation_mat3x3(-x, -y)
        >>>         Aff_ = affine_mat3x3(sx, sy, theta, shear, tx, ty)
        >>>         tr2_ = translation_mat3x3(x2, y2)
        >>>         Aff1 = tr2_.dot(Aff_).dot(tr1_)
        >>> for timer in ut.Timerit(1000, 'new'):  # 11.0242 µs
        >>>     with timer:
        >>>         Aff2 = affine_around_mat3x3(x, y, sx, sy, theta, shear,
        >>>                                     tx, ty, x2, y2)
        >>> assert np.all(np.isclose(Aff2, Aff1))

    Sympy:
        >>> from vtool.linalg import *  # NOQA
        >>> import vtool as vt
        >>> import sympy
        >>> # Shows the symbolic construction of the code
        >>> # https://groups.google.com/forum/#!topic/sympy/k1HnZK_bNNA
        >>> from sympy.abc import theta
        >>> x, y, sx, sy, theta, shear, tx, ty, x2, y2 = sympy.symbols(
        >>>     'x, y, sx, sy, theta, shear, tx, ty, x2, y2')
        >>> theta = sx = sy = tx = ty = 0
        >>> # move to center xy, apply affine transform, move center xy2
        >>> tr1_ = translation_mat3x3(-x, -y, dtype=None)
        >>> Aff_ = affine_mat3x3(sx, sy, theta, shear, tx, ty, trig=sympy)
        >>> tr2_ = translation_mat3x3(x2, y2, dtype=None)
        >>> # combine transformations
        >>> Aff = vt.sympy_mat(tr2_.dot(Aff_).dot(tr1_))
        >>> vt.evalprint('Aff')
        >>> print('-------')
        >>> print('Numpy')
        >>> vt.sympy_numpy_repr(Aff)
    """
    x2 = x if x2 is None else x2
    y2 = y if y2 is None else y2

    if flip_lr:
        # shear 180 degrees + rotate 180 == lr-flip
        theta += np.pi
        shear += np.pi

    if flip_ud:
        # shear 180 degrees == ud-flip
        shear += np.pi

    if math == 'skimage':
        import skimage.transform
        T1 = skimage.transform.AffineTransform(translation=(-x, -y))
        A = skimage.transform.AffineTransform(scale=(sx, sy), rotation=theta,
                                               shear=shear,
                                               translation=(tx, ty))
        T2 = skimage.transform.AffineTransform(translation=(x2, y2))
        M = T1 + A + T2
        return M.params[0:2].tolist()
    else:
        # Make auxially varables to reduce the number of sin/cosine calls
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)
        cos_shear_p_theta = math.cos(shear + theta)
        sin_shear_p_theta = math.sin(shear + theta)
        tx_ = -sx * x * cos_theta + sy * y * sin_shear_p_theta + tx + x2
        ty_ = -sx * x * sin_theta - sy * y * cos_shear_p_theta + ty + y2
        # Sympy compiled expression
        Aff = [[sx * cos_theta, -sy * sin_shear_p_theta, tx_],
               [sx * sin_theta,  sy * cos_shear_p_theta, ty_]]
        return Aff
