import numpy as np
TRANSFORM_DTYPE = np.float64


def rotation_mat3x3(radians, sin=np.sin, cos=np.cos):
    """
    References:
        https://en.wikipedia.org/wiki/Rotation_matrix
    """
    # TODO: handle array inputs
    sin_ = sin(radians)
    cos_ = cos(radians)
    R = np.array(((cos_, -sin_,  0),
                  (sin_,  cos_,  0),
                  (   0,     0,  1),))
    return R


def rotation_mat2x2(theta):
    sin_ = np.sin(theta)
    cos_ = np.cos(theta)
    rot_ = np.array(((cos_, -sin_),
                     (sin_,  cos_),))
    return rot_


def transform_around(M, x, y):
    """ translates to origin, applies transform and then translates back """
    tr1_ = translation_mat3x3(-x, -y)
    tr2_ = translation_mat3x3(x, y)
    M_ = tr2_.dot(M).dot(tr1_)
    return M_


def rotation_around_mat3x3(theta, x, y):
    # rot = rotation_mat3x3(theta)
    # return transform_around(rot, x, y)
    tr1_ = translation_mat3x3(-x, -y)
    rot_ = rotation_mat3x3(theta)
    tr2_ = translation_mat3x3(x, y)
    rot = tr2_.dot(rot_).dot(tr1_)
    return rot


def scale_around_mat3x3(sx, sy, x, y):
    scale_ = scale_mat3x3(sx, sy)
    return transform_around(scale_, x, y)


def rotation_around_bbox_mat3x3(theta, bbox):
    x, y, w, h = bbox
    centerx = x + (w / 2)
    centery = y + (h / 2)
    return rotation_around_mat3x3(theta, centerx, centery)


def translation_mat3x3(x, y, dtype=TRANSFORM_DTYPE):
    T = np.array([[1, 0,  x],
                  [0, 1,  y],
                  [0, 0,  1]], dtype=dtype)
    return T


def scale_mat3x3(sx, sy=None, dtype=TRANSFORM_DTYPE):
    sy = sx if sy is None else sy
    S = np.array([[sx, 0, 0],
                  [0, sy, 0],
                  [0,  0, 1]], dtype=dtype)
    return S


def shear_mat3x3(shear_x, shear_y, dtype=TRANSFORM_DTYPE):
    shear = np.array([[      1, shear_x, 0],
                      [shear_y,       1, 0],
                      [      0,       0, 1]], dtype=dtype)
    return shear


def affine_mat3x3(sx=1, sy=1, theta=0, shear=0, tx=0, ty=0, trig=np):
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
    sin1_ = trig.sin(theta)
    cos1_ = trig.cos(theta)
    sin2_ = trig.sin(theta + shear)
    cos2_ = trig.cos(theta + shear)
    Aff = np.array([
        [sx * cos1_, -sy * sin2_, tx],
        [sx * sin1_,  sy * cos2_, ty],
        [        0,            0,  1]
    ])
    return Aff


def affine_around_mat3x3(x, y, sx=1.0, sy=1.0, theta=0.0, shear=0.0, tx=0.0,
                         ty=0.0, x2=None, y2=None):
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
    # Make auxially varables to reduce the number of sin/cosine calls
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    cos_shear_p_theta = np.cos(shear + theta)
    sin_shear_p_theta = np.sin(shear + theta)
    tx_ = -sx * x * cos_theta + sy * y * sin_shear_p_theta + tx + x2
    ty_ = -sx * x * sin_theta - sy * y * cos_shear_p_theta + ty + y2
    # Sympy compiled expression
    Aff = np.array([[sx * cos_theta, -sy * sin_shear_p_theta, tx_],
                    [sx * sin_theta,  sy * cos_shear_p_theta, ty_],
                    [             0,                       0, 1]])
    return Aff


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m clab.util.util_affine
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
