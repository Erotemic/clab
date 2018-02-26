import numpy as np
import scipy.signal


def iter_reduce_ufunc(ufunc, arr_iter, out=None):
    """
    constant memory iteration and reduction

    applys ufunc from left to right over the input arrays

    Example:
        >>> # ENABLE_DOCTEST
        >>> arr_list = [
        ...     np.array([0, 1, 2, 3, 8, 9]),
        ...     np.array([4, 1, 2, 3, 4, 5]),
        ...     np.array([0, 5, 2, 3, 4, 5]),
        ...     np.array([1, 1, 6, 3, 4, 5]),
        ...     np.array([0, 1, 2, 7, 4, 5])
        ... ]
        >>> memory = np.array([9, 9, 9, 9, 9, 9])
        >>> gen_memory = memory.copy()
        >>> def arr_gen(arr_list, gen_memory):
        ...     for arr in arr_list:
        ...         gen_memory[:] = arr
        ...         yield gen_memory
        >>> print('memory = %r' % (memory,))
        >>> print('gen_memory = %r' % (gen_memory,))
        >>> ufunc = np.maximum
        >>> res1 = iter_reduce_ufunc(ufunc, iter(arr_list), out=None)
        >>> res2 = iter_reduce_ufunc(ufunc, iter(arr_list), out=memory)
        >>> res3 = iter_reduce_ufunc(ufunc, arr_gen(arr_list, gen_memory), out=memory)
        >>> print('res1       = %r' % (res1,))
        >>> print('res2       = %r' % (res2,))
        >>> print('res3       = %r' % (res3,))
        >>> print('memory     = %r' % (memory,))
        >>> print('gen_memory = %r' % (gen_memory,))
        >>> assert np.all(res1 == res2)
        >>> assert np.all(res2 == res3)
    """
    # Get first item in iterator
    try:
        initial = next(arr_iter)
    except StopIteration:
        return None
    # Populate the outvariable if specified otherwise make a copy of the first
    # item to be the output memory
    if out is not None:
        out[:] = initial
    else:
        out = initial.copy()
    # Iterate and reduce
    for arr in arr_iter:
        ufunc(out, arr, out=out)
    return out


def isect_flags(arr, other):
    """
    Example:
        >>> arr = np.array([
        >>>     [1, 2, 3, 4],
        >>>     [5, 6, 3, 4],
        >>>     [1, 1, 3, 4],
        >>> ])
        >>> other = np.array([1, 4, 6])
        >>> mask = isect_flags(arr, other)
        >>> print(mask)
        [[ True False False  True]
         [False  True False  True]
         [ True  True False  True]]
    """
    flags = iter_reduce_ufunc(np.logical_or, (arr == item for item in other)).ravel()
    flags.shape = arr.shape
    return flags


def atleast_nd(arr, n, tofront=False):
    r"""
    View inputs as arrays with at least n dimensions.
    TODO: Submit as a PR to numpy

    Args:
        arr (array_like): One array-like object.  Non-array inputs are
                converted to arrays.  Arrays that already have n or more
                dimensions are preserved.
        n (int): number of dimensions to ensure
        tofront (bool): if True new dimensions are added to the front of the
            array.  otherwise they are added to the back.

    Returns:
        ndarray :
            An array with ``a.ndim >= n``.  Copies are avoided where possible,
            and views with three or more dimensions are returned.  For example,
            a 1-D array of shape ``(N,)`` becomes a view of shape
            ``(1, N, 1)``, and a 2-D array of shape ``(M, N)`` becomes a view
            of shape ``(M, N, 1)``.

    See Also:
        ensure_shape, np.atleast_1d, np.atleast_2d, np.atleast_3d

    Example:
        >>> n = 2
        >>> arr = np.array([1, 1, 1])
        >>> arr_ = atleast_nd(arr, n)
        >>> result = ub.repr2(arr_.tolist())
        >>> print(result)
        [[1], [1], [1]]

    Example:
        >>> n = 4
        >>> arr1 = [1, 1, 1]
        >>> arr2 = np.array(0)
        >>> arr3 = np.array([[[[[1]]]]])
        >>> arr1_ = atleast_nd(arr1, n)
        >>> arr2_ = atleast_nd(arr2, n)
        >>> arr3_ = atleast_nd(arr3, n)
        >>> result1 = ub.repr2(arr1_.tolist())
        >>> result2 = ub.repr2(arr2_.tolist())
        >>> result3 = ub.repr2(arr3_.tolist())
        >>> result = '\n'.join([result1, result2, result3])
        >>> print(result)
        [[[[1]]], [[[1]]], [[[1]]]]
        [[[[0]]]]
        [[[[[1]]]]]

    Ignore:
        # Hmm, mine is actually faster
        %timeit atleast_nd(arr, 3)
        %timeit np.atleast_3d(arr)
    """
    arr_ = np.asanyarray(arr)
    ndims = len(arr_.shape)
    if n is not None and ndims <  n:
        # append the required number of dimensions to the front or back
        if tofront:
            expander = (None,) * (n - ndims) + (Ellipsis,)
        else:
            expander = (Ellipsis,) + (None,) * (n - ndims)
        arr_ = arr_[expander]
    return arr_


def apply_grouping(items, groupxs, axis=0):
    """
    applies grouping from group_indicies
    apply_grouping

    Args:
        items (ndarray):
        groupxs (list of ndarrays):

    Returns:
        list of ndarrays: grouped items

    SeeAlso:
        group_indices
        invert_apply_grouping

    Example:
        >>> # ENABLE_DOCTEST
        >>> #np.random.seed(42)
        >>> #size = 10
        >>> #idx2_groupid = np.array(np.random.randint(0, 4, size=size))
        >>> #items = np.random.randint(5, 10, size=size)
        >>> idx2_groupid = np.array([2, 1, 2, 1, 2, 1, 2, 3, 3, 3, 3])
        >>> items        = np.array([1, 8, 5, 5, 8, 6, 7, 5, 3, 0, 9])
        >>> (keys, groupxs) = group_indices(idx2_groupid)
        >>> grouped_items = apply_grouping(items, groupxs)
        >>> result = str(grouped_items)
        >>> print(result)
        [array([8, 5, 6]), array([1, 5, 8, 7]), array([5, 3, 0, 9])]
    """
    # SHOULD DO A CONTIGUOUS CHECK HERE
    #items_ = np.ascontiguousarray(items)
    return [items.take(xs, axis=axis) for xs in groupxs]


def group_indices(idx2_groupid, assume_sorted=False):
    r"""
    group_indices

    Args:
        idx2_groupid (ndarray): numpy array of group ids (must be numeric)

    Returns:
        tuple (ndarray, list of ndarrays): (keys, groupxs)

    Example0:
        >>> idx2_groupid = np.array([2, 1, 2, 1, 2, 1, 2, 3, 3, 3, 3])
        >>> (keys, groupxs) = group_indices(idx2_groupid)
        >>> result = ub.repr2((keys, groupxs), nobr=True, with_dtype=True)
        >>> print(result)
        np.array([1, 2, 3], dtype=np.int64),
        [
            np.array([1, 3, 5], dtype=np.int64),
            np.array([0, 2, 4, 6], dtype=np.int64),
            np.array([ 7,  8,  9, 10], dtype=np.int64),
        ],

    Example1:
        >>> idx2_groupid = np.array([[  24], [ 129], [ 659], [ 659], [ 24],
        ...       [659], [ 659], [ 822], [ 659], [ 659], [24]])
        >>> # 2d arrays must be flattened before coming into this function so
        >>> # information is on the last axis
        >>> (keys, groupxs) = group_indices(idx2_groupid.T[0])
        >>> result = ub.repr2((keys, groupxs), nobr=True, with_dtype=True)
        >>> print(result)
        np.array([ 24, 129, 659, 822], dtype=np.int64),
        [
            np.array([ 0,  4, 10], dtype=np.int64),
            np.array([1], dtype=np.int64),
            np.array([2, 3, 5, 6, 8, 9], dtype=np.int64),
            np.array([7], dtype=np.int64),
        ],

    Example2:
        >>> idx2_groupid = np.array([True, True, False, True, False, False, True])
        >>> (keys, groupxs) = group_indices(idx2_groupid)
        >>> result = ub.repr2((keys, groupxs), nobr=True, with_dtype=True)
        >>> print(result)
        np.array([False,  True], dtype=bool),
        [
            np.array([2, 4, 5], dtype=np.int64),
            np.array([0, 1, 3, 6], dtype=np.int64),
        ],

    Timeit:
        import numba
        group_indices_numba = numba.jit(group_indices)
        group_indices_numba(idx2_groupid)

    SeeAlso:
        apply_grouping

    References:
        http://stackoverflow.com/questions/4651683/
        numpy-grouping-using-itertools-groupby-performance

    TODO:
        Look into np.split
        http://stackoverflow.com/questions/21888406/
        getting-the-indexes-to-the-duplicate-columns-of-a-numpy-array
    """
    # Sort items and idx2_groupid by groupid
    if assume_sorted:
        sortx = np.arange(len(idx2_groupid))
        groupids_sorted = idx2_groupid
    else:
        sortx = idx2_groupid.argsort()
        groupids_sorted = idx2_groupid.take(sortx)

    # Ensure bools are internally cast to integers
    if groupids_sorted.dtype.kind == 'b':
        cast_groupids = groupids_sorted.astype(np.int8)
    else:
        cast_groupids = groupids_sorted

    num_items = idx2_groupid.size
    # Find the boundaries between groups
    diff = np.ones(num_items + 1, cast_groupids.dtype)
    np.subtract(cast_groupids[1:], cast_groupids[:-1], out=diff[1:num_items])
    idxs = np.flatnonzero(diff)
    # Groups are between bounding indexes
    # <len(keys) bottlneck>
    groupxs = [sortx[lx:rx] for lx, rx in zip(idxs, idxs[1:])]  # 34.5%
    # Unique group keys
    keys = groupids_sorted[idxs[:-1]]
    return keys, groupxs


def group_items(item_list, groupid_list, assume_sorted=False, axis=None):
    keys, groupxs = group_indices(groupid_list, assume_sorted=assume_sorted)
    grouped_values = apply_grouping(item_list, groupxs, axis=axis)
    return dict(zip(keys, grouped_values))


def argsubmax(ydata, xdata=None):
    """
    Finds a single submaximum value to subindex accuracy.
    If xdata is not specified, submax_x is a fractional index.
    Otherwise, submax_x is sub-xdata (essentially doing the index interpolation
    for you)

    Example:
        >>> ydata = [ 0,  1,  2, 1.5,  0]
        >>> xdata = [00, 10, 20,  30, 40]
        >>> result1 = argsubmax(ydata, xdata=None)
        >>> result2 = argsubmax(ydata, xdata=xdata)
        >>> result = ub.repr2([result1, result2], precision=4, nl=1, nobr=True)
        >>> print(result)
        (2.1667, 2.0208),
        (21.6667, 2.0208),

    Example:
        >>> hist_ = np.array([0, 1, 2, 3, 4])
        >>> centers = None
        >>> maxima_thresh=None
        >>> argsubmax(hist_)
        (4.0, 4.0)
    """
    if len(ydata) == 0:
        raise IndexError('zero length array')
    ydata = np.asarray(ydata)
    xdata = None if xdata is None else np.asarray(xdata)
    submaxima_x, submaxima_y = argsubmaxima(ydata, centers=xdata)
    idx = submaxima_y.argmax()
    submax_y = submaxima_y[idx]
    submax_x = submaxima_x[idx]
    return submax_x, submax_y


def argsubmaxima(hist, centers=None, maxima_thresh=None, _debug=False):
    r"""
    Determines approximate maxima values to subindex accuracy.

    Args:
        hist_ (ndarray): ydata, histogram frequencies
        centers (ndarray): xdata, histogram labels
        maxima_thresh (float): cutoff point for labeing a value as a maxima

    Returns:
        tuple: (submaxima_x, submaxima_y)

    CommandLine:
        python -m vtool.histogram argsubmaxima
        python -m vtool.histogram argsubmaxima --show

    Example:
        >>> maxima_thresh = .8
        >>> hist = np.array([6.73, 8.69, 0.00, 0.00, 34.62, 29.16, 0.00, 0.00, 6.73, 8.69])
        >>> centers = np.array([-0.39, 0.39, 1.18, 1.96,  2.75,  3.53, 4.32, 5.11, 5.89, 6.68])
        >>> (submaxima_x, submaxima_y) = argsubmaxima(hist, centers, maxima_thresh)
        >>> result = str((submaxima_x, submaxima_y))
        >>> print(result)
        >>> # xdoc: +REQUIRES(--show)
        >>> import plottool as pt
        >>> pt.draw_hist_subbin_maxima(hist, centers)
        >>> pt.show_if_requested()
        (array([ 3.0318792]), array([ 37.19208239]))
    """
    maxima_x, maxima_y, argmaxima = _hist_argmaxima(hist, centers, maxima_thresh=maxima_thresh)
    argmaxima = np.asarray(argmaxima)
    if _debug:
        print('Argmaxima: ')
        print(' * maxima_x = %r' % (maxima_x))
        print(' * maxima_y = %r' % (maxima_y))
        print(' * argmaxima = %r' % (argmaxima))
    flags = (argmaxima == 0) | (argmaxima == len(hist) - 1)
    argmaxima_ = argmaxima[~flags]
    submaxima_x_, submaxima_y_ = _interpolate_submaxima(argmaxima_, hist, centers)
    if np.any(flags):
        endpts = argmaxima[flags]
        submaxima_x = (np.hstack([submaxima_x_, centers[endpts]])
                       if centers is not None else
                       np.hstack([submaxima_x_, endpts]))
        submaxima_y = np.hstack([submaxima_y_, hist[endpts]])
    else:
        submaxima_y = submaxima_y_
        submaxima_x = submaxima_x_
    if _debug:
        print('Submaxima: ')
        print(' * submaxima_x = %r' % (submaxima_x))
        print(' * submaxima_y = %r' % (submaxima_y))
    return submaxima_x, submaxima_y


def _hist_argmaxima(hist, centers=None, maxima_thresh=None):
    """
    must take positive only values

    CommandLine:
        python -m vtool.histogram _hist_argmaxima

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.histogram import *  # NOQA
        >>> maxima_thresh = .8
        >>> hist = np.array([    6.73, 8.69, 0.00, 0.00, 34.62, 29.16, 0.00, 0.00, 6.73, 8.69])
        >>> centers = np.array([-0.39, 0.39, 1.18, 1.96,  2.75,  3.53, 4.32, 5.11, 5.89, 6.68])
        >>> maxima_x, maxima_y, argmaxima = _hist_argmaxima(hist, centers)
        >>> result = str((maxima_x, maxima_y, argmaxima))
        >>> print(result)
        (array([ 0.39,  2.75]), array([  8.69,  34.62]), array([1, 4]))
    """
    # FIXME: Not handling general cases
    # [0] index because argrelmaxima returns a tuple
    argmaxima_ = scipy.signal.argrelextrema(hist, np.greater)[0]
    if len(argmaxima_) == 0:
        argmaxima_ = hist.argmax()
    if maxima_thresh is not None:
        # threshold maxima to be within a factor of the maximum
        maxima_y = hist[argmaxima_]
        isvalid = maxima_y > maxima_y.max() * maxima_thresh
        argmaxima = argmaxima_[isvalid]
    else:
        argmaxima = argmaxima_
    maxima_y = hist[argmaxima]
    maxima_x = argmaxima if centers is None else centers[argmaxima]
    return maxima_x, maxima_y, argmaxima


def _interpolate_submaxima(argmaxima, hist_, centers=None):
    r"""
    Args:
        argmaxima (ndarray): indicies into ydata / centers that are argmaxima
        hist_ (ndarray): ydata, histogram frequencies
        centers (ndarray): xdata, histogram labels

    FIXME:
        what happens when argmaxima[i] == len(hist_)

    CommandLine:
        python -m vtool.histogram --test-_interpolate_submaxima --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.histogram import *  # NOQA
        >>> argmaxima = np.array([1, 4, 7])
        >>> hist_ = np.array([    6.73, 8.69, 0.00, 0.00, 34.62, 29.16, 0.00, 0.00, 6.73, 8.69])
        >>> centers = np.array([-0.39, 0.39, 1.18, 1.96,  2.75,  3.53, 4.32, 5.11, 5.89, 6.68])
        >>> submaxima_x, submaxima_y = _interpolate_submaxima(argmaxima, hist_, centers)
        >>> locals_ = ub.exec_func_src(_interpolate_submaxima,
        >>>                            key_list=['x123', 'y123', 'coeff_list'])
        >>> x123, y123, coeff_list = locals_
        >>> res = (submaxima_x, submaxima_y)
        >>> result = ub.repr2(res, nl=1, nobr=True, precision=2, with_dtype=True)
        >>> print(result)
        >>> # xdoc: +REQUIRES(--show)
        >>> import plottool as pt
        >>> pt.ensureqt()
        >>> pt.figure(fnum=pt.ensure_fnum(None))
        >>> pt.plot(centers, hist_, '-')
        >>> pt.plot(centers[argmaxima], hist_[argmaxima], 'o', label='argmaxima')
        >>> pt.plot(submaxima_x, submaxima_y, 'b*', markersize=20, label='interp maxima')
        >>> # Extract parabola points
        >>> pt.plt.plot(x123, y123, 'o', label='maxima neighbors')
        >>> xpoints = [np.linspace(x1, x3, 50) for (x1, x2, x3) in x123.T]
        >>> ypoints = [np.polyval(coeff, x_pts) for x_pts, coeff in zip(xpoints, coeff_list)]
        >>> # Draw Submax Parabola
        >>> for x_pts, y_pts in zip(xpoints, ypoints):
        >>>     pt.plt.plot(x_pts, y_pts, 'g--', lw=2)
        >>> pt.show_if_requested()
        np.array([ 0.15,  3.03,  5.11], dtype=np.float64),
        np.array([  9.2 ,  37.19,   0.  ], dtype=np.float64),

    Example:
        >>> hist_ = np.array([5])
        >>> argmaxima = [0]

    """
    if len(argmaxima) == 0:
        return [], []
    argmaxima = np.asarray(argmaxima)
    neighbs = np.vstack((argmaxima - 1, argmaxima, argmaxima + 1))
    # flags = (neighbs[2] > (len(hist_) - 1)) | (neighbs[0] < 0)
    # neighbs = np.clip(neighbs, 0, len(hist_) - 1)
    # if np.any(flags):
    #     # Clip out of bounds positions
    #     neighbs[0, flags] = neighbs[1, flags]
    #     neighbs[2, flags] = neighbs[1, flags]
    y123 = hist_[neighbs]
    x123 = neighbs if centers is None else centers[neighbs]
    # if np.any(flags):
    #     # Make symetric values so maxima is found exactly in center
    #     y123[0, flags] = y123[1, flags] - 1
    #     y123[2, flags] = y123[1, flags] - 1
    #     x123[0, flags] = x123[1, flags] - 1
    #     x123[2, flags] = x123[1, flags] - 1
    # Fit parabola around points
    coeff_list = [np.polyfit(x123_, y123_, deg=2)
                  for (x123_, y123_) in zip(x123.T, y123.T)]
    A, B, C = np.vstack(coeff_list).T
    submaxima_x, submaxima_y = _maximum_parabola_point(A, B, C)

    # Check to make sure submaxima is not less than original maxima
    # (can be the case only if the maxima is incorrectly given)
    # In this case just return what the user wanted as the maxima
    maxima_y = y123[1, :]
    invalid = submaxima_y < maxima_y
    if np.any(invalid):
        if centers is not None:
            submaxima_x[invalid] = centers[argmaxima[invalid]]
        else:
            submaxima_x[invalid] = argmaxima[invalid]
        submaxima_y[invalid] = hist_[argmaxima[invalid]]
    return submaxima_x, submaxima_y


def _maximum_parabola_point(A, B, C):
    """ Maximum x point is where the derivative is 0 """
    xv = -B / (2 * A)
    yv = C - B * B / (4 * A)
    return xv, yv
