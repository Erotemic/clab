import numpy as np


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
        >>> result = ut.repr2(arr_.tolist())
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
        >>> result1 = ut.repr2(arr1_.tolist())
        >>> result2 = ut.repr2(arr2_.tolist())
        >>> result3 = ut.repr2(arr3_.tolist())
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
