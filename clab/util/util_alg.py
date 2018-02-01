import numpy as np


def mincost_assignment(cost, infok=False):
    """
    Does linear_sum_assignment, but can handle non-square matrices and infinite
    values.

    Args:
        cost (ndarray): matrix of costs to assign i to j

    Returns:
        rxs, cxs - indices corresponding to assignment. If cost is not square
            some indices will not be assigned.

    Example:
        >>> from clab.util.util_alg import *
        >>> cost = np.array([
        >>>     [10, 10, 10, 10, 4],
        >>>     [ 2, 10, 10, 10, 9],
        >>>     [ 2,  2,  2, 10, 7],
        >>>     [ 2, 10,  2, 10, 3],
        >>>     [10, 10,  2,  3, 3]
        >>>  ])
        >>> rxs, cxs = mincost_assignment(cost)
        >>> print(ub.repr2(dict(zip(rxs, cxs)), nl=0))
        {0: 4, 1: 0, 2: 1, 3: 2, 4: 3}
        >>> rxs, cxs = mincost_assignment(cost[0:2, :])
        >>> print(ub.repr2(dict(zip(rxs, cxs)), nl=0))
        {0: 4, 1: 0}
        >>> rxs, cxs = mincost_assignment(cost[:, 0:2])
        >>> print(ub.repr2(dict(zip(rxs, cxs)), nl=0))
        {2: 1, 3: 0}

        >>> cost = np.array([[np.inf, -np.inf]])
        >>> rxs, cxs = mincost_assignment(cost, infok=True)
        >>> print(ub.repr2(dict(zip(rxs, cxs)), nl=0))
        {0: 1}
        >>> cost = np.array([[np.inf, np.inf, np.inf]]).T
        >>> rxs, cxs = mincost_assignment(cost, infok=True)
        >>> print(ub.repr2(dict(zip(rxs, cxs)), nl=0))
        {2: 0}
        >>> cost = np.array([[np.inf, np.inf, np.inf]]).T
        >>> rxs, cxs = mincost_assignment(cost)
        >>> print(ub.repr2(dict(zip(rxs, cxs)), nl=0))
        {}
        >>> cost = np.array([[9000, np.inf, np.inf]]).T
        >>> rxs, cxs = mincost_assignment(cost)
        >>> print(ub.repr2(dict(zip(rxs, cxs)), nl=0))
        {0: 0}

    Example:
        >>> cost = np.array([[]])
        >>> rxs, cxs = mincost_assignment(cost)
        >>> print(ub.repr2(dict(zip(rxs, cxs)), nl=0))
        {}
    """
    import scipy
    import scipy.optimize
    partial = cost.copy()
    nrows, ncols = partial.shape
    ndim = max(nrows, ncols)
    area = (nrows * ncols) + 1

    finite_vals = partial[np.isfinite(partial)]
    if len(finite_vals) == 0:
        finite_vals  = np.array([1])

    # find numbers that are effective infinities
    # (TODO: might be able to get away by just multiplying by ndim)
    posinf_1 = +(abs(finite_vals.max()) + 1) * area
    neginf_1 = -(abs(finite_vals.min()) + 1) * area

    partial[np.isposinf(partial)] = posinf_1
    partial[np.isneginf(partial)] = neginf_1

    # make the matrix square
    extra_rows = []
    extra_cols = []

    # Find numbers that are even bigger infinities
    # (this allows inf to be a solution for the non-square matrix, but never
    # for a padded value)
    # (TODO: might be able to get away by just multiplying by ndim)
    posinf_2 = (posinf_1 + 1) * area

    extra_rows = np.full((max(ncols - nrows, 0), ncols), posinf_2)
    extra_cols = np.full((ndim, max(nrows - ncols, 0)), posinf_2)

    cost_matrix = partial.copy()
    cost_matrix = np.vstack([cost_matrix, extra_rows])
    cost_matrix = np.hstack([cost_matrix, extra_cols])

    rxs, cxs = scipy.optimize.linear_sum_assignment(cost_matrix)

    # Remove solutions that lie in extra rows / columns
    flags = (rxs < nrows) & (cxs < ncols)

    if not infok:
        # positive infinite assignments are not ok
        flags &= (cost_matrix[rxs, cxs] < posinf_1)

    valid_rxs = rxs[flags]
    valid_cxs = cxs[flags]
    return valid_rxs, valid_cxs
