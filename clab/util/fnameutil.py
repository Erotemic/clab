# -*- coding: utf-8 -*-
"""
Processing for filenames. The logic is relatively hacky.

pip install pygtrie
"""
from __future__ import print_function, division
from os.path import commonprefix, isdir, dirname, relpath, splitext
from collections import deque
import pygtrie


from clab import profiler
@profiler.profile_onthefly
def shortest_unique_prefixes(items, sep=None):
    """
    The shortest unique prefix algorithm.

    Args:
        items (list of str): returned prefixes will be unique wrt this set
        sep (str): if specified, all characters between separators are treated
            as a single symbol. Makes the algo much faster.

    Returns:
        list of str: a prefix for each item that uniquely identifies it
           wrt to the original items.

    References:
        http://www.geeksforgeeks.org/find-all-shortest-unique-prefixes-to-represent-each-word-in-a-given-list/
        https://github.com/Briaares/InterviewBit/blob/master/Level6/Shortest%20Unique%20Prefix.cpp

    Requires:
        pip install pygtrie

    Doctest:
        >>> from clab.fnameutil import *
        >>> items = ["zebra", "dog", "duck", "dove"]
        >>> shortest_unique_prefixes(items)
        ['z', 'dog', 'du', 'dov']

    Timeing:
        >>> # make numbers larger to stress test
        >>> # L = max length of a string, N = number of strings,
        >>> # C = smallest gaurenteed common length
        >>> # (the setting N=10000, L=100, C=20 is feasible we are good)
        >>> import random
        >>> def make_data(N, L, C):
        >>>     rng = random.Random(0)
        >>>     return [''.join(['a' if i < C else chr(rng.randint(97, 122))
        >>>                      for i in range(L)]) for _ in range(N)]
        >>> items = make_data(N=1000, L=10, C=0)
        >>> %timeit shortest_unique_prefixes(items)
        17.5 ms ± 244 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
        >>> items = make_data(N=1000, L=100, C=0)
        >>> %timeit shortest_unique_prefixes(items)
        141 ms ± 1.05 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
        >>> items = make_data(N=1000, L=100, C=70)
        >>> %timeit shortest_unique_prefixes(items)
        141 ms ± 1.05 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
        >>> items = make_data(N=10000, L=250, C=20)
        >>> %timeit shortest_unique_prefixes(items)
        3.55 s ± 23 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
    """
    if len(set(items)) != len(items):
        raise ValueError('inputs must be unique')

    # construct trie
    if sep is None:
        trie = pygtrie.CharTrie.fromkeys(items, value=0)
    else:
        # In some simple cases we can avoid constructing a trie
        tokens = [item.split(sep) for item in items]
        naive_solution = [t[0] for t in tokens]
        if len(naive_solution) == len(set(naive_solution)):
            return naive_solution
        # naive_solution = ['-'.join(t[0:2]) for t in tokens]
        # if len(naive_solution) == len(set(naive_solution)):
        #     return naive_solution

        trie = pygtrie.StringTrie.fromkeys(items, value=0, separator=sep)

    # Hack into the internal structure and insert frequencies at each node
    def _iternodes(self):
        """
        Generates all nodes in the trie
        """
        stack = deque([[self._root]])
        while stack:
            for node in stack.pop():
                yield node
                stack.append(node.children.values())

    # Set the value (frequency) of all nodes to zero.
    for node in _iternodes(trie):
        node.value = 0

    # For each item trace its path and increment frequencies
    for item in items:
        final_node, trace = trie._get_node(item)
        for key, node in trace:
            node.value += 1

    # if not isinstance(node.value, int):
    #     node.value = 0

    # Query for the first prefix with frequency 1 for each item.
    # This is the shortest unique prefix over all items.
    unique = []
    for item in items:
        freq = None
        for prefix, freq in trie.prefixes(item):
            if freq == 1:
                break
        assert freq == 1, 'item={} has no unique prefix'.format(item)
        unique.append(prefix)
    return unique


@profiler.profile_onthefly
def shortest_unique_suffixes(items, sep=None):
    """
    Example:
        >>> from clab.fnameutil import *
        >>> items = ["zebra", "dog", "duck", "dove"]
        >>> shortest_unique_suffixes(items)
        ['a', 'g', 'k', 'e']

    Example:
        >>> from clab.fnameutil import *
        >>> items = ["aa/bb/cc", "aa/bb/bc", "aa/bb/dc", "aa/cc/cc"]
        >>> shortest_unique_suffixes(items)
        ['a', 'g', 'k', 'e']
    """
    snoitpo = [p[::-1] for p in items]
    sexiffus = shortest_unique_prefixes(snoitpo, sep=sep)
    suffixes = [s[::-1] for s in sexiffus]
    return suffixes


# def _align_fallback(paths1, paths2):
#     """
#     Ignore:
#         >>> import itertools as it
#         >>> from clab.util.fnameutil import *
#         >>> from clab.util.fnameutil import _align_fallback, _safepaths
#         >>> def _make_input(fmt, n=10):
#         >>>     for i in range(n):
#         >>>         yield (fmt.format(id=i, type='im'), fmt.format(id=i, type='gt'))
#         >>>         #yield (fmt.format(id=i, type='im'), fmt.format(id=i, type='gt'))
#         >>> #
#         >>> n = 4
#         >>> paths1, paths2 = map(list, zip(*it.chain(
#         >>>     _make_input('{type}/{id}.png', n=n),
#         >>>     _make_input('{id}/{type}.png', n=n),
#         >>>     #_make_input('{type}/{id}-{type}.png', n=n),
#         >>>     #_make_input('{type}/{type}-{id}.png', n=n),
#         >>>     #_make_input('{id}/{type}-{id}.png', n=n),
#         >>>     #_make_input('{id}/{id}-{type}.png', n=n),
#         >>>     )))
#         >>> np.random.shuffle(paths2)
#     """
#     import numpy as np
#     import editdistance

#     safe_paths1 = _safepaths(paths1)
#     safe_paths2 = _safepaths(paths2)

#     # initialize a cost matrix
#     shape = (len(safe_paths1), len(safe_paths2))
#     cost_matrix = np.full(shape, fill_value=np.inf)

#     # Can we come up with the right distance function?
#     # edit-distance wont work for long type specifiers
#     # does tokenized strings help?
#     import re
#     tokens1 = [re.split('[-.]', p) for p in safe_paths1]
#     tokens2 = [re.split('[-.]', p) for p in safe_paths2]

#     import ubelt as ub
#     # import itertools as it
#     # TODO: use frequency weights
#     # token_freq = ub.dict_hist(it.chain(*(tokens1 + tokens2)))
#     # token_weights = ub.map_vals(lambda x: 1 / x, token_freq)

#     # The right distance function might be to weight the disagree bit by the
#     # frequency of the token in the dataset.

#     # only compute one half of the triangle
#     idxs1, idxs2 = np.triu_indices(len(safe_paths1), k=0)
#     distances = [
#         editdistance.eval(tokens1[i], tokens2[j])
#         for i, j in zip(idxs1, idxs2)
#     ]
#     cost_matrix[(idxs1, idxs2)] = distances

#     # make costs symmetric
#     cost_matrix = np.minimum(cost_matrix.T, cost_matrix)

#     import scipy.optimize
#     assign = scipy.optimize.linear_sum_assignment(cost_matrix)

#     sortx = assign[1][assign[0].argsort()]

#     import ubelt as ub
#     list(ub.take(paths2, sortx))


@profiler.profile_onthefly
def dumpsafe(paths, repl='<sl>'):
    """
    enforces that filenames will not conflict.
    Removes common the common prefix, and replaces slashes with <sl>

    >>> paths = ['foo/{:04d}/{:04d}'.format(i, j) for i in range(2) for j in range(20)]
    >>> list(dumpsafe(paths, '-'))
    """
    common_pref = commonprefix(paths)
    if not isdir(common_pref):
        im_pref = dirname(common_pref)
        if common_pref[len(im_pref):len(im_pref) + 1] == '/':
            im_pref += '/'
        elif common_pref[len(im_pref):len(im_pref) + 1] == '\\':
            im_pref += '\\'
    else:
        im_pref = common_pref

    start = len(im_pref)
    dump_paths = (
        p[start:].replace('/', repl).replace('\\', repl)  # faster
        # relpath(p, im_pref).replace('/', repl).replace('\\', repl)
        for p in paths
    )
    return dump_paths


@profiler.profile_onthefly
def _safepaths(paths):
    """
    x = '/home/local/KHQ/jon.crall/code/clab/clab/live/urban_train.py'
    import re
    %timeit splitext(x.replace('<sl>', '-').replace('_', '-'))[0]
    %timeit splitext(re.sub('<sl>|_', '-', x))
    %timeit x[:x.rfind('.')].replace('<sl>', '-').replace('_', '-')
    %timeit fast_name_we(x)
    %timeit x[:x.rfind('.')]

    >>> paths = ['foo/{:04d}/{:04d}'.format(i, j) for i in range(2) for j in range(20)]
    >>> _safepaths(paths)
    """
    def fast_name_we(fname):
        # Assume that extensions are no more than 7 chars for extra speed
        pos = fname.rfind('.', -7)
        return fname if pos == -1 else fname[:pos]

    safe_paths = [
        # faster than splitext
        fast_name_we(x).replace('_', '-').replace('<sl>', '-')
        # splitext(x.replace('<sl>', '-').replace('_', '-'))[0]
        for x in dumpsafe(paths, repl='-')
    ]
    return safe_paths


@profiler.profile_onthefly
def align_paths(paths1, paths2):
    """
    return path2 in the order of path1

    This function will only work where file types (i.e. image / groundtruth)
    are specified by EITHER a path prefix XOR a path suffix (note this is an
    exclusive or. do not mix prefixes and suffixes), either as part of a
    filename or parent directory. In the case of a filename it is assumped this
    "type identifier" is separated from the rest of the path by an underscore
    or hyphen.

    paths1, paths2 = gt_paths, pred_paths

    Doctest:
        >>> from clab.util.fnameutil import *
        >>> def test_gt_arrangements(paths1, paths2, paths2_):
        >>>     # no matter what order paths2_ comes in, it should align with the groundtruth
        >>>     assert align_paths(paths1, paths2_) == paths2
        >>>     assert align_paths(paths1[::-1], paths2_) == paths2[::-1]
        >>>     assert align_paths(paths1[0::2] + paths1[1::2], paths2_) == paths2[0::2] + paths2[1::2]
        >>>     sortx = np.arange(len(paths1))
        >>>     np.random.shuffle(sortx)
        >>>     assert align_paths(list(np.take(paths1, sortx)), paths2_) == list(np.take(paths2, sortx))
        >>> #
        >>> def test_input_arrangements(paths1, paths2):
        >>>     paths2_ = paths2.copy()
        >>>     test_gt_arrangements(paths1, paths2, paths2_)
        >>>     test_gt_arrangements(paths1, paths2, paths2_[::-1])
        >>>     np.random.shuffle(paths2_)
        >>>     test_gt_arrangements(paths1, paths2, paths2_)
        >>> paths1 = ['foo/{:04d}/{:04d}'.format(i, j) for i in range(2) for j in range(20)]
        >>> paths2 = ['bar/{:04d}/{:04d}'.format(i, j) for i in range(2) for j in range(20)]
        >>> test_input_arrangements(paths1, paths2)
        >>> paths1 = ['foo/{:04d}/{:04d}'.format(i, j) for i in range(2) for j in range(20)]
        >>> paths2 = ['bar<sl>{:04d}<sl>{:04d}'.format(i, j) for i in range(2) for j in range(20)]
        >>> test_input_arrangements(paths1, paths2)

    Speed:
        >>> import ubelt as ub
        >>> paths1 = [ub.truepath('~/foo/{:04d}/{:04d}').format(i, j) for i in range(2) for j in range(10000)]
        >>> paths2 = [ub.truepath('~/bar/{:04d}/{:04d}').format(i, j) for i in range(2) for j in range(10000)]
        >>> np.random.shuffle(paths2)
        >>> aligned = align_paths(paths1, paths2)

        items = [p[::-1] for p in _safepaths(paths1)]

    Ignore:
        >>> # pathological case (can we support this?)
        >>> aligned = [
        >>>     ('ims/aaa.png', 'gts/aaa.png'),
        >>>     ('ims/bbb.png', 'gts/bbb.png'),
        >>>     ('ims/ccc.png', 'gts/ccc.png'),
        >>>     # ---
        >>>     ('aaa/im.png', 'aaa/gt.png'),
        >>>     ('bbb/im.png', 'bbb/gt.png'),
        >>>     ('ccc/im.png', 'ccc/gt.png'),
        >>>     # ---
        >>>     ('ims/im-aaa.png', 'gts/gt-aaa.png'),
        >>>     ('ims/im-bbb.png', 'gts/gt-bbb.png'),
        >>>     ('ims/im-ccc.png', 'gts/gt-ccc.png'),
        >>>     # ---
        >>>     ('ims/aaa-im.png', 'gts/aaa-gt.png'),
        >>>     ('ims/bbb-im.png', 'gts/bbb-gt.png'),
        >>>     ('ims/ccc-im.png', 'gts/ccc-gt.png'),
        >>> ]
        >>> paths1, paths2 = zip(*aligned)

    """
    import numpy as np
    assert len(paths1) == len(paths2), (
        'cannot align unequal no of items {} != {}.'.format(len(paths1), len(paths2)))
    safe_paths1 = _safepaths(paths1)
    safe_paths2 = _safepaths(paths2)

    # unique identifiers that should be comparable
    unique1 = shortest_unique_suffixes(safe_paths1, sep='-')
    unique2 = shortest_unique_suffixes(safe_paths2, sep='-')

    def not_comparable_msg():
        return '\n'.join([
            'paths are not comparable'
            'safe_paths1 = {}'.format(safe_paths1[0:3]),
            'safe_paths2 = {}'.format(safe_paths1[0:3]),
            'paths1 = {}'.format(safe_paths1[0:3]),
            'paths2 = {}'.format(safe_paths1[0:3]),
            'unique1 = {}'.format(unique1[0:3]),
            'unique2 = {}'.format(unique2[0:3]),
        ])

    try:
        # Assert these are unique identifiers common between paths
        assert sorted(set(unique1)) == sorted(unique1), not_comparable_msg()
        assert sorted(set(unique2)) == sorted(unique2), not_comparable_msg()
        assert sorted(unique1) == sorted(unique2), not_comparable_msg()
    except AssertionError:
        unique1 = shortest_unique_prefixes(safe_paths1, sep='-')
        unique2 = shortest_unique_prefixes(safe_paths2, sep='-')
        # Assert these are unique identifiers common between paths
        assert sorted(set(unique1)) == sorted(unique1), not_comparable_msg()
        assert sorted(set(unique2)) == sorted(unique2), not_comparable_msg()
        assert sorted(unique1) == sorted(unique2), not_comparable_msg()

    lookup = {k: v for v, k in enumerate(unique1)}
    sortx = np.argsort([lookup[u] for u in unique2])

    sorted_paths2 = [paths2[x] for x in sortx]
    return sorted_paths2


def check_aligned(paths1, paths2):
    from os.path import basename
    if len(paths1) != len(paths2):
        return False

    # Try to short circuit common cases
    basenames1 = map(basename, paths1)
    basenames2 = map(basename, paths2)
    if all(p1 == p2 for p1, p2 in zip(basenames1, basenames2)):
        return True

    try:
        # Full case
        aligned_paths2 = align_paths(paths1, paths2)
    except AssertionError:
        return False
    return aligned_paths2 == paths2
