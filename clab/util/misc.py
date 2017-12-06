import six
import numpy as np
import h5py
import tqdm


def isiterable(obj):
    """
    Returns if the object can be iterated over and is NOT a string

    Args:
        obj (object): python object

    Returns:
        bool: True if a non-string iterable

    Example:
        >>> obj_list = [3, [3], '3', (3,), [3,4,5]]
        >>> [isiterable(obj) for obj in obj_list]
        [False, True, False, True, True]
    """
    try:
        iter(obj)
        return not isinstance(obj, six.string_types)
    except:
        return False


def super2(this_class, self):
    """
    Fixes an error where reload causes super(X, self) to raise an exception

    The problem is that reloading causes X to point to the wrong version of the
    class.  This function fixes the problem by searching and returning the
    correct version of the class. See example for proper usage.

    Notes:
        This is only useful in python2 where you call super like this:
            >>> super(TheClass, self).__thefunc__(*theargs)

        In python 3 you can just call super like this:
            >>> super().__thefunc__(*theargs)

        So, if you dont need to support python2 you don't need this function.

    Args:
        this_class (class): class passed into super
        self (instance): instance passed into super

    DisableExample:
        >>> # If the parent module is reloaded, the super call may fail
        >>> # super(Foo, self).__init__()
        >>> # This will work around the problem most of the time
        >>> # super2(Foo, self).__init__()
        >>> class Parent(object):
        >>>     def __init__(self):
        >>>         self.parent_attr = 'bar'
        >>> class ChildSafe(Parent):
        >>>     def __init__(self):
        >>>         super2(ChildSafe, self).__init__()
        >>> class ChildDanger(Parent):
        >>>     def __init__(self):
        >>>         super(ChildDanger, self).__init__()
        >>> # initial loading is fine
        >>> safe1 = ChildSafe()
        >>> danger1 = ChildDanger()
        >>> assert safe1.parent_attr == 'bar'
        >>> assert danger1.parent_attr == 'bar'
        >>> # But if we reload (via simulation), then there will be issues
        >>> Parent_orig = Parent
        >>> ChildSafe_orig = ChildSafe
        >>> ChildDanger_orig = ChildDanger
        >>> # reloading the changes the outer classname
        >>> # but the inner class is still bound via the closure
        >>> # (we simulate this by using the old functions)
        >>> # (note in reloaded code the names would not change)
        >>> class Parent_new(object):
        >>>     __init__ = Parent_orig.__init__
        >>> Parent_new.__name__ = 'Parent'
        >>> class ChildSafe_new(Parent_new):
        >>>     __init__ = ChildSafe_orig.__init__
        >>> ChildSafe_new.__name__ = 'ChildSafe'
        >>> class ChildDanger_new(Parent_new):
        >>>     __init__ = ChildDanger_orig.__init__
        >>> ChildDanger_new.__name__ = 'ChildDanger'
        >>> #
        >>> safe2 = ChildSafe_new()
        >>> assert safe2.parent_attr == 'bar'
        >>> import pytest
        >>> with pytest.raises(TypeError):
        >>>     danger2 = ChildDanger_new()
    """

    if isinstance(self, this_class):
        # Case where everything is ok
        this_class_now = this_class
    else:
        # Case where we need to search for the right class
        def find_parent_class(leaf_class, target_name):
            from collections import deque
            target_class = None
            queue = deque()
            queue.append(leaf_class)
            seen_ = set([])
            while len(queue) > 0:
                related_class = queue.pop()
                if related_class.__name__ != target_name:
                    for base in related_class.__bases__:
                        if base not in seen_:
                            queue.append(base)
                            seen_.add(base)
                else:
                    target_class = related_class
                    break
            return target_class
        # Find new version of class
        leaf_class = self.__class__
        target_name = this_class.__name__
        target_class = find_parent_class(leaf_class, target_name)

        this_class_now = target_class
        #print('id(this_class)     = %r' % (id(this_class),))
        #print('id(this_class_now) = %r' % (id(this_class_now),))
    assert isinstance(self, this_class_now), (
        'Failed to fix %r, %r, %r' % (self, this_class, this_class_now))

    return super(this_class_now, self)


def roundrobin(iterables):
    """
    Round robin, iteration strategy

    In constrast to the recipie in itertools docs, the number of initial
    iterables does not need to be known, so it may be very large. This is
    useful if you only intend to extract a fixed number of items from the
    resulting iterable. Startup is instantainous.

    Args:
        iterables : an iterable of iterables

    Example:
        >>> list(roundrobin(['ABC', 'D', 'EF']))
        ['A', 'D', 'E', 'B', 'F', 'C']
    """
    curr_alive = map(iter, iterables)
    while curr_alive:
        next_alive = []
        for gen in curr_alive:
            try:
                yield next(gen)
            except StopIteration:
                pass
            else:
                next_alive.append(gen)
        curr_alive = next_alive


def read_h5arr(fpath):
    with h5py.File(fpath, 'r') as hf:
        return hf['arr_0'][...]


def write_h5arr(fpath, arr):
    with h5py.File(fpath, 'w') as hf:
        hf.create_dataset('arr_0', data=arr)


def read_arr(fpath):
    if fpath.endswith('.npy'):
        return np.read(fpath)
    elif fpath.endswith('.h5'):
        return read_h5arr(fpath)
    else:
        raise KeyError(fpath)


def write_arr(fpath, arr):
    if fpath.endswith('.npy'):
        return np.save(fpath, arr)
    elif fpath.endswith('.h5'):
        return write_h5arr(fpath, arr)
    else:
        raise KeyError(fpath)


def cc_locs(mask):
    """
    Grouped row/col locations of 4-connected-components
    """
    from clab import util
    import cv2
    ccs = cv2.connectedComponents(mask.astype(np.uint8), connectivity=4)[1]
    rc_locs = np.where(mask > 0)
    rc_ids = ccs[rc_locs]
    rc_arrs = np.ascontiguousarray(np.vstack(rc_locs).T)
    cc_to_loc = util.group_items(rc_arrs, rc_ids, axis=0)
    return cc_to_loc


def compact_idstr(dict_):
    """
    A short unique id string for a dict param config that is semi-interpretable
    """
    from clab import util
    import ubelt as ub
    short_keys = util.shortest_unique_prefixes(dict_.keys())
    short_dict = ub.odict(sorted(zip(short_keys, dict_.values())))
    idstr = ub.repr2(short_dict, nobr=1, itemsep='', si=1, nl=0,
                     explicit=1)
    return idstr


# class CacheStamp(object):
#     def __init__(self, fname, dpath=None, cfgstr=None):
#         import ubelt as ub
#         self.cacher = ub.Cacher(fname, dpath=dpath, cfgstr=cfgstr)

#     def __enter__(self):

#         return self


def protect_print(print):
    # def protected_print(*args, **kw):
    def protected_print(msg):
        if len(getattr(tqdm.tqdm, '_instances', [])):
            tqdm.tqdm.write(str(msg))
        else:
            # otherwise use the print / logger
            # (ensure logger has custom logic to exclude logging at this exact
            # place)
            print(msg)
    return protected_print
