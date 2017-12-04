import heapq
import ubelt as ub
import sortedcontainers
import operator


def _heappush_max(heap, item):
    """ why is this not in heapq """
    heap.append(item)
    heapq._siftdown_max(heap, 0, len(heap) - 1)


# class _Simple_iLocIndexer(object):
#     def __init__(self, data):
#         self.data = data

#     def __getitem__(self, idx):
#         return self.data.__getitem__(idx)

#     # def __setitem__(self, idx, value):
#     #     return self.data.__setitem__(idx, value)

#     # def __delitem__(self, idx):
#     #     return self.data.__delitem__(idx)


class SortedQueue(ub.NiceRepr):
    """

    Example:
        >>> items = [('a', 9), ('b', 8), ('c', 6), ('d', 5)]
        >>> self = SortedQueue(items, maxlen=3)
        >>> initial = list(self)
        >>> # should be unable to add something with a higher value
        >>> self['e'] = 10
        >>> assert 'e' not in self
        >>> # a lower value should kick out the worst so far
        >>> self['f'] = 1
        >>> assert 'f' in self
        >>> assert self['f'] < self['c']
        >>> self['c'] = 0
        >>> assert self['f'] > self['c']

        >>> pickled = pickle.loads(pickle.dumps(self))
        >>> assert self is not pickled
        >>> assert self == pickled
    """
    def __init__(self, items=None, maxlen=None):
        self.maxlen = maxlen
        self._dict = {}
        self._list = sortedcontainers.SortedListWithKey(key=operator.itemgetter(1))
        if items:
            self.update(items)
        # self.iloc = _Simple_iLocIndexer(self._list)

    def update(self, items):
        for key, value in items:
            self[key] = value

    def __len__(self):
        return len(self._dict)

    def __nice__(self):
        return 'size=%r' % (len(self),)

    def __contains__(self, key):
        return key in self._dict

    def __eq__(self, other):
        return self._dict == other._dict

    def _key_idx(self, key):
        if key in self._dict:
            # linear time when replacing an existing key
            for ix, item in enumerate(self):
                if item[0] == key:
                    return ix
            assert False
        else:
            return None

    def __setitem__(self, key, value):
        less_extreme = operator.lt  # change to gt if we want big things

        idx = self._key_idx(key)
        if idx is not None:
            # overwrite existing item
            del self._dict[key]
            del self._list[idx]

        if self.maxlen is not None and len(self) >= self.maxlen:
            extreme_index = -1
            extreme_key, extreme_val = self._list[extreme_index]
            if less_extreme(value, extreme_val):
                # if extreme_key not in self._dict:
                #     import utool
                #     utool.embed()

                del self._dict[extreme_key]
                del self._list[-1]
                self._list.add((key, value))
                self._dict[key] = value
        else:
            self._list.add((key, value))
            self._dict[key] = value
        assert len(self._list) == len(self._dict)

    def __getitem__(self, key):
        return self._dict[key]

    def __iter__(self):
        return iter(self._list)

    def peek(self):
        """
        Peek at the next item in the queue
        """
        return self.peek_many(1)[0]

    def peek_many(self, n):
        """
        Peek at the next n items in the queue
        """
        return self._list[:n]


class PriorityQueue(ub.NiceRepr):
    """
    abstracted priority queue for our needs

    Combines properties of dicts and heaps
    Uses a heap for fast minimum/maximum value search
    Uses a dict for fast read only operations

    Args:
        items (list): initial key/value pairs. Values are priorities.
        ascending (bool): if True, lower numbers have higher priority

    References:
        http://code.activestate.com/recipes/522995-priority-dict-a-priority-queue-with-updatable-prio/
        https://stackoverflow.com/questions/33024215/built-in-max-heap-api-in-python

    Example:
        >>> from clab.util.priority_queue import *
        >>> items = dict(a=42, b=29, c=40, d=95, e=10)
        >>> self = PriorityQueue(items)
        >>> print(self)
        >>> assert len(self) == 5
        >>> print(self.pop())
        >>> assert len(self) == 4
        >>> print(self.pop())
        >>> assert len(self) == 3
        >>> print(self.pop())
        >>> print(self.pop())
        >>> print(self.pop())
        >>> assert len(self) == 0

    Example:
        >>> items = dict(a=(1.0, (2, 3)), b=(1.0, (1, 2)), c=(.9, (3, 2)))
        >>> self = PriorityQueue(items)
        >>> pickled = pickle.loads(pickle.dumps(self))
        >>> assert self is not pickled
        >>> assert self == pickled

    Ignore:
        # TODO: can also use sortedcontainers to maintain priority queue
        import sortedcontainers
        queue = sortedcontainers.SortedListWithKey(items, key=lambda x: x[1])
        queue.add(('a', 1))
    """
    def __init__(self, items=None, ascending=True):
        # Use a heap for the priority queue aspect
        self._heap = []
        # Use a dict for very quick read only operations
        self._dict = {}
        if ascending:
            self._heapify = heapq.heapify
            self._heappush = heapq.heappush
            self._heappop = heapq.heappop
        else:
            self._heapify = heapq._heapify_max
            self._heappush = _heappush_max
            self._heappop = heapq._heappop_max
        if items is not None:
            self.update(items)

    def _rebuild(self):
        # Worst Case O(N)
        self._heap = [(v, k) for k, v in self._dict.items()]
        self._heapify(self._heap)

    def __len__(self):
        return len(self._dict)

    def __nice__(self):
        return 'size=%r' % (len(self),)

    def __contains__(self, key):
        return key in self._dict

    def __eq__(self, other):
        return self._dict == other._dict

    def __getitem__(self, key):
        # Worse Case O(1)
        return self._dict[key]

    def get(self, key, default=None):
        return self._dict.get(key, default)

    def __setitem__(self, key, val):
        # Ammortized O(1)
        # assert not np.isnan(val), 'no nan in PQ: {}, {}'.format(key, val)
        self._dict[key] = val
        if len(self._heap) > 2 * len(self._dict):
            # When the heap grows larger than 2 * len(self), we rebuild it from
            # scratch to avoid wasting too much memory.
            self._rebuild()
        else:
            # Simply append the new value
            self._heappush(self._heap, (val, key))

    def clear(self):
        self._heap.clear()
        self._dict.clear()

    def __delitem__(self, key):
        del self._dict[key]

    def update(self, items):
        if isinstance(items, dict):
            items = items.items()
        items = list(items)
        if items:
            if len(items) > len(self._dict) / 2:
                self._dict.update(items)
                self._rebuild()
            else:
                for k, v in items:
                    self[k] = v

    def delete_items(self, key_list):
        for key in key_list:
            try:
                del self[key]
            except KeyError:
                pass

    def peek(self):
        """
        Peek at the next item in the queue
        """
        # Ammortized O(1)
        _heap = self._heap
        _dict = self._dict
        val, key = _heap[0]
        # Remove items marked for lazy deletion as they are encountered
        while key not in _dict or _dict[key] != val:
            self._heappop(_heap)
            val, key = _heap[0]
        return key, val

    def peek_many(self, n):
        """
        Actually this can be quite inefficient

        Example:
            >>> items = list(zip(range(256), range(256)))
            >>> n = 32
            >>> np.random.shuffle(items)
            >>> self = PriorityQueue(items, ascending=False)
            >>> self.peek_many(56)
        """
        if n == 0:
            return []
        elif n == 1:
            return [self.peek()]
        else:
            items = list(self.pop_many(n))
            self.update(items)
            return items

    def pop_many(self, n):
        count = 0
        while len(self._dict) > 0 and count < n:
            yield self.pop()
            count += 1

    def pop(self, key=ub.NoParam, default=ub.NoParam):
        """
        Pop the next item off the queue
        """
        # Dictionary pop if key is specified
        if key is not ub.NoParam:
            if default is ub.NoParam:
                return (key, self._dict.pop(key))
            else:
                return (key, self._dict.pop(key, default))
        # Otherwise do a heap pop
        try:
            # Ammortized O(1)
            _heap = self._heap
            _dict = self._dict
            val, key = self._heappop(_heap)
            # Remove items marked for lazy deletion as they are encountered
            while key not in _dict or _dict[key] != val:
                val, key = self._heappop(_heap)
        except IndexError:
            if len(_heap) == 0:
                raise IndexError('queue is empty')
            else:
                raise
        del _dict[key]
        return key, val
