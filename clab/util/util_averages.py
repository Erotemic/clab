import collections
import pandas as pd
import ubelt as ub


class MovingAve(ub.NiceRepr):
    def average(self):
        raise NotImplementedError()

    def update(self, other):
        raise NotImplementedError()

    def __nice__(self):
        return str(ub.repr2(self.average(), nl=0))

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)


class CumMovingAve(MovingAve):
    """
    Cumulative moving average of dictionary values

    References:
        https://en.wikipedia.org/wiki/Moving_average

    Example:
        >>> from clab.metrics import *
        >>> self = CumMovingAve()
        >>> print(str(self.update({'a': 10})))
        <CumMovingAve({'a': 10})>
        >>> print(str(self.update({'a': 0})))
        <CumMovingAve({'a': 5.0})>
        >>> print(str(self.update({'a': 2})))
        <CumMovingAve({'a': 4.0})>
    """
    def __init__(self):
        self.totals = ub.odict()
        self.n = 0

    def average(self):
        return {k: v / self.n for k, v in self.totals.items()}

    def update(self, other):
        self.n += 1
        for k, v in other.items():
            if pd.isnull(v):
                v = 0
            if k not in self.totals:
                self.totals[k] = 0
            self.totals[k] += v
        return self


class WindowedMovingAve(MovingAve):
    """
    Windowed moving average of dictionary values

    Args:
        window (int): number of previous observations to consider

    Example:
        >>> from clab.metrics import *
        >>> self = WindowedMovingAve(window=3)
        >>> print(str(self.update({'a': 10})))
        <WindowedMovingAve({'a': 10})>
        >>> print(str(self.update({'a': 0})))
        <WindowedMovingAve({'a': 5.0})>
        >>> print(str(self.update({'a': 2})))
        <WindowedMovingAve({'a': 1.0})>
    """
    def __init__(self, window=500):
        self.window = window
        self.totals = ub.odict()
        self.history = {}

    def average(self):
        return {k: v / len(self.history[k]) for k, v in self.totals.items()}

    def update(self, other):
        for k, v in other.items():
            if pd.isnull(v):
                v = 0
            if k not in self.totals:
                self.history[k] = collections.deque()
                self.totals[k] = 0
            self.totals[k] += v
            self.history[k].append(v)
            if len(self.history[k]) > self.window:
                # Push out the oldest value
                self.totals[k] -= self.history[k].popleft()
        return self


class ExpMovingAve(MovingAve):
    """
    Exponentially weighted moving average of dictionary values

    Args:
        span (float): roughly corresponds to window size.
            equivalent to (2 / alpha) - 1
        alpha (float): roughly corresponds to window size.
            equivalent to 2 / (span + 1)

    References:
        http://greenteapress.com/thinkstats2/html/thinkstats2013.html

    Example:
        >>> from clab.metrics import *
        >>> self = ExpMovingAve(span=3)
        >>> print(str(self.update({'a': 10})))
        <ExpMovingAve({'a': 10})>
        >>> print(str(self.update({'a': 0})))
        <ExpMovingAve({'a': 5.0})>
        >>> print(str(self.update({'a': 2})))
        <ExpMovingAve({'a': 3.5})>
    """
    def __init__(self, span=None, alpha=None):
        values = ub.odict()
        self.values = values
        if span is None and alpha is None:
            alpha = 0
        if not bool(span is None) ^ bool(alpha is None):
            raise ValueError('specify either alpha xor span')

        if alpha is not None:
            self.alpha = alpha
        elif span is not None:
            self.alpha = 2 / (span + 1)
        else:
            raise AssertionError('impossible state')

    def average(self):
        return self.values

    def update(self, other):
        alpha = self.alpha
        for k, v in other.items():
            if pd.isnull(v):
                v = 0
            if k not in self.values:
                self.values[k] = v
            else:
                self.values[k] = (alpha * v) + (1 - alpha) * self.values[k]
        return self

if __name__ == '__main__':
    r"""
    CommandLine:
        python -m clab.util.util_averages all
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
