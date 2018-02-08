from clab import util
import numpy as np
import ubelt as ub


class Monitor(object):
    """
    Example:
        >>> # simulate loss going down and then overfitting
        >>> from clab.monitor import *
        >>> rng = np.random.RandomState(0)
        >>> n = 300
        >>> losses = (sorted(rng.randint(10, n, size=n)) + rng.randint(0, 20, size=n) - 10)[::-1]
        >>> mious = (sorted(rng.randint(10, n, size=n)) + rng.randint(0, 20, size=n) - 10)
        >>> monitor = Monitor(min_keys=['loss'], max_keys=['miou'], smoothing=.6)
        >>> for epoch, (loss, miou) in enumerate(zip(losses, mious)):
        >>>     monitor.update(epoch, {'loss': loss, 'miou': miou})
        >>> # xdoc: +REQUIRES(--show)
        >>> monitor.show()
    """

    def __init__(monitor, min_keys=['loss'], max_keys=[], smoothing=.6,
                 patience=40):
        monitor.ewma = util.ExpMovingAve(alpha=1 - smoothing)
        monitor.raw_metrics = []
        monitor.smooth_metrics = []
        monitor.epochs = []
        monitor.is_good = []
        # monitor.other_data = []

        # Keep track of which metrics we want to maximize / minimize
        monitor.min_keys = min_keys
        monitor.max_keys = max_keys
        print('monitor.min_keys = {!r}'.format(monitor.min_keys))
        print('monitor.max_keys = {!r}'.format(monitor.max_keys))

        monitor.best_raw_metrics = None
        monitor.best_smooth_metrics = None
        monitor.best_epoch = None

        monitor.rel_threshold = 1e-4

        # early stopping

        monitor.patience = patience
        monitor.n_bad_epochs = 0

    def show(monitor):
        import matplotlib.pyplot as plt
        from clab.util import mplutil
        import pandas as pd
        mplutil.qtensure()
        smooth_ydatas = pd.DataFrame.from_dict(monitor.smooth_metrics).to_dict('list')
        raw_ydatas = pd.DataFrame.from_dict(monitor.raw_metrics).to_dict('list')
        keys = monitor.min_keys + monitor.max_keys
        pnum_ = mplutil.PlotNums(nSubplots=len(keys))
        for i, key in enumerate(keys):
            mplutil.multi_plot(
                monitor.epochs, {'raw ' + key: raw_ydatas[key],
                                 'smooth ' + key: smooth_ydatas[key]},
                xlabel='epoch', ylabel=key, pnum=pnum_[i], fnum=1,
                # markers={'raw ' + key: '-', 'smooth ' + key: '--'},
                # colors={'raw ' + key: 'b', 'smooth ' + key: 'b'},
            )

            # star all the good epochs
            flags = np.array(monitor.is_good)
            if np.any(flags):
                plt.plot(list(ub.compress(monitor.epochs, flags)),
                         list(ub.compress(smooth_ydatas[key], flags)), 'b*')

    def __getstate__(monitor):
        state = monitor.__dict__.copy()
        ewma = state.pop('ewma')
        state['ewma_state'] = ewma.__dict__
        return state

    def __setstate__(monitor, state):
        ewma_state = state.pop('ewma_state', None)
        if ewma_state is not None:
            monitor.ewma = util.ExpMovingAve()
            monitor.ewma.__dict__.update(ewma_state)
        monitor.__dict__.update(**state)

    def state_dict(monitor):
        return monitor.__getstate__()

    def load_state_dict(monitor, state):
        return monitor.__setstate__(state)

    def update(monitor, epoch, raw_metrics):
        monitor.epochs.append(epoch)
        monitor.raw_metrics.append(raw_metrics)
        monitor.ewma.update(raw_metrics)
        # monitor.other_data.append(other)

        smooth_metrics = monitor.ewma.average()
        monitor.smooth_metrics.append(smooth_metrics.copy())

        improved = monitor._improved(smooth_metrics, monitor.best_smooth_metrics)
        if improved:
            monitor.best_smooth_metrics = smooth_metrics
            monitor.best_raw_metrics = raw_metrics
            monitor.best_epoch = epoch
            monitor.n_bad_epochs = 0
        else:
            monitor.n_bad_epochs += 1
        monitor.is_good.append(improved)
        return improved

    def _improved(monitor, metrics, best_metrics):
        """
        If any of the metrics we care about is improving then we are happy

        Example:
            >>> from clab.monitor import *
            >>> monitor = Monitor(['loss'], ['acc'])
            >>> metrics = {'loss': 5, 'acc': .99}
            >>> best_metrics = {'loss': 4, 'acc': .98}
        """
        def _as_minimization(metrics):
            # convert to a minimization problem
            max_metrics = list(ub.take(metrics, monitor.max_keys))
            min_metrics = list(ub.take(metrics, monitor.min_keys))
            sign = np.array(([-1] * len(max_metrics)) + ([1] * len(min_metrics)))
            chosen = np.array(max_metrics + min_metrics)
            return chosen, sign

        if not best_metrics:
            return True

        current, sign1 = _as_minimization(metrics)
        best, sign2 = _as_minimization(best_metrics)

        # only use threshold rel mode
        rel_epsilon = 1.0 - monitor.rel_threshold
        improved_flags = (sign1 * current) < (sign2 * best) * rel_epsilon
        improved = np.any(improved_flags)
        return improved

    def is_done(monitor):
        return monitor.n_bad_epochs >= monitor.patience

    def message(monitor):
        if not monitor.epochs:
            return 'vloss is unevaluated'
        # if monitor.is_improved():
            # message = 'vloss: {:.4f} (new_best)'.format(monitor.best_loss)

        prev_loss = monitor.smooth_metrics[-1]['loss']
        best_loss = monitor.best_smooth_metrics['loss']

        message = 'vloss: {:.4f} (n_bad_epochs={:2d}, best={:.4f})'.format(
            prev_loss, monitor.n_bad_epochs, best_loss,
        )
        if monitor.n_bad_epochs <= int(monitor.patience * .25):
            message = ub.color_text(message, 'green')
        elif monitor.n_bad_epochs >= int(monitor.patience * .75):
            message = ub.color_text(message, 'red')
        else:
            message = ub.color_text(message, 'yellow')
        return message
