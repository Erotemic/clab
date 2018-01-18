import ubelt as ub
import collections
from clab.metrics import ExpMovingAve


class EarlyStop(object):
    """
    Rename to Training Monitor for early stop and backtracking
    Or pick a better name... not sure
    Should EarlyStop still be a class?

    TODO: Poisson based early stop

    TODO: smoothing

    Example:
        >>> monitor = EarlyStop()
        >>> monitor.update(1, .5)
        >>> monitor.update(2, .4)
        >>> monitor.update(3, .6)
        >>> monitor.update(4, .3)
        >>> monitor.update(5, .31)
        >>> monitor.update(6, .2)
        >>> monitor.best_epochs()
        >>> print('Best epochs / loss: {}'.format(ub.repr2(list(monitor.memory), nl=1, precision=6)))
    """
    def __init__(monitor, patience=10):
        monitor.ewma = ExpMovingAve(alpha=.6)
        monitor.raw_loss = []
        monitor.smooth_loss = []
        monitor.epochs = []

        n_remember = 3
        monitor.memory = collections.deque(maxlen=n_remember)
        monitor.prev_epoch = None
        monitor.prev_loss = None

        monitor.best_epoch = None
        monitor.best_loss = None

        monitor.patience = patience
        monitor.n_bad_epochs = 0

    def load_state_dict(monitor, state):
        return monitor.__dict__.update(state)
        # let some of the state be lost to force training for just a bit more
        monitor.n_bad_epochs = min(1, monitor.n_bad_epochs)
        # state = ub.dict_subset(state, ['ewma', 'raw_loss', 'smooth_loss', 'epochs'])

    def state_dict(self):
        return self.__dict__.copy()

    def update(monitor, epoch, loss):

        monitor.ewma.update({'loss': loss})
        smooth = monitor.ewma.average()['loss']
        # hack overwrite loss with a smoothed version

        monitor.raw_loss.append(loss)
        monitor.smooth_loss.append(smooth)
        monitor.epochs.append(epoch)

        monitor.prev_epoch = epoch
        monitor.prev_loss = smooth

        # Dont allow overfitting epochs to be recorded as top-epochs
        if len(monitor.memory) == 0:
            monitor.memory.appendleft((epoch, smooth))
        elif smooth < monitor.memory[0][1]:
            # TODO: delete snapshots as they become irrelevant
            monitor.memory.appendleft((epoch, smooth))

        if monitor.best_loss is None or smooth < monitor.best_loss:
            monitor.best_loss = smooth
            monitor.best_epoch = epoch
            monitor.n_bad_epochs = 0
        else:
            monitor.n_bad_epochs += 1

    def is_improved(monitor):
        """
        returns True if the last update improved the validation loss
        """
        return monitor.n_bad_epochs == 0

    def is_done(monitor):
        return monitor.n_bad_epochs >= monitor.patience

    def best_epochs(monitor):
        return [epoch for epoch, loss in monitor.memory]

    def message(monitor):
        if monitor.prev_epoch is None:
            return 'vloss is unevaluated'
        # if monitor.is_improved():
            # message = 'vloss: {:.4f} (new_best)'.format(monitor.best_loss)
        message = 'vloss: {:.4f} (n_bad_epochs={:2d}, best={:.4f})'.format(
            monitor.prev_loss, monitor.n_bad_epochs,
            monitor.best_loss,
        )
        if monitor.n_bad_epochs <= int(monitor.patience * .25):
            message = ub.color_text(message, 'green')
        elif monitor.n_bad_epochs >= int(monitor.patience * .75):
            message = ub.color_text(message, 'red')
        else:
            message = ub.color_text(message, 'yellow')
        return message
