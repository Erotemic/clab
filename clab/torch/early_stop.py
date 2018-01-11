import ubelt as ub
import collections
from clab.torch.metrics import ExpMovingAve


class EarlyStop(object):
    """

    TODO: Poisson based early stop

    TODO: smoothing

    Example:
        >>> stopping = EarlyStop()
        >>> stopping.update(1, .5)
        >>> stopping.update(2, .4)
        >>> stopping.update(3, .6)
        >>> stopping.update(4, .3)
        >>> stopping.update(5, .31)
        >>> stopping.update(6, .2)
        >>> stopping.best_epochs()
        >>> print('Best epochs / loss: {}'.format(ub.repr2(list(stopping.memory), nl=1, precision=6)))
    """
    def __init__(stopping, patience=10):
        # import sortedcontainers
        # store tuples of (loss, epoch)

        n_remember = 3
        stopping.memory = collections.deque(maxlen=n_remember)
        stopping.ewma = ExpMovingAve(alpha=.6)

        stopping.raw_loss = []
        stopping.smooth_loss = []
        stopping.epochs = []
        # util.SortedQueue(maxsize=3)

        stopping.prev_epoch = None
        stopping.prev_loss = None

        stopping.best_epoch = None
        stopping.best_loss = None

        stopping.patience = patience
        stopping.n_bad_epochs = 0

    def update(stopping, epoch, loss):

        stopping.ewma.update({'loss': loss})
        smooth = stopping.ewma.average()['loss']
        # hack overwrite loss with a smoothed version
        loss = smooth

        stopping.raw_loss.append(loss)
        stopping.smooth_loss.append(smooth)
        stopping.epochs.append(epoch)

        stopping.prev_epoch = epoch
        stopping.prev_loss = loss

        # Dont allow overfitting epochs to be recorded as top-epochs
        if len(stopping.memory) == 0:
            stopping.memory.appendleft((epoch, loss))
        elif loss < stopping.memory[0][1]:
            # TODO: delete snapshots as they become irrelevant
            stopping.memory.appendleft((epoch, loss))

        # if len(stopping.memory) == stopping.memory.maxsize:
        #     if loss < stopping.memory.peek()[1]:
        #         [epoch] = loss
        # stopping.memory[epoch] = loss

        if stopping.best_loss is None or loss < stopping.best_loss:
            stopping.best_loss = loss
            stopping.best_epoch = epoch
            stopping.n_bad_epochs = 0
        else:
            stopping.n_bad_epochs += 1

    def is_improved(stopping):
        """
        returns True if the last update improved the validation loss
        """
        return stopping.n_bad_epochs == 0

    def is_done(stopping):
        return stopping.n_bad_epochs >= stopping.patience

    def best_epochs(stopping):
        return [epoch for epoch, loss in stopping.memory]

    def message(stopping):
        if stopping.prev_epoch is None:
            return 'vloss is unevaluated'
        # if stopping.is_improved():
            # message = 'vloss: {:.4f} (new_best)'.format(stopping.best_loss)
        message = 'vloss: {:.4f} (n_bad_epochs={:2d}, best={:.4f})'.format(
            stopping.prev_loss, stopping.n_bad_epochs,
            stopping.best_loss,
        )
        if stopping.n_bad_epochs <= int(stopping.patience * .25):
            message = ub.color_text(message, 'green')
        elif stopping.n_bad_epochs >= int(stopping.patience * .75):
            message = ub.color_text(message, 'red')
        else:
            message = ub.color_text(message, 'yellow')
        return message
