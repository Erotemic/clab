import ubelt as ub
import collections


class EarlyStop(object):
    """

    TODO: Poisson based early stop

    TODO: smoothing

    Example:
        >>> early_stop = EarlyStop()
        >>> early_stop.update(1, .5)
        >>> early_stop.update(2, .4)
        >>> early_stop.update(3, .6)
        >>> early_stop.update(4, .3)
        >>> early_stop.update(5, .31)
        >>> early_stop.update(6, .2)
        >>> early_stop.best_epochs()
        >>> print('Best epochs / loss: {}'.format(ub.repr2(list(early_stop.memory), nl=1, precision=6)))
    """
    def __init__(early_stop, patience=10):
        # import sortedcontainers
        # store tuples of (loss, epoch)

        n_remember = 3
        early_stop.memory = collections.deque(maxlen=n_remember)
        # util.SortedQueue(maxsize=3)

        early_stop.prev_epoch = None
        early_stop.prev_loss = None

        early_stop.best_epoch = None
        early_stop.best_loss = None

        early_stop.patience = patience
        early_stop.n_bad_epochs = 0

    def update(early_stop, epoch, loss):
        early_stop.prev_epoch = epoch
        early_stop.prev_loss = loss

        # Dont allow overfitting epochs to be recorded as top-epochs
        if len(early_stop.memory) == 0:
            early_stop.memory.appendleft((epoch, loss))
        elif loss < early_stop.memory[0][1]:
            # TODO: delete snapshots as they become irrelevant
            early_stop.memory.appendleft((epoch, loss))

        # if len(early_stop.memory) == early_stop.memory.maxsize:
        #     if loss < early_stop.memory.peek()[1]:
        #         [epoch] = loss
        # early_stop.memory[epoch] = loss

        if early_stop.best_loss is None or loss < early_stop.best_loss:
            early_stop.best_loss = loss
            early_stop.best_epoch = epoch
            early_stop.n_bad_epochs = 0
        else:
            early_stop.n_bad_epochs += 1

    def is_improved(early_stop):
        """
        returns True if the last update improved the validation loss
        """
        return early_stop.n_bad_epochs == 0

    def is_done(early_stop):
        return early_stop.n_bad_epochs >= early_stop.patience

    def best_epochs(early_stop):
        return [epoch for epoch, loss in early_stop.memory]

    def message(early_stop):
        if early_stop.prev_epoch is None:
            return 'vloss is unevaluated'
        # if early_stop.is_improved():
            # message = 'vloss: {:.4f} (new_best)'.format(early_stop.best_loss)
        message = 'vloss: {:.4f} (n_bad_epochs={:2d}, best={:.4f})'.format(
            early_stop.prev_loss, early_stop.n_bad_epochs,
            early_stop.best_loss,
        )
        if early_stop.n_bad_epochs <= int(early_stop.patience * .25):
            message = ub.color_text(message, 'green')
        elif early_stop.n_bad_epochs >= int(early_stop.patience * .75):
            message = ub.color_text(message, 'red')
        else:
            message = ub.color_text(message, 'yellow')
        return message
