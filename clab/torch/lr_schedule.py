
class BaseLRSchedule(object):
    def _update_optimizer(self, lr, optimizer=None):
        if optimizer is not None:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr


class Constant(BaseLRSchedule):
    def __init__(self, base_lr=0.001):
        self.base_lr = base_lr

    def __call__(self, epoch, optimizer=None):
        """
        If optimizer is specified, its learning rate is modified inplace.
        """
        lr = self.base_lr
        self._update_optimizer(lr, optimizer)
        return lr


class Exponential(BaseLRSchedule):
    """
    Decay learning rate by a factor of `gamma` every `stepsize` epochs.

    Example:
        >>> from ibeis.algo.verif.torch.lr_schedule import *
        >>> from clab.torch.lr_schedule import *
        >>> lr_scheduler = Exponential(stepsize=2)
        >>> rates = np.array([lr_scheduler(i) for i in range(6)])
        >>> target = np.array([1E-3, 1E-3, 1E-5, 1E-5, 1E-7, 1E-7])
        >>> assert all(list(np.isclose(target, rates)))
    """
    def __init__(self, base_lr=0.001, gamma=1.0, stepsize=100):
        self.base_lr = base_lr
        self.gamma = gamma
        self.stepsize = stepsize

    def __call__(self, epoch, optimizer=None):
        """
        If optimizer is specified, its learning rate is modified inplace.
        """
        n_decays = epoch // self.stepsize
        lr = self.base_lr * (self.gamma ** n_decays)
        self._update_optimizer(lr, optimizer)
        return lr
