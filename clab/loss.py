import torch  # NOQA
import torch.nn as nn
import torch.nn.functional as F

# from utils import one_hot_embedding
from torch import autograd


def one_hot_embedding(labels, num_classes):
    """
    Embedding labels to one-hot form.

    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.

    Returns:
      (tensor) encoded labels, sized [N,#classes].
    """
    y = torch.eye(num_classes)  # [D,D]
    return y[labels]            # [N,D]


class FocalLoss(nn.Module):
    """
    Args:
        size_average (bool, optional): By default, the losses are averaged
           over observations for each minibatch. However, if the field
           size_average is set to ``False``, the losses are instead summed for
           each minibatch. Ignored when reduce is ``False``. Default: ``True``
        reduce (bool, optional): By default, the losses are averaged or summed
           for each minibatch. When reduce is ``False``, the loss function returns
           a loss per batch element instead and ignores size_average.
           Default: ``True``

    References:
        https://github.com/kuangliu/pytorch-retinanet/blob/master/loss.py
    """
    def __init__(self, size_average=True, reduce=True):
        super(FocalLoss, self).__init__()
        self.size_average = size_average
        self.reduce = reduce

    def focal_loss(self, input, target):
        """
        Focal loss.

        Args:
          input: (tensor) sized [N,D].
          target: (tensor) sized [N,].

        Return:
          (tensor) focal loss.
        """
        alpha = 0.25
        gamma = 2

        num_classes = input.shape[1]

        # t = one_hot_embedding(target.data.cpu(), 1 + num_classes)  # [N,21]
        # t = t[:, 1:]  # exclude background
        # t = autograd.Variable(t).cuda()  # [N,20]
        eye = torch.eye(num_classes)
        from clab import xpu_device
        eye = xpu_device.XPU.cast(target).move(eye)
        t = eye[target.data]
        t = autograd.Variable(t)

        p = input.sigmoid()
        pt = p * t + (1 - p) * (1 - t)         # pt = p if t > 0 else 1-p
        # w = alpha if t > 0 else 1-alpha
        w = alpha * t + (1 - alpha) * (1 - t)
        w = w * (1 - pt).pow(gamma)
        output = F.binary_cross_entropy_with_logits(input, t, w, size_average=False)
        return output

    def focal_loss_alt(self, input, target):
        """
        Focal loss alternative.

        Args:
          input: (tensor) sized [N,D].
          target: (tensor) sized [N,].

        Return:
          (tensor) sized [N,] focal loss for each class
        """
        alpha = 0.25
        num_classes = input.shape[1]

        eye = torch.eye(num_classes)
        from clab import xpu_device
        eye = xpu_device.XPU.cast(target).move(eye)
        t = eye[target.data]
        t = autograd.Variable(t)
        # if target.data.is_cuda:
        #     t = t.cuda(target.data.get_device())
        # t = one_hot_embedding(target.data.cpu(), 1 + num_classes)
        # t = t[:, 1:]   # exclude background

        xt = input * (2 * t - 1)  # xt = input if t > 0 else -input
        pt = (2 * xt + 1).sigmoid()

        w = alpha * t + (1 - alpha) * (1 - t)
        loss_parts = -w * pt.log() / 2
        output = loss_parts.sum(dim=1)
        return output

    def forward(self, input, target):
        """
        Args:
          input: (tensor) predicted class confidences, sized [batch_size, #classes].
          target: (tensor) encoded target labels, sized [batch_size].

        Returns:
            (tensor) loss

        Example:
            >>> loss = FocalLoss()
            >>> # input is of size N x C
            >>> N, C = 8, 5
            >>> data = autograd.Variable(torch.randn(N, C), requires_grad=True)
            >>> # each element in target has to have 0 <= value < C
            >>> target = autograd.Variable((torch.rand(N) * C).long())
            >>> input = nn.LogSoftmax(dim=1)(data)
            >>> loss.focal_loss_alt(input, target).sum()
            >>> loss.focal_loss(input, target)

            >>> output = loss(input, target)
            >>> output.backward()
        """
        output = self.focal_loss_alt(input, target)
        if self.reduce:
            output = output.sum()
            if self.size_average:
                output = output / input.shape[0]
        return output
