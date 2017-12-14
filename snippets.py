# flake8: noqa


def persample_nll_loss(log_probs, target):
    """
    Args:
        logits: A Variable containing a FloatTensor of size
            (batch, max_len, num_classes) which contains the
            unnormalized probability for each class.
        target: A Variable containing a LongTensor of size
            (batch, max_len) which contains the index of the true
            class for each corresponding step.

    References:
        https://gist.github.com/jihunchoi/f1434a77df9db1bb337417854b398df1

    Returns:
        loss: An average loss value masked by the length.

    Example:
        >>> B, C, D, H, W, n_classes = 32, 1, 15, 15, 15, 2
        >>> logits = Variable(torch.FloatTensor(
        >>>     [[.1, .9], [.1, .9], [.2, .8], [.2, .8], [0, 1], [0, 1]]))
        >>> target = Variable(torch.LongTensor([0, 1, 0, 1, 0, 1]))
        >>> log_probs = torch.nn.functional.log_softmax(logits, dim=1)
        >>> losses = persample_nll_loss(log_probs, target)
        >>> ave_loss = torch.nn.functional.nll_loss(log_probs, target)
        >>> assert np.isclose(ave_loss.data.numpy(), losses.mean().data.numpy())

    Ignore:
        torch.nn.functional.softmax(Variable(torch.FloatTensor([[0, 1]])), dim=1)
        torch.nn.functional.softmax(Variable(torch.FloatTensor([[0, 2]])), dim=1)
        torch.nn.functional.softmax(Variable(torch.FloatTensor([[0, 100]])), dim=1)
        torch.nn.functional.softmax(Variable(torch.FloatTensor([[0, 10]])), dim=1)
    """
    target_ = target.view(-1, 1)
    losses = -torch.gather(log_probs, dim=1, index=target_)
    return losses
