import torch
from torch import nn
from clab import util
import numpy as np


def default_nonlinearity():
    return nn.LeakyReLU(negative_slope=1e-2, inplace=True)


def logsumexp(x, dim=1):
    """
    This is a numerically stable log(sum(exp(x))) operator

    Example:
        >>> x = torch.autograd.Variable(torch.rand(2048, 1000), volatile=True)
        >>> stable_result = logsumexp(x)
        >>> naive_result = torch.log(torch.sum(torch.exp(x), dim=1))
        >>> diff = (stable_result - naive_result).sum()
        >>> print('Difference between naive and stable logsumexp: {}'.format(diff.data[0]))
    """
    max_x, _ = torch.max(x, dim=dim, keepdim=True)
    part = torch.log(torch.sum(torch.exp(x - max_x), dim=dim, keepdim=True))
    return max_x + part


class BatchRenorm2d(nn.Module):
    """

    References:
        https://discuss.pytorch.org/t/computing-the-gradients-for-batch-renormalization/828

    >>> inputs = torch.autograd.Variable(torch.randn((4, 32, 24, 24)))
    >>> self = BatchRenorm2d(inputs.shape[1])
    >>> self(inputs)

    """
    def __init__(self, channels, eps=1e-5, rmax=3, dmax=5, lr=0.001):
        util.super2(BatchRenorm2d, self).__init__()
        self.training = True
        self.is_unlock = False
        self.eps = eps
        self.eps_sqrt = np.sqrt(self.eps)
        self.channels = channels
        self.rmax = rmax
        self.dmax = dmax
        self.lr = lr
        self.use_cuda = True
        self.sigma = torch.ones((1, channels)).float()
        self.mean = torch.zeros((1, channels)).float()

    def forward(self, inputs):
        if self.training:
            batch_size = inputs.size()[0]
            feature_shape_size = inputs.size()[2] * inputs.size()[3]

            _ = torch.zeros(batch_size, self.channels)
            if self.use_cuda:
                _ = _.cuda()
            sig_sqr_sum = torch.autograd.Variable(_)

            mu_b = inputs.mean(0, keepdim=True).mean(-1).mean(-1)
            xview = inputs.view(batch_size, self.channels, feature_shape_size)

            for j in range(self.channels):
                mu_b_0_j = mu_b[0, j].repeat(feature_shape_size)
                for i in range(batch_size):
                    sig_sqr_sum[i, j] = ((xview[i, j] - mu_b_0_j) ** 2).mean()

            sigma_b = sig_sqr_sum.mean(0, keepdim=True)
            sigma_b += self.eps
            sigma_b = torch.sqrt(sigma_b)
            if self.is_unlock:
                r = sigma_b.data / self.sigma
                r.clamp_(1.0 / self.rmax, self.rmax)
                d = (mu_b.data - self.mean) / (self.sigma + self.eps_sqrt)
                d.clamp_(-self.dmax, self.dmax)
            else:
                r = torch.zeros(1, self.channels) + 1.0
                d = torch.zeros(1, self.channels)

            _ = torch.zeros(inputs.size())
            if self.use_cuda:
                _ = _.cuda()
            x_hat = torch.autograd.Variable(_)

            for j in range(self.channels):
                mu_b_0_j = mu_b[0, j].repeat(
                    feature_shape_size).view(inputs.size()[2], inputs.size()[3])
                sigma_b_0_j = sigma_b[0, j].repeat(
                    feature_shape_size).view(inputs.size()[2], inputs.size()[3])
                for i in range(batch_size):
                    x_hat_i_j = inputs[i, j, :, :].clone()
                    x_hat_i_j -= mu_b_0_j
                    x_hat_i_j /= sigma_b_0_j
                    x_hat_i_j *= r[0, j]
                    x_hat_i_j += d[0, j]
                    x_hat[i, j, :, :] = x_hat_i_j
                    self.mean += self.lr * (mu_b.data - self.mean)
            self.sigma += self.lr * (sigma_b.data - self.sigma)
        else:
            mu_b = torch.autograd.Variable(self.mean)
            sigma_b = torch.autograd.Variable(self.sigma)
            for j in range(self.channels):
                mu_b_0_j = mu_b[0, j].repeat(
                    feature_shape_size).view(inputs.size()[2], inputs.size()[3])
                sigma_b_0_j = sigma_b[0, j].repeat(
                    feature_shape_size).view(inputs.size()[2], inputs.size()[3])
                for i in range(batch_size):
                    x_hat_i_j = inputs[i, j, :, :].clone()
                    x_hat_i_j -= mu_b_0_j
                    x_hat_i_j /= sigma_b_0_j
                    x_hat_i_j *= r[0, j]
                    x_hat_i_j += d[0, j]
                    x_hat[i, j, :, :] = x_hat_i_j
        return x_hat


class MixtureOfLogSoftmax(nn.Module):
    """
    Drop-in replacement for a Linear layer followed by a log_softmax.

    References:
        http://smerity.com/articles/2017/mixture_of_softmaxes.html
        https://arxiv.org/pdf/1711.03953.pdf
        https://colab.research.google.com/notebook#fileId=1Pvb0-Qdc-JjMVLihsUJwUztDiIFE9o3l&scrollTo=V1VHakggmRKF

    Example:
        >>> n_input = 100
        >>> n_output = 1000
        >>> n_mixtures = 2
        >>> n_samples = 2048
        >>> inputs = torch.autograd.Variable(torch.rand(n_samples, n_input), volatile=True)

        >>> # Using n_mixtures > 1 drastically increases the rank of the output batch for large # of classes
        >>> self = MixtureOfLogSoftmax(n_input, n_output, n_mixtures)
        >>> result = self(inputs)
        >>> print('mixed rank = {}'.format(np.linalg.matrix_rank(result.data.numpy(), tol=1e-3)))

        >>> # Using n_mixtures=1 is the same as 2 Linear layers followed by softmax
        >>> result = MixtureOfLogSoftmax(n_input, n_output, n_mixtures=1)(inputs)
        >>> print('naive rank = {}'.format(np.linalg.matrix_rank(result.data.numpy(), tol=1e-3)))
    """

    def __init__(self, n_input, n_output, n_mixtures=2):
        util.super2(MixtureOfLogSoftmax, self).__init__()
        self.n_mixtures = n_mixtures
        self.n_input = n_input
        self.n_output = n_output
        self.mixer = torch.nn.Linear(n_input, n_mixtures * n_input)
        self.project = torch.nn.Linear(n_input, n_output)
        self.noli = default_nonlinearity()

        # For the different components, we're assuming equal mixing
        # self.log_prior = torch.log(torch.autograd.Variable(torch.ones(1, n_mixtures, 1) * (1 / n_mixtures)))
        self.log_prior = (1 / float(n_mixtures))

    def forward(self, inputs):
        mixed = self.noli(self.mixer(inputs))
        parts = torch.chunk(mixed, self.n_mixtures, dim=1)

        results = []
        for part in parts:
            # For each part project it into a new space and take the softmax
            part_logit = self.project(part)
            part_logprob = torch.nn.functional.log_softmax(part_logit, dim=1)
            results.append(part_logprob)

        # Take a linear combination of the projected softmaxes
        # * transform into a [B x M x C] tensor
        result = torch.cat(results, dim=1).view(-1,
                                                self.n_mixtures, self.n_output)
        # * Weight each mixture by its respective prior weight.
        #   (note: because we're in log-prob-space, adding the logpriors is
        #    equivalent to multiplication in prob-space)
        result = self.log_prior + result

        # Add the weighted components to obtain a [B x C] vector of log-probs
        log_probs = logsumexp(result).view(-1, self.n_output)
        return log_probs

    def output_shape_for(self, input_shape):
        """
        input_shape = (1, 1, 16, 16, 16)
        """
        return [input_shape[0], self.n_output]


class LinearLogSoftmax(nn.Module):
    """
    A Linear layer followed by a log_softmax.

    This is the naive alternative to a MixtureOfLogSoftmax.

    References:
        http://smerity.com/articles/2017/mixture_of_softmaxes.html
        https://arxiv.org/pdf/1711.03953.pdf
        https://colab.research.google.com/notebook#fileId=1Pvb0-Qdc-JjMVLihsUJwUztDiIFE9o3l&scrollTo=V1VHakggmRKF

    Example:
        >>> n_input = 100
        >>> n_output = 1000
        >>> n_mixtures = 2
        >>> n_samples = 2048
        >>> inputs = torch.autograd.Variable(torch.rand(n_samples, n_input), volatile=True)
        >>> # Using n_mixtures=1 is the same as a Linear layer followed by softmax
        >>> result = LinearLogSoftmax(n_input, n_output)(inputs)
        >>> print('naive rank = {}'.format(np.linalg.matrix_rank(result.data.numpy(), tol=1e-3)))
    """

    def __init__(self, n_input, n_output):
        util.super2(LinearLogSoftmax, self).__init__()
        self.n_input = n_input
        self.n_output = n_output
        self.project = torch.nn.Linear(n_input, n_output)
        self.noli = default_nonlinearity()

    def forward(self, inputs):
        logits = self.project(inputs)
        log_probs = torch.nn.functional.log_softmax(logits, dim=1)
        return log_probs

    def output_shape_for(self, input_shape):
        """
        input_shape = (1, 1, 16, 16, 16)
        """
        return [input_shape[0], self.n_output]
