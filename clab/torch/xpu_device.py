import ubelt as ub
import torch
from clab.torch import nnio
from clab.util import gpu_util


class XPU(ub.NiceRepr):
    """
    A processing device, either a GPU or CPU
    """
    def __init__(xpu, gpu_num=None):
        xpu.gpu_num = xpu._cast_as_int(gpu_num)

    def _cast_as_int(xpu, other):
        if other is None:
            return None
        elif isinstance(other, int):
            return other
        else:
            return other.num
        return None if other is None else int(other)

    def __str__(self):
        return self.__nice__()

    def __nice__(self):
        if self.is_gpu():
            return 'GPU({})'.format(self.num)
        else:
            return 'CPU'

    @property
    def num(xpu):
        return xpu.gpu_num

    def is_gpu(xpu):
        return xpu.gpu_num is not None

    def __int__(xpu):
        return xpu.num

    @classmethod
    def available(XPU, min_memory=6000):
        gpu_num = gpu_util.find_unused_gpu(min_memory=min_memory)
        xpu = XPU(gpu_num)
        return xpu

    @classmethod
    def from_argv(XPU, **kwargs):
        """
        Respect command line gpu argument
        """
        gpu_num = ub.argval('--gpu', default=None)
        if gpu_num is None:
            xpu = XPU.available(**kwargs)
        else:
            if gpu_num.lower() == 'none':
                xpu = XPU(None)
            else:
                xpu = XPU(int(gpu_num))
        return xpu

    def to_xpu(xpu, data):
        """
        Args:
            data (torch.Tensor): raw data
        """
        # if False:
        #     harn.data = torch.nn.DataParallel(harn.data, device_ids=[3, 2]).cuda()
        if xpu.is_gpu():
            return data.cuda(xpu.gpu_num)
        else:
            return data.cpu()

    def to_xpu_var(xpu, *args, **kw):
        """
        Puts data on this XPU device

        Args:
            *args: list of tensors to put data
            **kwargs: Mainly used for volatile, forwarded to `torch.autograd.Variable`.
        """
        if xpu.is_gpu():
            args = [torch.autograd.Variable(item.cuda(xpu.gpu_num), **kw) for item in args]
        else:
            args = [torch.autograd.Variable(item.cpu(), **kw) for item in args]
        return args

    def set_as_default(xpu):
        """ Sets this device as the default torch GPU """
        if xpu.is_gpu():
            torch.cuda.set_device(xpu.num)

    def map_location(xpu):
        return nnio.device_mapping(xpu.num)

    def load(xpu, fpath):
        print('Loading data onto {} from {}'.format(xpu, fpath))
        return torch.load(fpath, map_location=xpu.map_location())
