# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

from .nms.cpu_nms import cpu_nms
from .nms.gpu_nms import gpu_nms


# def nms(dets, thresh, force_cpu=False):
#     """Dispatch to either CPU or GPU NMS implementations."""
#
#     if dets.shape[0] == 0:
#         return []
#     if cfg.USE_GPU_NMS and not force_cpu:
#         return gpu_nms(dets, thresh, device_id=cfg.GPU_ID)
#     else:
#         return cpu_nms(dets, thresh)


def nms(dets, thresh, device=None):
    """
    Dispatch to either CPU or GPU NMS implementations.

    Example:
        >>> dets = np.array([
        >>>     [0, 0, 100, 100, .9],
        >>>     [100, 100, 10, 10, .1],
        >>>     [10, 10, 100, 100, .5],
        >>>     [50, 50, 100, 100, 1.0],
        >>> ], dtype=np.float32)
        >>> nms(dets, .5, device=None)
        >>> from clab import xpu_device
        >>> nms(dets, .5, device=xpu_device.XPU.default_gpu())
    """

    if dets.shape[0] == 0:
        return []
    if device is None:
        return cpu_nms(dets, thresh)
    return gpu_nms(dets, thresh)


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m xdoctest clab.models.yolo2.utils.nms_wrapper all
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
