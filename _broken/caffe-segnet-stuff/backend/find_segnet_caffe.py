#!/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import os
import platform
import sys
import ubelt as ub
from os.path import expanduser, join, exists, abspath  # NOQA
from pysseg.util import gpu_util

from pysseg import getLogger
logger = getLogger(__name__)
print = logger.info


def get_segnet_caffe_python_root():
    """
    Returns the directory containing the segnet pycaffe module.

    TODO:
        generalize me
    """
    python_caffe_root = join(get_segnet_caffe_root(), 'python')
    if not exists(python_caffe_root):
        raise RuntimeError('python_segnet_caffe_root does not exist')
    print('python_caffe_root exists = {!r}'.format(python_caffe_root))
    return python_caffe_root


def get_segnet_caffe_root():
    """
    Returns the directory containing the segnet pycaffe module.

    TODO:
        generalize me
    """

    paths_to_check = [
        '~/code/caffe-segnet-cudnn5',
        '~/sseg/caffe-segnet',
        'caffe-segnet',
    ]
    if 'CAFFE_SEGNET_ROOT' in os.environ:
        paths_to_check.insert(0, os.environ['CAFFE_SEGNET_ROOT'])

    for path in paths_to_check:
        path = expanduser(path)
        if exists(path):
            caffe_root = path
            print('caffe_root exists = {!r}'.format(caffe_root))
            return caffe_root
    raise RuntimeError('segnet-caffe-root does not exist')


def find_segnet_caffe_bin():
    if 'CAFFE_SEGNET_BIN' in os.environ:
        if exists(os.environ['CAFFE_SEGNET_BIN']):
            caffe_bin = os.environ['CAFFE_SEGNET_BIN']
            return caffe_bin
    caffe_bin = join(get_segnet_caffe_root(), 'build/tools/caffe')
    if not exists(caffe_bin):
        raise IOError('Please write a better caffe bin finder')
    return caffe_bin


def import_module_from_fpath(module_fpath):
    """
    imports module from a file path

    Args:
        module_fpath (str):

    Returns:
        module: module

    Example:
        >>> module_fpath = '/path/to/a/python_module'
        >>> module = import_module_from_fpath(module_fpath)
        >>> print('module = {!r}'.format(module))
    """
    from os.path import basename, splitext, isdir, join, exists, dirname, split
    if isdir(module_fpath):
        module_fpath = join(module_fpath, '__init__.py')
    print('importing module_fpath = {!r}'.format(module_fpath))
    if not exists(module_fpath):
        raise ImportError('module_fpath={!r} does not exist'.format(
            module_fpath))
    python_version = platform.python_version()
    modname = splitext(basename(module_fpath))[0]
    if modname == '__init__':
        modname = split(dirname(module_fpath))[1]
    if python_version.startswith('2.7'):
        import imp
        module = imp.load_source(modname, module_fpath)
    elif python_version.startswith('3'):
        import importlib.machinery
        loader = importlib.machinery.SourceFileLoader(modname, module_fpath)
        module = loader.load_module()
    else:
        raise AssertionError('invalid python version={!r}'.format(
            python_version))
    return module


CAFFE_SEGNET_MODULE = None


def import_segnet_caffe(gpu_num=ub.NoParam):
    """
    from pysseg.backend.find_segnet_caffe import get_segnet_caffe_python_root, PYTHONPATH_CONTEXT
    from pysseg.backend import find_segnet_caffe
    find_segnet_caffe.CAFFE_SEGNET_MODULE
    find_segnet_caffe.CAFFE_SEGNET_MODULE = None
    caffe = find_segnet_caffe.import_segnet_caffe()
    """
    global CAFFE_SEGNET_MODULE
    # TODO: should rename caffe modulename to segnet-caffe

    if gpu_num is ub.NoParam:
        if CAFFE_SEGNET_MODULE is None:
            gpu_num = gpu_util.find_unused_gpu()
        else:
            gpu_num = CAFFE_SEGNET_MODULE.GPU_NUM

    # Method 1
    if CAFFE_SEGNET_MODULE is None:
        print('Attempting to load segnet-caffe module')
        caffe_root = get_segnet_caffe_python_root()
        # sys.path.insert(0, caffe_root)

        # if 'caffe' in sys.modules:
        #     del sys.modules['caffe']
        #     for key in list(sys.modules.keys()):
        #         if key.startswith('caffe.'):
        #             del sys.modules[key]
        with PYTHONPATH_CONTEXT(caffe_root):
            import caffe
        CAFFE_SEGNET_MODULE = caffe

        # does pycaffe expose flags describing if it was built with GPU support?
        # ...probably not :(
        if gpu_num is None:
            print('setting caffe mode to CPU')
            CAFFE_SEGNET_MODULE.set_mode_cpu()
            CAFFE_SEGNET_MODULE.GPU_NUM = gpu_num
        else:
            print('setting caffe mode to GPU {}'.format(gpu_num))
            CAFFE_SEGNET_MODULE.set_mode_gpu()
            CAFFE_SEGNET_MODULE.set_device(gpu_num)
            CAFFE_SEGNET_MODULE.GPU_NUM = gpu_num
    else:
        pass
        # print('Return previous segnet-caffe module')

    if CAFFE_SEGNET_MODULE.GPU_NUM != gpu_num:
        if gpu_num is None:
            print('setting caffe mode to CPU')
            CAFFE_SEGNET_MODULE.set_mode_cpu()
            CAFFE_SEGNET_MODULE.GPU_NUM = gpu_num
        else:
            print('setting caffe mode to GPU {}'.format(gpu_num))
            CAFFE_SEGNET_MODULE.set_mode_gpu()
            CAFFE_SEGNET_MODULE.set_device(gpu_num)
            CAFFE_SEGNET_MODULE.GPU_NUM = gpu_num

    # Method 2
    # pycaffe_fpath = join(get_segnet_caffe_python_root(), 'caffe')
    # caffe = import_module_from_fpath(pycaffe_fpath)
    return CAFFE_SEGNET_MODULE


# METHOD 1
# Change this to the absolute directoy to SegNet Caffe
# caffe_path = join(get_segnet_caffe_root(), 'python')
# sys.path.insert(0, caffe_path)
# try:
#     import caffe
# except ImportError:
#     print('Caffe was not found in caffe_path = {!r}'.format(caffe_path))
#     raise


class PYTHONPATH_CONTEXT(object):
    """
    Attempt to be a little safer when mucking with the PYTHONPATH
    """
    def __init__(self, path):
        self.path = path

    def __enter__(self):
        # insert the path in to PYTHONPATH
        sys.path.insert(0, self.path)

    def __exit__(self, type_, value, trace):
        if sys.path[0] != self.path:
            raise RuntimeError('PYTHONPATH was changed inside this context')
        # Remove the path from PYTHONPATH
        del sys.path[0]
        if trace is not None:
            # return False on error
            return False  # nocover
