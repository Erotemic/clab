#!/usr/local/env python
"""
This belongs in a root setup.py
"""
from os.path import join, dirname, exists
import torch
import ubelt as ub
from torch.utils.ffi import create_extension
import os


class ChdirContext(object):
    """
    References http://www.astropython.org/snippet/2009/10/chdir-context-manager
    """
    def __init__(self, dpath=None, verbose=1):
        self.verbose = verbose
        self.dpath = dpath
        self.curdir = os.getcwd()

    def __enter__(self):
        if self.dpath is not None:
            if self.verbose:
                print('[path.push] Change directory to %r' % (self.dpath,))
            os.chdir(self.dpath)
        return self

    def __exit__(self, type_, value, trace):
        if self.verbose:
            print('[path.pop] Change directory to %r' % (self.curdir,))
        os.chdir(self.curdir)
        if trace is not None:
            if self.verbose:
                print('[util_path] Error in chdir context manager!: ' + str(value))
            return False


def build_roi_pooling_extension(root, with_cuda):
    path = join(root, 'layers/roi_pooling')
    print('BUILDING = {!r}'.format(path))

    sources = [join(path, 'src/roi_pooling.c')]
    headers = [join(path, 'src/roi_pooling.h')]
    defines = []

    if with_cuda:

        with ChdirContext(join(path, 'src/cuda')):
            info = ub.cmd('nvcc -c -o roi_pooling_kernel.cu.o roi_pooling_kernel.cu -x cu -Xcompiler -fPIC -arch=sm_52', verbout=1, verbose=2)
        if info['ret'] != 0:
            raise Exception('Failed to build roi pooling extension')

        sources += [join(path, 'src/roi_pooling_cuda.c')]
        headers += [join(path, 'src/roi_pooling_cuda.h')]
        defines += [('WITH_CUDA', None)]

    # needs to have been compiled by nvcc first
    extra_objects = [join(path, 'src/cuda/roi_pooling_kernel.cu.o')]
    for fpath in extra_objects:
        if not exists(fpath):
            raise Exception('Extra object {} does not exist'.format(fpath))

    ffi = create_extension(
        '_ext.roi_pooling',
        headers=headers,
        sources=sources,
        define_macros=defines,
        relative_to=path,
        with_cuda=with_cuda,
        extra_objects=extra_objects
    )
    ffi.build()


def build_reorg_extension(root, with_cuda):
    path = join(root, 'layers/reorg')
    print('BUILDING = {!r}'.format(path))

    sources = [join(path, 'src/reorg_cpu.c')]
    headers = [join(path, 'src/reorg_cpu.h')]
    defines = []

    if with_cuda:
        with ChdirContext(join(path, 'src')):
            info = ub.cmd('nvcc -c -o reorg_cuda_kernel.cu.o reorg_cuda_kernel.cu -x cu -Xcompiler -fPIC -arch=sm_52', verbout=1, verbose=2)
        if info['ret'] != 0:
            raise Exception('Failed to build reorg extension')

        sources += [join(path, 'src/reorg_cuda.c')]
        headers += [join(path, 'src/reorg_cuda.h')]
        defines += [('WITH_CUDA', None)]

    # needs to have been compiled by nvcc first
    extra_objects = [join(path, 'src/reorg_cuda_kernel.cu.o')]
    for fpath in extra_objects:
        if not exists(fpath):
            raise Exception('Extra object {} does not exist'.format(fpath))

    ffi = create_extension(
        '_ext.reorg_layer',
        headers=headers,
        sources=sources,
        define_macros=defines,
        relative_to=path,
        with_cuda=with_cuda,
        extra_objects=extra_objects
    )
    ffi.build()


def main():
    with_cuda = torch.cuda.is_available()

    # import ubelt as ub
    # root = ub.truepath('~/code/clab/clab/models/yolo2')
    root = os.path.abspath(dirname(__file__))

    build_reorg_extension(root, with_cuda)
    build_roi_pooling_extension(root, with_cuda)


if __name__ == '__main__':
    main()
