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


# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import os
from os.path import join as pjoin
import numpy as np
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext


def find_in_path(name, path):
    """
    Find a file in a search path
    adapted fom
    http://code.activestate.com/recipes/52224-find-a-file-given-a-search-path/
    """
    for dir in path.split(os.pathsep):
        binpath = pjoin(dir, name)
        if os.path.exists(binpath):
            return os.path.abspath(binpath)
    return None


def locate_cuda():
    """Locate the CUDA environment on the system

    Returns a dict with keys 'home', 'nvcc', 'include', and 'lib64'
    and values giving the absolute path to each directory.

    Starts by looking for the CUDAHOME env variable. If not found, everything
    is based on finding 'nvcc' in the PATH.
    """

    # first check if the CUDAHOME env variable is in use
    if 'CUDAHOME' in os.environ:
        home = os.environ['CUDAHOME']
        nvcc = pjoin(home, 'bin', 'nvcc')
    else:
        # otherwise, search the PATH for NVCC
        default_path = pjoin(os.sep, 'usr', 'local', 'cuda', 'bin')
        nvcc = find_in_path('nvcc',
                            os.environ['PATH'] + os.pathsep + default_path)
        if nvcc is None:
            raise EnvironmentError('The nvcc binary could not be '
                                   'located in your $PATH. '
                                   'Either add it to your path, '
                                   'or set $CUDAHOME')
        home = os.path.dirname(os.path.dirname(nvcc))

    cudaconfig = {'home': home, 'nvcc': nvcc,
                  'include': pjoin(home, 'include'),
                  'lib64': pjoin(home, 'lib64')}
    for k, v in cudaconfig.items():
        if not os.path.exists(v):
            raise EnvironmentError('The CUDA %s path could not '
                                   'be located in %s' % (k, v))

    return cudaconfig


CUDA = locate_cuda()

# Obtain the numpy include directory.  This logic works across numpy versions.
try:
    numpy_include = np.get_include()
except AttributeError:
    numpy_include = np.get_numpy_include()


def customize_compiler_for_nvcc(self):
    """inject deep into distutils to customize how the dispatch
    to gcc/nvcc works.

    If you subclass UnixCCompiler, it's not trivial to get your subclass
    injected in, and still have the right customizations (i.e.
    distutils.sysconfig.customize_compiler) run on it. So instead of going
    the OO route, I have this. Note, it's kindof like a wierd functional
    subclassing going on."""

    # tell the compiler it can processes .cu
    self.src_extensions.append('.cu')

    # save references to the default compiler_so and _comple methods
    default_compiler_so = self.compiler_so
    super = self._compile

    # now redefine the _compile method. This gets executed for each
    # object but distutils doesn't have the ability to change compilers
    # based on source extension: we add it.
    def _compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
        print(extra_postargs)
        if os.path.splitext(src)[1] == '.cu':
            # use the cuda for .cu files
            self.set_executable('compiler_so', CUDA['nvcc'])
            # use only a subset of the extra_postargs, which are 1-1 translated
            # from the extra_compile_args in the Extension class
            postargs = extra_postargs['nvcc']
        else:
            postargs = extra_postargs['gcc']

        super(obj, src, ext, cc_args, postargs, pp_opts)
        # reset the default compiler_so, which we might have changed for cuda
        self.compiler_so = default_compiler_so

    # inject our redefined _compile method into the class
    self._compile = _compile


# run the customize_compiler
class custom_build_ext(build_ext):
    def build_extensions(self):
        customize_compiler_for_nvcc(self.compiler)
        build_ext.build_extensions(self)


ext_modules = [
    Extension(
        "clab.models.yolo2.utils.nms.cython_bbox",
        ["clab/models/yolo2/utils/nms/cython_bbox.pyx"],
        extra_compile_args={'gcc': ["-Wno-cpp", "-Wno-unused-function"]},
        include_dirs=[numpy_include]
    ),
    Extension(
        "clab.models.yolo2.utils.nms.cython_yolo",
        ["clab/models/yolo2/utils/nms/cython_yolo.pyx"],
        extra_compile_args={'gcc': ["-Wno-cpp", "-Wno-unused-function"]},
        include_dirs=[numpy_include]
    ),
    Extension(
        "clab.models.yolo2.utils.nms.cpu_nms",
        ["clab/models/yolo2/utils/nms/cpu_nms.pyx"],
        extra_compile_args={'gcc': ["-Wno-cpp", "-Wno-unused-function"]},
        include_dirs=[numpy_include]
    ),
    Extension('nms.gpu_nms',
              ['nms/nms_kernel.cu', 'nms/gpu_nms.pyx'],
              library_dirs=[CUDA['lib64']],
              libraries=['cudart'],
              language='c++',
              runtime_library_dirs=[CUDA['lib64']],
              # this syntax is specific to this build system
              # we're only going to use certain compiler args with
              # nvcc and not with gcc
              # the implementation of this trick is in
              # customize_compiler() below
              extra_compile_args={'gcc': ["-Wno-unused-function"],
                                  'nvcc': ['-arch=sm_35',
                                           '--ptxas-options=-v',
                                           '-c',
                                           '--compiler-options',
                                           "'-fPIC'"]},
              include_dirs=[numpy_include, CUDA['include']]
              ),
    Extension(
        'pycocotools._mask',
        sources=['pycocotools/maskApi.c', 'pycocotools/_mask.pyx'],
        include_dirs=[numpy_include, 'pycocotools'],
        extra_compile_args={
            'gcc': ['-Wno-cpp', '-Wno-unused-function', '-std=c99']},
    ),
]

setup(
    name='mot_utils',
    ext_modules=ext_modules,
    # inject our custom trigger
    cmdclass={'build_ext': custom_build_ext},
)


if __name__ == '__main__':
    main()
