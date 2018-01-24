# -*- coding: utf-8 -*-
"""
REFACTOR
"""
from __future__ import absolute_import, division, print_function
import cv2
import ubelt as ub
from os.path import join, expanduser, basename, splitext, abspath, exists  # NOQA
from PIL import Image
from clab import inputs
from clab.util import fnameutil  # NOQA
from clab.util import imutil
from clab.util import hashutil
import numpy as np

from clab import getLogger
logger = getLogger(__name__)
print = logger.info


class Preprocessor(object):
    """

    TODO: Rename to input chipper


    Class for adapting the resolution of input images into a size suitable for
    a network. Also handles on-disk data augmentation.

    Preprocessed data is written to subdirectories of the root `datadir`

    TODO: remove augmentation, doesn't belong here

    Ignore:
        sudo apt-get install gdal-bin
        sudo apt-get install libgdal-dev libgdal1h
        pip install GDAL==$(gdal-config --version) --global-option=build_ext --global-option="-I/usr/include/gdal"
    """

    def __init__(prep, datadir):
        prep.datadir = datadir
        prep.input_shape = (360, 480)
        prep.ignore_label = None
        prep.part_config = {
            'overlap': .25,
            'keepbound': True,
        }

    def subdir(prep, *args, **kw):
        dpath = join(*((prep.datadir,) + args))
        if kw.get('ensure', True):
            ub.ensuredir(dpath)
        return dpath

    def _mode_paths(prep, mode, input, clear=False):
        out_dpaths = {}
        if input.im_paths:
            out_dpaths['im'] = prep.subdir('im', mode)
        if input.gt_paths:
            out_dpaths['gt'] = prep.subdir('gt', mode)

        if input.aux_paths:
            out_dpaths['aux'] = {}
            for aux in input.aux_paths.keys():
                out_dpaths['aux'][aux] = prep.subdir('aux', aux, mode)
        if clear:
            # Start fresh. Remove existing files
            if 'im' in out_dpaths:
                ub.delete(out_dpaths['im'], verbose=False)
                ub.ensuredir(out_dpaths['im'])

            if 'gt' in out_dpaths:
                ub.delete(out_dpaths['gt'], verbose=False)
                ub.ensuredir(out_dpaths['gt'])

            if 'aux' in out_dpaths:
                for aux in out_dpaths['aux'].keys():
                    ub.delete(out_dpaths['aux'][aux], verbose=False)
                    ub.ensuredir(out_dpaths['aux'][aux])

        return out_dpaths

    def _mode_new_input(prep, mode, input, clear=False, mult=1):
        out_dpaths = prep._mode_paths(mode, input, clear=clear)
        new_input = inputs.Inputs()
        new_input.tag = mode
        if 'im' in out_dpaths:
            new_input.imdir = out_dpaths['im']
        if 'gt' in out_dpaths:
            new_input.gtdir = out_dpaths['gt']
        if 'aux' in out_dpaths:
            new_input.auxdir = out_dpaths['aux']

        if not clear:
            try:
                new_input.prepare_image_paths()
            except AssertionError:  # hack
                return prep._mode_new_input(mode, input, clear=True)

            if len(new_input.paths) > 0:
                n_loaded = min(map(len, new_input.paths.values()))
                min_n_expected = len(input) * mult
                print(' * n_loaded = {!r}'.format(n_loaded))
                print(' * min_n_expected = {!r}'.format(min_n_expected))
                # Short curcuit augmentation if we found stuff
                if n_loaded >= min_n_expected:
                    print('short circuit {}'.format(mode))
                    return new_input, True

        if 'im' in out_dpaths:
            new_input.im_paths = []
        if 'gt' in out_dpaths:
            new_input.gt_paths = []
        if 'aux' in out_dpaths:
            new_input.aux_paths = {k: [] for k in input.aux_paths.keys()}
        return new_input, False

    def make_lowres(prep, fullres, clear=False):
        # Define the output path for this preprocessing mode
        mode = 'lowres_' + '_'.join(list(map(str, prep.input_shape)))

        lowres, flag = prep._mode_new_input(mode, fullres, clear=clear, mult=1)
        if flag:
            return lowres

        target_dsize = tuple(prep.input_shape[::-1])

        def _lowres_proc(tag, out_dpath, in_paths, out_names, interpolation):
            """
            pip install tifffile
            """
            _iter = zip(in_paths, out_names)
            prog = ub.ProgIter(_iter, length=len(in_paths), label='lowres ' + tag)
            for in_path, out_name in prog:
                out_path = join(out_dpath, out_name)
                if not exists(out_path):
                    in_data = imutil.imread(in_path)
                    out_data = cv2.resize(in_data, target_dsize,
                                          interpolation=interpolation)
                    imutil.imwrite(out_path, out_data)
                yield out_path

        out_names = fullres.dump_im_names

        if fullres.im_paths:
            out_dpath = lowres.dirs['im']
            in_paths = fullres.paths['im']
            interpolation = cv2.INTER_LANCZOS4
            lowres.paths['im'] = list(_lowres_proc('im', out_dpath, in_paths,
                                                   out_names, interpolation))

        if fullres.gt_paths:
            out_dpath = lowres.dirs['gt']
            in_paths = fullres.paths['gt']
            interpolation = cv2.INTER_NEAREST
            lowres.paths['gt'] = list(_lowres_proc('gt', out_dpath, in_paths,
                                                   out_names, interpolation))

        if fullres.aux_paths:
            for aux, in_paths in fullres.aux_paths.items():
                out_dpath = lowres.dirs[aux]
                in_paths = fullres.paths[aux]
                # Unknown data is represented as a magic number -32767, so we
                # shouldnt use fancy interpolation.
                interpolation = cv2.INTER_NEAREST
                lowres.paths[aux] = list(_lowres_proc(aux, out_dpath, in_paths,
                                                      out_names,
                                                      interpolation))
        return lowres

    def make_parts(prep, fullres, scale=1, clear=False):
        """
        Slices the fullres images into smaller parts that fit into the network
        but are at the original resolution (or higher).

        >>> from clab.tasks.urban_mapper_3d import *
        >>> task = UrbanMapper3D(root='~/remote/aretha/data/UrbanMapper3D', workdir='~/data/work/urban_mapper')
        >>> task.prepare_fullres_inputs()
        >>> fullres = task.fullres
        >>> datadir = ub.ensuredir((task.workdir, 'data'))
        >>> prep = Preprocessor(datadir)
        >>> scale = 1
        >>> clear = False
        >>> lowres = prep.make_parts(fullres, scale)
        """
        part_config = prep.part_config
        hashid = hashutil.hash_data(ub.repr2(part_config), hashlen=8)
        shapestr = '_'.join(list(map(str, prep.input_shape)))
        mode = 'part-scale{}-{}-{}'.format(scale, shapestr, hashid)

        parts, flag = prep._mode_new_input(mode, fullres, clear=clear)
        if flag:
            return parts

        input_shape = prep.input_shape
        overlap = part_config['overlap']
        keepbound = part_config['keepbound']

        records = list(fullres.iter_records())
        for record in ub.ProgIter(records, label='make ' + mode):
            dump_fname = basename(record['dump_fname'])

            im_shape = np.array(Image.open(record['im']).size[::-1])
            im_shape = tuple(np.floor(im_shape * scale).astype(np.int))

            # Consolodate all channels that belong to this record
            in_paths = record.get('aux').copy()
            for k in ['im', 'gt']:
                if k in record:
                    in_paths[k] = record[k]

            # Read the images for this record and resize if necessary
            in_images = {k: imutil.imread(v) for k, v in in_paths.items()}  # 9% of the time
            if scale != 1.0:
                for k in in_images.keys():
                    interp = cv2.INTER_LANCZOS4 if k == 'im' else cv2.INTER_NEAREST
                    in_images[k] = imutil.imscale(in_images[k], scale, interp)[0]

            sl_gen = imutil.image_slices(im_shape, input_shape, overlap, keepbound)
            for idx, rc_slice in enumerate(sl_gen):
                rsl, csl = rc_slice
                suffix = '_part{:0=4d}_{:0=3d}_{:0=3d}'.format(idx, rsl.start,
                                                               csl.start)
                fname = ub.augpath(dump_fname, suffix=suffix)

                for k, in_data in in_images.items():
                    out_data = in_data[rc_slice]
                    out_fpath = join(parts.dirs[k], fname)
                    imutil.imwrite(out_fpath, out_data)  # 84% of the time
                    parts.paths[k].append(out_fpath)

        return parts
