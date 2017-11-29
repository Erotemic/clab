# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
from os.path import basename, abspath, join, exists
import json
import numpy as np
import six
import pandas as pd
import ubelt as ub
from .util import fnameutil
from .util import imutil
from .util import hashutil

from clab import getLogger
logger = getLogger(__name__)
print = logger.info


def make_input_file(im_paths, gt_paths=None, ext='.png', dpath=None):
    # TODO: remove or refactor (holdover from caffe)
    input_fpath = join(dpath, 'inputs.txt')
    if gt_paths is not None:
        assert fnameutil.check_aligned(im_paths, gt_paths), (
            'image names should correspond')
        text = '\n'.join([
            '{} {}'.format(xpath, ypath)
            for xpath, ypath in zip(im_paths, gt_paths)
        ])
    else:
        text = '\n'.join(im_paths)
    ub.writeto(input_fpath, text)
    return input_fpath


class DictLike(object):
    """
    An inherited class must specify the ``__getitem__``, ``__setitem__``, and
      ``keys`` methods.
    """

    def keys(self):
        raise NotImplementedError('abstract keys function')

    def __delitem__(self, key):
        raise NotImplementedError('abstract __delitem__ function')

    def __getitem__(self, key):
        raise NotImplementedError('abstract __getitem__ function')

    def __setitem__(self, key, value):
        raise NotImplementedError('abstract __setitem__ function')

    def __repr__(self):
        return repr(self.to_dict())

    def __str__(self):
        return str(self.to_dict())

    def __len__(self):
        return len(list(self.keys()))

    def __contains__(self, key):
        return key in self.keys()

    def items(self):
        if six.PY2:
            return list(self.iteritems())
        else:
            return self.iteritems()

    def values(self):
        if six.PY2:
            return [self[key] for key in self.keys()]
        else:
            return (self[key] for key in self.keys())

    def copy(self):
        return dict(self.items())

    def to_dict(self):
        return dict(self.items())

    def iteritems(self):
        for key, val in zip(self.iterkeys(), self.itervalues()):
            yield key, val

    def itervalues(self):
        return (self[key] for key in self.keys())

    def iterkeys(self):
        return (key for key in self.keys())

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default


class InputAttrDictProxy(DictLike):
    """
    Allows you to use key / values for im, gt, and all aux in a flat-like dict
    """

    def __init__(self, parent, suffix):
        self.parent = parent
        self.suffix = suffix

    @property
    def _aux(self):
        return getattr(self.parent, 'aux' + self.suffix)

    def __getitem__(self, key):
        if key in ['gt', 'im']:
            attr = getattr(self.parent, key + self.suffix)
        else:
            attr = self._aux[key]
        return attr

    def __setitem__(self, key, value):
        if key in ['gt', 'im']:
            setattr(self.parent, key + self.suffix, value)
        else:
            self._aux[key] = value

    def __delitem__(self, key):
        if key in ['gt', 'im']:
            setattr(self.parent, key + self.suffix, None)
        else:
            del self._aux[key]

    def keys(self):
        if self['im'] is not None:
            yield 'im'
        if self['gt'] is not None:
            yield 'gt'
        aux = self._aux
        if aux:
            for k in aux.keys():
                yield k


class Inputs(ub.NiceRepr):
    """
    Class to manage image / groundtruth inputs, abstracting away the notion of
    test/train.

    >>> import clab
    >>> harn = clab.harness.Harness()
    >>> train_data, test_data = next(harn.task.xval_splits())
    >>> harn.test.im_paths = test_data[0]
    >>> harn.test.gt_paths = test_data[1]
    >>> harn.train.im_paths = train_data[0]
    >>> harn.train.gt_paths = train_data[1]
    >>> self = harn.train
    >>> self.prepare_input()
    >>> harn.prepare_solver()
    >>> harn.prepare_test_model()
    """

    def __init__(self, tag=None):
        self.tag = tag

        self.imdir = None
        self.gtdir = None
        self.auxdir = None

        self.im_paths = None
        self.gt_paths = None
        self.aux_paths = None

        self.metadata = None

        self.n_input = None
        self.input_id = None

        self.base_dpath = None

        self.input_dpath = None
        self.input_fpath = None

        self.abbrev = 8

        # Mainly for training
        self.gtstats = None

        # Mainly for testing
        # (rename to dump_im_names)
        self.dump_im_names = None

    @classmethod
    def from_paths(cls, im=None, gt=None, tag=None, **aux):
        """
        Convinience constructor
        """
        self = cls(tag)
        self.paths['im'] = im
        self.paths['gt'] = gt
        if aux and self.aux_paths is None:
            self.aux_paths = {}
        for k, v in aux.items():
            self.paths[k] = v
        return self

    @classmethod
    def from_dirs(cls, im=None, gt=None, tag=None, **aux):
        """
        Convinience constructor
        """
        self = cls(tag)
        self.dirs['im'] = im
        self.dirs['gt'] = gt
        for k, v in aux.items():
            self.dirs[k] = v
        return self

    def union(self, other):
        keys = sorted(self.paths.keys())
        assert keys == sorted(other.paths.keys())
        new = Inputs.from_paths(**{
            k: self.paths[k] + other.paths[k]
            for k in keys})
        return new

    @staticmethod
    def union_all(*args):
        """
        Example:
            >>> from clab.inputs import *
            >>> a = Inputs.from_paths(im=['%d.png' % i for i in range(00, 10)])
            >>> b = Inputs.from_paths(im=['%d.png' % i for i in range(10, 20)])
            >>> c = Inputs.from_paths(im=['%d.png' % i for i in range(20, 30)])
            >>> abc = Inputs.union_all(a, b, c)
            >>> assert not ub.find_duplicates(abc.paths['im'])
        """
        result = args[0]
        for b in args[1:]:
            result = result.union(b)
        return result

    def take(self, indicies):
        keys = sorted(self.paths.keys())
        new = Inputs.from_paths(**{
            key: list(ub.take(self.paths[key], indicies))
            for key in keys
        })
        return new

    def __add__(self, other):
        return self.union(other)

    @property
    def paths(self):
        return InputAttrDictProxy(self, suffix='_paths')

    @property
    def dirs(self):
        return InputAttrDictProxy(self, suffix='dir')

    def __len__(self):
        if not len(self.paths):
            n_loaded = 0
        else:
            n_loaded = min(list(map(len, self.paths.values())))
        return n_loaded

    def __nice__(self):
        if self.im_paths:
            n = len(self.im_paths)
        elif self.gt_paths:
            n = len(self.gt_paths)
        else:
            n = None
        return '{} {}'.format(self.tag, n)

    def _get_record(self, i):
        record = {}
        if self.dump_im_names:
            record['dump_fname'] = self.dump_im_names[i]
        if self.im_paths:
            record['im'] = self.im_paths[i]
        if self.gt_paths:
            record['gt'] = self.gt_paths[i]
        if self.metadata is not None:
            record['meta'] = self.metadata.iloc[i]
        if self.aux_paths is not None:
            record['aux'] = {k: v[i] for k, v in self.aux_paths.items()}
        return record

    def __getitem__(self, index):
        """
        Example:
            >>> from clab.inputs import *
            >>> self = Inputs.from_paths(im=['%d.png' % i for i in range(00, 10)])
            >>> assert isinstance(self[0], dict)
            >>> assert isinstance(self[0:1], Inputs)
            >>> assert len(self[0:1]) == 1
            >>> assert len(self[0:10]) == 10
            >>> assert len(self[0:20]) == 10
            >>> assert len(self[0:20:2]) == 5
            >>> assert len(self[:]) == 10
        """
        if isinstance(index, slice):
            indices = list(range(*index.indices(len(self))))
            return self.take(indices)
        else:
            return self._get_record(index)

    def iter_records(self):
        if self.im_paths:
            n_records = len(self.im_paths)
        if self.gt_paths:
            n_records = len(self.gt_paths)

        for i in range(n_records):
            yield self._get_record(i)

    def make_dumpsafe_names(self):
        """
        makes paths suitable for dumping into a single directory
        """
        if self.dump_im_names is None:
            self.dump_im_names = fnameutil.dumpsafe(self.im_paths)

    def set_ims(self, ims):
        """ Can be a directory or a list of images """
        if isinstance(ims, list):
            self.im_paths = ims
        else:
            self.imdir = abspath(ims)

    def set_gts(self, gts):
        """ Can be a directory or a list of images """
        if isinstance(gts, list):
            self.gt_paths = gts
        else:
            self.gtdir = abspath(gts)

    def prepare_truth(self):
        if self.gt_paths is None:
            self.gt_paths = imutil.load_image_paths(self.gtdir, ext='.png')

    def prepare_image_paths(self):
        print('Preparing {} image paths'.format(self.tag))

        any_loaded = False

        if self.im_paths is None and self.imdir:
            self.im_paths = imutil.load_image_paths(self.imdir)
            any_loaded = len(self.im_paths) > 0

        if self.gt_paths is None and self.gtdir:
            self.gt_paths = imutil.load_image_paths(self.gtdir)
            any_loaded = len(self.gt_paths) > 0

        if self.aux_paths is None and self.auxdir:
            self.aux_paths = {k: imutil.load_image_paths(v)
                              for k, v in self.auxdir.items()}
            any_loaded = any(self.aux_paths.values())

        if any_loaded:
            # Ensure alignment between loaded data
            if self.im_paths and self.gt_paths:
                self.gt_paths = fnameutil.align_paths(self.im_paths,
                                                      self.gt_paths)
            if self.aux_paths:
                base = self.im_paths or self.gt_paths
                for k in self.aux_paths.keys():
                    self.aux_paths[k] = fnameutil.align_paths(
                        base, self.aux_paths[k])

            print('Prepared {} image paths'.format(len(self)))

    def prepare_images(self, ext='.png', force=False):
        """
        If not already done, loads paths to images into memory and constructs a
        unique id for that set of im/gt images.

        It the paths are already set, then only the input-id is constructed.
        """
        if self.n_input is not None and not force:
            return

        self.prepare_image_paths()

        if self.aux_paths:
            # new way
            depends = sorted(self.paths.items())
        else:
            depends = []
            depends.append(self.im_paths)
            depends.append(self.gt_paths)
            if self.gt_paths:
                # HACK: We will assume image data depends only on the filename
                # HACK: be respectful of gt label changes (ignore aug)
                label_hashid = hashutil.hash_data(
                    # stride=32 is fast but might break
                    # stride=1 is the safest
                    [hashutil.get_file_hash(p, stride=32) for p in self.gt_paths
                     if 'aug' not in basename(p) and 'part' not in basename(p)]
                )
                depends.append(label_hashid)
        n_im = None if self.im_paths is None else len(self.im_paths)
        n_gt = None if self.gt_paths is None else len(self.gt_paths)
        self.n_input = n_im or n_gt

        hashid = hashutil.hash_data(depends)[:self.abbrev]
        self.input_id = '{}-{}'.format(self.n_input, hashid)

        print(' * n_images = {}'.format(n_im))
        print(' * n_groundtruth = {}'.format(n_gt))
        print(' * input_id = {}'.format(self.input_id))

    def prepare_input(self):
        """
        Prepare the text file containing inputs that can be passed to caffe.
        """
        self.prepare_images()
        if self.input_fpath is None:
            assert self.base_dpath is not None
            assert self.input_id is not None
            self.input_dpath = ub.ensuredir((self.base_dpath,
                                             self.input_id))
            # TODO: remove or refactor (holdover from caffe)
            self.input_fpath = make_input_file(
                self.im_paths, self.gt_paths, dpath=self.input_dpath)
            print('{} input_fpath = {!r}'.format(self.tag,
                                                 ub.compressuser(self.input_fpath)))

    def prepare_center_stats(self, task, nan_value=-32767.0, colorspace='RGB'):
        """
        Ignore:
            >>> from clab.torch.sseg_train import *
            >>> #task = get_task('urban_mapper_3d')
            >>> task = get_task('camvid')
            >>> self, test = next(task.xval_splits())
            >>> self.base_dpath = ub.ensuredir((task.workdir, 'inputs'))
            >>> nan_value = -32767.0
            >>> colorspace = 'RGB'
            >>> colorspace = 'LAB'
            >>> self.prepare_center_stats(task, colorspace=colorspace)
        """
        # TODO: handle different color spaces
        # colorspace = 'RGB'
        import pickle
        import copy
        from .util import jsonutil
        from .torch.transforms import NaNInputer
        from .torch import im_loaders

        self.prepare_input()
        fpath = join(self.input_dpath, 'center_stats{}.pkl'.format(colorspace))
        simple_fpath = join(self.input_dpath, 'center_stats_simple{}.json'.format(colorspace))
        print('Checking intensity stats')

        if not exists(fpath):
            print('Need to compute intensity stats')
            im_paths = self.im_paths

            class IntensityStats(object):
                def __init__(p):
                    p.run = imutil.RunningStats()
                    p.scalar_internals = imutil.InternalRunningStats()
                    p.channel_internals = imutil.InternalRunningStats(axis=(0, 1))

                def update(p, im):
                    p.run.update(im)
                    p.scalar_internals.update(im)
                    p.channel_internals.update(im)

                def info(p):
                    image_simple_info = {
                        'desc': 'statistics about the per channel/entire intensity averaged across the dataset',
                        'channel': p.run.simple(axis=(0, 1)),
                        'image': p.run.simple(),
                    }

                    image_detail_info = {
                        'desc': 'statistics about the per pixel intensity averaged across the dataset',
                        'pixel': p.run.detail(),
                    }

                    image_internal_info = {
                        'desc': 'statistics about the average internal mean/med/mad/std intensity within an image',
                        'image': p.scalar_internals.info(),
                        'channel': p.channel_internals.info(),
                    }

                    return {
                        'simple': image_simple_info,
                        'detail': image_detail_info,
                        'internal': image_internal_info,
                    }

            run_im = IntensityStats()
            for path in ub.ProgIter(im_paths, label='computing mean image', verbose=1):
                im = im_loaders.np_loader(path, colorspace=colorspace)
                run_im.update(im)

            if self.aux_paths:
                # nan_value = -32767.0
                nan_inputer = NaNInputer(fill='median', nan_value=nan_value)
                run_aux = IntensityStats()
                prog = ub.ProgIter(label='aux stats', verbose=1)
                for aux_paths in prog(list(zip(*self.aux_paths.values()))):
                    aux = np.dstack([im_loaders.np_loader(path) for path in aux_paths])
                    aux = nan_inputer(aux)
                    run_aux.update(aux)

                aux_channel_names = list(self.aux_paths.keys())
                aux_info = run_aux.info()
                aux_info['channel_names'] = aux_channel_names
            else:
                aux_info = None

            im_info = run_im.info()
            im_info['colorspace'] = colorspace

            detail_info = {
                'aux': aux_info,
                'image': im_info,
            }
            print('writing to fpath = {!r}'.format(fpath))
            with open(fpath, 'wb') as file:
                pickle.dump(detail_info, file)

            simple_info = copy.deepcopy(detail_info)

            # Write a simpler version of the file
            if simple_info['aux']:
                simple_info['aux'].pop('detail')
            simple_info['image'].pop('detail')

            # pretty writing of json stats
            json_text = json.dumps(simple_info, cls=jsonutil.NumpyAwareJSONEncoder, indent=4)
            print('writing to simple_fpath = {!r}'.format(simple_fpath))
            ub.writeto(simple_fpath, json_text)

        print('reading fpath = {!r}'.format(fpath))
        with open(fpath, 'rb') as file:
            info = pickle.load(file)
        return info

    def prepare_gtstats(self, task, force=False):
        """
        Caches stats like class frequency to disk, before we start training
        """
        self.prepare_input()

        gtstats_fpath = join(self.input_dpath, 'gtstats_v1.json')
        if force or exists(gtstats_fpath):
            gtstats = pd.read_json(gtstats_fpath)
        else:
            self.prepare_images()
            gt_paths = self.paths['gt']
            gtstats = self._compute_gt_info(gt_paths, task)
            # pretty writing of json stats
            json_text = (json.dumps(json.loads(gtstats.to_json()), indent=4))
            ub.writeto(gtstats_fpath, json_text)
        self.gtstats = gtstats
        return self.gtstats

    def _compute_gt_info(self, gt_paths, task):
        # self.imdir  = expanduser('~/store/segnet-exact/SegNet-Tutorial/CamVid/train')
        # self.gtdir  = expanduser('~/store/segnet-exact/SegNet-Tutorial/CamVid/trainannot')
        # def insert_empty_rows(df, new_index, fill_value=np.nan):
        #     # TODO: Is there a better way to do this? Maybe inplace?
        #     # shape = (len(new_index), len(df.columns))
        #     # data = np.full(shape, fill_value=fill_value)
        #     fill = pd.DataFrame(fill_value, index=new_index, columns=df.columns)
        #     return pd.concat([df, fill])
        # new_index = pxlfreq.index.difference(gtstats.index).astype(np.int64)
        # if len(new_index) > 0:
        #     # Expand the dataframe to capture all classes
        #     gtstats = insert_empty_rows(gtstats, new_index, fill_value=0)
        index = pd.Index(task.labels, name='labels')
        gtstats = pd.DataFrame(0, index=index, columns=['pxlfreq', 'imfreq'],
                               dtype=np.int)

        for path in ub.ProgIter(gt_paths, label='computing class weights', verbose=1):
            y_true = imutil.imread(path).ravel()
            pxlfreq = pd.value_counts(y_true)
            gtstats.pxlfreq.loc[pxlfreq.index] += pxlfreq
            gtstats.imfreq.loc[pxlfreq.index] += 1
        gtstats.pxlfreq = pd.to_numeric(gtstats.pxlfreq)
        gtstats.imfreq = pd.to_numeric(gtstats.imfreq)

        gtstats['classname'] = list(ub.take(task.classnames, gtstats.index))
        gtstats['mf_weight'] = gtstats.pxlfreq.median() / gtstats.pxlfreq
        gtstats.loc[~np.isfinite(gtstats.mf_weight), 'mf_weight'] = 1

        # Clip weights, so nothing gets crazy high weights, low weights are ok
        gtstats['loss_weight'] = np.clip(gtstats.mf_weight, a_min=None, a_max=4)
        gtstats = gtstats.sort_index()
        gtstats.index.name = 'label'
        gtstats = gtstats.reset_index().set_index('classname', drop=False)
        return gtstats

    def align(self, other):
        return fnameutil.align_paths(self.im_paths, other)
