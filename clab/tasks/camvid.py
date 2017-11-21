# -*- coding: utf-8 -*-
import ubelt as ub
from clab import inputs
from clab.tasks._sseg import SemanticSegmentationTask
from collections import OrderedDict
from os.path import exists, join, expanduser
from clab.util import imutil


class CamVid(SemanticSegmentationTask):
    def __init__(task, repo=None, workdir=None):
        classnames = [
            'Sky', 'Building', 'Pole', 'Road', 'Pavement', 'Tree',
            'SignSymbol', 'Fence', 'Car', 'Pedestrian', 'Bicyclist',
            'Unlabelled']
        null_classname = 'Unlabelled'
        super(CamVid, task).__init__(classnames, null_classname)
        task.class_colors = OrderedDict([
            ('Sky',        [128, 128, 128]),
            ('Building',   [128, 0, 0]),
            ('Pole',       [192, 192, 128]),
            ('Road',       [128, 64, 128]),
            ('Pavement',   [60, 40, 222]),
            ('Tree',       [128, 128, 0]),
            ('SignSymbol', [192, 128, 128]),
            ('Fence',      [64, 64, 128]),
            ('Car',        [64, 0, 128]),
            ('Pedestrian', [64, 64, 0]),
            ('Bicyclist',  [0, 128, 192]),
            ('Unlabelled', [0, 0, 0]),
        ])

        task.input_shape = (360, 480)
        task.workdir = expanduser(workdir)
        task.repo = repo

    def _load_image_paths(task, subdir):
        assert task.repo is not None
        assert exists(task.repo), 'repo must exist'
        dpath = join(task.repo, subdir)
        assert exists(dpath), 'repo subdir must exist'
        return imutil.load_image_paths(dpath, ext='.png')

    def train_im_paths(task):
        return task._load_image_paths('CamVid/train')

    def train_gt_paths(task):
        return task._load_image_paths('CamVid/trainannot')

    def test_im_paths(task):
        return task._load_image_paths('CamVid/test')

    def test_gt_paths(task):
        return task._load_image_paths('CamVid/testannot')

    def all_im_paths(task):
        return task.train_im_paths() + task.test_im_paths()

    def all_gt_paths(task):
        return task.train_gt_paths() + task.test_gt_paths()

    def xval_splits(task):
        train = inputs.Inputs.from_paths(task.train_im_paths(), task.train_gt_paths(),
                                         tag='train')
        test = inputs.Inputs.from_paths(task.test_im_paths(), task.test_gt_paths(),
                                        tag='test')

        assert not bool(ub.find_duplicates(test.im_paths))
        assert not bool(ub.find_duplicates(train.im_paths))

        xval_split = (train, test)
        yield xval_split
