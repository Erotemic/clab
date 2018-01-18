# -*- coding: utf-8 -*-
# flake8: noqa
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np  # NOQA
import ubelt as ub
import os  # NOQA
from os.path import join, expanduser, splitext, basename  # NOQA
from clab import xpu_device
from clab.util import imutil
from clab import models

from clab.sseg_train import get_task, SSegInputsWrapper


def copy_latest_snapshots():
    train_base = ub.truepath('~/remote/aretha/data/work/urban_mapper/arch/unet/train')

    import glob
    import shutil
    def update_latest(train_dpath):
        load_path = most_recent_snapshot(train_dpath)
        suffix = load_path.split('/')[-1]
        new_path = join(train_dpath, basename(train_dpath) + suffix)
        print('new_path = {!r}'.format(new_path))
        shutil.copy2(load_path, new_path)

    for train_dpath in glob.glob(train_base + '/input_*/solver_*'):
        if os.path.isdir(train_dpath):
            print('train_dpath = {!r}'.format(train_dpath))
            update_latest(train_dpath)
