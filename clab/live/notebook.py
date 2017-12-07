# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import torch
import glob
import numpy as np  # NOQA
import ubelt as ub
import os  # NOQA
from os.path import join, expanduser, splitext, basename  # NOQA
from clab import xpu_device
from clab.util import imutil
from clab import models

from clab.torch.sseg_train import get_task, SSegInputsWrapper


def hack_camvid_eval():
    """
    hacked together script to get the testing data and run prediction for submission
    """
    task = get_task('camvid')
    # load_path = os.path.expanduser('~/remote/aretha/data/work/pycamvid/arch/unet/train/input_331-funxsazz/solver_331-funxsazz_unet_vgg_adam_6_3/torch_snapshots/_epoch_00000050.pt')
    # load_path = os.path.expanduser('~/remote/aretha/data/work/pycamvid/arch/segnet/train/input_331-funxsazz/solver_331-funxsazz_segnet_vgg_adam_6_3/torch_snapshots/_epoch_00000050.pt')
    load_path = expanduser('~/remote/aretha/data/work/pycamvid/arch/segnet/train/input_331-funxsazz/solver_331-funxsazz_segnet_vgg_adam_6_3/torch_snapshots/_epoch_00000200.pt')
    load_path = expanduser('~/remote/aretha/data/work/pycamvid/arch/segnet/train/input_331-funxsazz/solver_331-funxsazz_segnet_vgg_adam_6_3/torch_snapshots/_epoch_00000800.pt')
    (train, test), = task.xval_splits()
    inputs = test
    dump_task_inference(task, inputs, load_path)


def most_recent_snapshot(train_dpath):
    snapshots = sorted(glob.glob(train_dpath + '/*/_epoch_*.pt'))
    load_path = snapshots[-1]
    return load_path


def dump_task_inference(task, inputs, load_path):
    """

        task = get_task('urban_mapper_3d')
        (train, test), = task.xval_splits()
        inputs = test

        eval_dataset = SSegInputsWrapper(inputs, task, colorspace='RGB')
        eval_dataset.with_gt = False
        eval_dataset.inputs.make_dumpsafe_names()

        load_path = 'solver_4214-yxalqwdk_unet_vgg_nttxoagf_a=1,n_ch=5,n_cl=3_epoch_00000236.pt'

    """
    eval_dataset = SSegInputsWrapper(inputs, task, colorspace='RGB')
    eval_dataset.with_gt = False
    eval_dataset.inputs.make_dumpsafe_names()

    if True:
        # TODO: make model metadata know about this
        eval_dataset.center_inputs = eval_dataset._original_urban_mapper_normalizer()

    eval_dpath = ub.ensuredir((task.workdir, 'eval', 'input_' + eval_dataset.input_id))
    subdir = list(ub.take(os.path.splitext(load_path)[0].split('/'), [-3, -1]))
    # base output dump path on the training id string
    test_dump_dpath = ub.ensuredir((eval_dpath, '/'.join(subdir)))
    print('test_dump_dpath = {!r}'.format(test_dump_dpath))

    datasets = {'eval': eval_dataset}

    # TODO n_classes and n_channels should be saved as model metadata
    n_classes = datasets['eval'].n_classes
    n_channels = datasets['eval'].n_channels

    xpu = xpu_device.XPU.from_argv()
    print('Loading snapshot onto {}'.format(xpu))
    snapshot = torch.load(load_path, map_location=xpu.map_location())
    # Infer which model this belongs to
    if snapshot['model_class_name'] == 'UNet':
        model = models.UNet(in_channels=n_channels, n_classes=n_classes)
    elif snapshot['model_class_name'] == 'SegNet':
        model = models.SegNet(in_channels=n_channels, n_classes=n_classes)

    model = xpu.to_xpu(model)
    model.load_state_dict(snapshot['model_state_dict'])

    print('Preparing to predict {} on {}'.format(model.__class__.__name__, xpu))
    model.train(False)

    for ix in ub.ProgIter(range(len(eval_dataset)), label='dumping'):
        inputs_ = eval_dataset[ix][None, :]

        inputs_ = xpu.to_xpu(inputs_)
        inputs_ = torch.autograd.Variable(inputs_)

        output_tensor = model(inputs_)
        log_prob_tensor = torch.nn.functional.log_softmax(output_tensor, dim=1)[0]
        log_probs = log_prob_tensor.data.cpu().numpy()

        # Just reload rgb data without trying to go through the reverse
        # transform
        img = imutil.imread(eval_dataset.inputs.im_paths[ix])

        # ut.save_cPkl('crf_testdata.pkl', {
        #     'log_probs': log_probs,
        #     'img': img,
        # })

        from clab.torch import filters

        posterior = filters.crf_posterior(img, log_probs)
        # output = prob_tensor.data.cpu().numpy()[0]

        pred = log_probs.argmax(axis=0)
        pred_crf = posterior.argmax(axis=0)

        fname = eval_dataset.inputs.dump_im_names[ix]
        fname = os.path.splitext(fname)[0] + '.png'

        # pred = argmax.data.cpu().numpy()[0]
        blend_pred = task.colorize(pred, img)
        blend_pred_crf = task.colorize(pred_crf, img)
        # color_pred = task.colorize(pred)

        output_dict = {
            'log_probs': log_probs,
            'blend_pred': blend_pred,
            # 'color_pred': color_pred,
            'blend_pred_crf': blend_pred_crf,
            'pred_crf': pred_crf,
            'pred': pred,
        }

        if eval_dataset.with_gt:
            true = imutil.imread(eval_dataset.inputs.gt_paths[ix])
            blend_true = task.colorize(true, img, alpha=.5)
            # color_true = task.colorize(true, alpha=.5)
            output_dict['true'] = true
            output_dict['blend_true'] = blend_true
            # output_dict['color_true'] = color_true

        for key, img in output_dict.items():
            dpath = join(test_dump_dpath, key)
            ub.ensuredir(dpath)
            fpath = join(dpath, fname)
            if key == 'log_probs':
                np.savez(fpath.replace('.png', ''), img)
            else:
                imutil.imwrite(fpath, img)

    return test_dump_dpath
