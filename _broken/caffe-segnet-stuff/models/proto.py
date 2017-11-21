# -*- coding: utf-8 -*-
from pysseg.models import segnet_basic
from pysseg.models import segnet_proper
from pysseg.util import fnameutil
import ubelt as ub
from os.path import join
import re

from pysseg import getLogger
logger = getLogger(__name__)
print = logger.info

model_modules = {
    'segnet_basic': segnet_basic,
    'segnet_proper': segnet_proper,
}


def default_hyperparams(arch):
    return model_modules[arch].DEFAULT_HYPERPARAMS


def make_model_file(input_fpath, arch='segnet_basic', mode='predict',
                    dpath=None, modelkw={}, params=None):
    # assert input_fpath, 'must specify'
    model_fname = '{}_{}_model.prototext'.format(arch, mode)
    model_fpath = join(dpath, model_fname)

    text = make_prototext(input_fpath, arch=arch, mode=mode, params=params,
                          **modelkw)
    ub.writeto(model_fpath, text)
    print('made model_fpath = {!r}'.format(ub.compressuser(model_fpath)))
    return model_fpath


def make_solver_file(input_fpath, arch='segnet_basic', dpath=None, modelkw={},
                     params=None, gpu_num=0):
    assert input_fpath, 'must specify'

    model_fpath = make_model_file(input_fpath, arch=arch, mode='fit',
                                  dpath=dpath, modelkw=modelkw)
    solver_fname = '{}_solver.prototext'.format(arch)
    solver_fpath = join(dpath, solver_fname)
    snapshot_dpath = ub.ensuredir((dpath, 'snapshots'))
    snapshot_prefix = snapshot_dpath + '/'
    text = make_solver(model_fpath, snapshot_prefix=snapshot_prefix,
                       params=params, gpu_num=gpu_num)
    ub.writeto(solver_fpath, text)
    print('made solver_fpath = {!r}'.format(ub.compressuser(solver_fpath)))
    return solver_fpath


def make_input_file(im_paths, gt_paths=None, ext='.png', dpath=None):
    """
    Example:
        >>> from pysseg import util
        >>> imdir = '~/sseg/virat/imall'
        >>> gtdir = '~/sseg/virat/gtall'
        >>> im_paths = util.load_image_paths(imdir)
        >>> ext = '.png'
    """
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


def make_solver(train_model_prototext_fpath, snapshot_prefix=None,
                params=None, gpu_num=0):
    """
    Makes the solver file that holds where the model architecture is, and the
    hyperparameters that will be used to train it.

    References:
        # See this for info on params
        https://github.com/BVLC/caffe/blob/master/src/caffe/proto/caffe.proto#L102
    """
    assert train_model_prototext_fpath, 'must specify'
    solver_mode = 'CPU' if gpu_num is None else 'GPU'
    lines = [
        'net: "{}"'.format(train_model_prototext_fpath),
        'snapshot_prefix: "{}"'.format(snapshot_prefix),
        'solver_mode: {}'.format(solver_mode),
    ]
    lines.extend(list(params.protolines()))
    text = '\n'.join(lines)
    return text


def make_prototext(image_list_fpath, arch, mode='fit', batch_size=1,
                   n_classes=None, class_weights=None, ignore_label=None,
                   shuffle=None, params=None):

    assert mode in {'fit', 'predict'}
    mod = model_modules[arch]
    if shuffle is None:
        shuffle = (mode == 'fit')

    if n_classes is None:
        n_classes = len(class_weights)
    elif ignore_label is not None:
        # this is really weird
        # with 12 classes we need to make the number of outputs be 11 because
        # we are ignoring the last label. However, when class_weights are
        # passed in we only send it the used weights, so that's already the
        # right number. Not sure what happend when ignore_label=0 and not 11
        n_classes -= 1

    fmtdict = {
        'shuffle': str(shuffle).lower(),
        'batch_size': batch_size,
        'image_list_fpath': image_list_fpath,
        'n_classes': n_classes,
        'arch_name': arch,
    }

    if image_list_fpath is None:
        # Input layer when we use blobs
        # maybe use this def instead?
        # layer {
        #   name: "input"
        #   type: "Input"
        #   top: "data"
        #   input_param {
        #     shape {
        #       dim: 1
        #       dim: 3
        #       dim: 360
        #       dim: 480
        #     }
        #   }
        # }
        input_layer_fmt = ub.codeblock(
            '''
            input: "data"
            input_dim: {batch_size}
            input_dim: 3
            input_dim: 360
            input_dim: 480
            ''')
    else:
        # Layer when input is specified in a txt
        input_layer_fmt = ub.codeblock(
            '''
            name: "{arch_name}"
            layer {{
              name: "data"
              type: "DenseImageData"
              top: "data"
              top: "label"
              dense_image_data_param {{
                source: "{image_list_fpath}"
                batch_size: {batch_size}
                shuffle: {shuffle}
              }}
            }}
            '''
        )

    input_layer = input_layer_fmt.format(**fmtdict)

    if hasattr(mod, 'make_core_layers'):
        if params is not None:
            freeze_before = params['freeze_before']
            finetune_decay = params['finetune_decay']
        else:
            freeze_before = 0
            finetune_decay = 1
        core = mod.make_core_layers(n_classes, freeze_before, finetune_decay)
    else:
        core = mod.CORE_LAYERS.format(**fmtdict)

    if mode == 'fit':
        # remove batch-norm inference when fitting
        core = re.sub('^\s*bn_mode:\s*INFERENCE$', '', core, flags=re.M)
        class_weights_line = ['class_weighting: {}'.format(w) for w in class_weights]
        class_weights_line += ['ignore_label: {}'.format(ignore_label)]
        class_weights_text = ub.indent('\n'.join(class_weights_line), ' ' * 4).lstrip()
        fmtdict['class_weights_text'] = class_weights_text
        footer_fmt = mod.FIT_FOOTER
    else:
        footer_fmt = mod.PREDICT_FOOTER

    footer = footer_fmt.format(**fmtdict)

    text = '\n'.join([input_layer, core, footer])
    return text
