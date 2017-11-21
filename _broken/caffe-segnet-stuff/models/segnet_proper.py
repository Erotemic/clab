# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import ubelt as ub


# TODO: encapsulate model prototext in classes.  potentially abstract each
# layer into a node in a digraph with format text.
# Caffe#2086 has this python net-spec, but not this old segnet code

# # See this for info on params
# https://github.com/BVLC/caffe/blob/master/src/caffe/proto/caffe.proto#L102
DEFAULT_HYPERPARAMS = {
    'test_initialization': False,
    'test_iter': 1,
    'test_interval': 10000000,
    'base_lr': 0.001,
    'lr_policy': "step",
    'gamma': 1.0,
    'stepsize': 10000000,
    'display': 20,
    'momentum': 0.9,
    'max_iter': 40000,
    'weight_decay': 0.0005,
    'snapshot': 1000,
}


class WrappedProtoLayer(object):
    def __init__(self, fmt, params):
        params.pop('fmt', None)
        self.fmt = fmt
        self.params = params

    def format(self):
        try:
            # Hack: force lr_mult2 to be twice lr_mult
            if 'lr_mult2' in self.params:
                self.params['lr_mult2'] = self.params['lr_mult'] * 2

            return self.fmt.format(**self.params)
        except KeyError:
            print('Failed to format: self.fmt = """')
            print(self.fmt)
            print('"""')
            print('With self.params = {}'.format(ub.repr2(self.params)))
            raise


def convolution(bot, top, name, nfilt, ksize=3, lr_mult=1):
    """
    References:
        http://caffe.berkeleyvision.org/tutorial/layers/convolution.html
    """
    lr_mult2 = lr_mult * 2
    type = 'Convolution'
    fmt = ub.codeblock(
        '''
        layer {{
          bottom: "{bot}"
          top: "{name}"
          name: "{name}"
          type: "{type}"
          # First param is learning/decay rate for the filter weights
          param {{
            lr_mult: {lr_mult}
            decay_mult: 1
          }}
          # Second param is learning/decay rate for the biases
          param {{
            lr_mult: {lr_mult2}
            decay_mult: 0
          }}
          convolution_param {{
            weight_filler {{
              type: "msra"
            }}
            bias_filler {{
              type: "constant"
            }}
            num_output: {nfilt}
            pad: 1
            kernel_size: {ksize}
          }}
        }}
        ''')
    return WrappedProtoLayer(fmt, locals())


def batch_norm(bot, top, name, lr_mult=1):
    """
    this is the SegNet version of BatchNorm, normal caffe has a different one

    References:
        https://github.com/alexgkendall/caffe-segnet/blob/288e1cb1ec3759e9e1a8fb2d7e3e314f7b35fe1e/src/caffe/proto/caffe.proto#L426
    """
    type = 'BN'
    fmt = ub.codeblock(
        '''
        layer {{
          bottom: "{bot}"
          top: "{top}"
          name: "{name}"
          type: "{type}"
          # First param is learning/decay rate for the scale (weight)
          param {{
            lr_mult: {lr_mult}
            decay_mult: 1
          }}
          # First param is learning/decay rate for the shift (bias)
          param {{
            lr_mult: {lr_mult}
            decay_mult: 0
          }}
          bn_param {{
            bn_mode: INFERENCE
            scale_filler {{
              type: "constant"
              value: 1
            }}
            shift_filler {{
              type: "constant"
              value: 0.001
            }}
         }}
        }}
        ''')
    return WrappedProtoLayer(fmt, locals())


def relu(bot, top, name):
    type = 'ReLU'
    fmt = ub.codeblock(
        '''
        layer {{
          bottom: "{bot}"
          top: "{top}"
          name: "{name}"
          type: "{type}"
        }}
        ''')
    return WrappedProtoLayer(fmt, locals())


def pool(bot, name, stride=2, ksize=2):
    type = 'Pooling'
    fmt = ub.codeblock(
        '''
        layer {{
          bottom: "{bot}"
          top: "{name}"
          top: "{name}_mask"
          name: "{name}"
          type: "{type}"
          pooling_param {{
            pool: MAX
            kernel_size: {ksize}
            stride: {stride}
          }}
        }}
        ''')
    return WrappedProtoLayer(fmt, locals())


def upsample(bot, suffix, scale, width=None, height=None):
    type = 'Upsample'
    if width is not None:
        fmt = ub.codeblock(
            '''
            layer {{
              name: "upsample{suffix}"
              type: "{type}"
              bottom: "{bot}"
              top: "pool{suffix}_D"
              bottom: "pool{suffix}_mask"
              upsample_param {{
                scale: {scale}
                upsample_w: {width}
                upsample_h: {height}
              }}
            }}
            ''')
    else:
        fmt = ub.codeblock(
            '''
            layer {{
              name: "upsample{suffix}"
              type: "{type}"
              bottom: "{bot}"
              top: "pool{suffix}_D"
              bottom: "pool{suffix}_mask"
              upsample_param {{
                scale: {scale}
              }}
            }}
            ''')
    return WrappedProtoLayer(fmt, locals())


def conv_bn_relu(bot, suffix, nfilt, ksize=3, lr_mult=1):
    conv_name = 'conv' + suffix
    bn_name = conv_name + '_bn'
    relu_name = 'relu' + suffix
    return [
        convolution(bot, conv_name, conv_name, nfilt=nfilt, ksize=ksize,
                    lr_mult=lr_mult),
        batch_norm(conv_name, conv_name, bn_name, lr_mult=lr_mult),
        relu(conv_name, conv_name, relu_name),
    ]


def make_core_layers(n_classes, freeze_before=0, finetune_decay=1):
    layers = []
    # == # conv1
    layers += conv_bn_relu(bot='data',    suffix='1_1', nfilt=64)
    layers += conv_bn_relu(bot='conv1_1', suffix='1_2', nfilt=64)
    layers += [pool('conv1_2', 'pool1')]
    # ==== # conv2
    layers += conv_bn_relu(bot='pool1',   suffix='2_1', nfilt=128)
    layers += conv_bn_relu(bot='conv2_1', suffix='2_2', nfilt=128)
    layers += [pool('conv2_2', 'pool2')]
    # ====== # conv3
    layers += conv_bn_relu(bot='pool2',   suffix='3_1', nfilt=256)
    layers += conv_bn_relu(bot='conv3_1', suffix='3_2', nfilt=256)
    layers += conv_bn_relu(bot='conv3_2', suffix='3_3', nfilt=256)
    layers += [pool('conv3_3', 'pool3')]
    # ======== # conv4
    layers += conv_bn_relu(bot='pool3',     suffix='4_1', nfilt=512)
    layers += conv_bn_relu(bot='conv4_1',   suffix='4_2', nfilt=512)
    layers += conv_bn_relu(bot='conv4_2',   suffix='4_3', nfilt=512)
    layers += [pool('conv4_3', 'pool4')]
    # ========== # conv5
    layers += conv_bn_relu(bot='pool4',     suffix='5_1', nfilt=512)
    layers += conv_bn_relu(bot='conv5_1',   suffix='5_2', nfilt=512)
    layers += conv_bn_relu(bot='conv5_2',   suffix='5_3', nfilt=512)
    layers += [pool('conv5_3', 'pool5')]
    # ========== # up+conv5
    layers += [upsample('pool5', '5', 2, width=30, height=23)]
    layers += conv_bn_relu(bot='pool5_D',     suffix='5_3_D', nfilt=512)
    layers += conv_bn_relu(bot='conv5_3_D',   suffix='5_2_D', nfilt=512)
    layers += conv_bn_relu(bot='conv5_2_D',   suffix='5_1_D', nfilt=512)
    # ======== # up+conv4
    layers += [upsample('conv5_1_D', '4', scale=2, width=60, height=45)]
    layers += conv_bn_relu(bot='pool4_D',     suffix='4_3_D', nfilt=512)
    layers += conv_bn_relu(bot='conv4_3_D',   suffix='4_2_D', nfilt=512)
    layers += conv_bn_relu(bot='conv4_2_D',   suffix='4_1_D', nfilt=256)
    # ====== # up+conv3
    layers += [upsample('conv4_1_D', '3', scale=2)]
    layers += conv_bn_relu(bot='pool3_D',     suffix='3_3_D', nfilt=256)
    layers += conv_bn_relu(bot='conv3_3_D',   suffix='3_2_D', nfilt=256)
    layers += conv_bn_relu(bot='conv3_2_D',   suffix='3_1_D', nfilt=128)
    # ==== # up+conv2
    layers += [upsample('conv3_1_D', '2', scale=2)]
    layers += conv_bn_relu(bot='pool2_D',     suffix='2_2_D', nfilt=128)
    layers += conv_bn_relu(bot='conv2_2_D',   suffix='2_1_D', nfilt=64)
    # == # up+conv1
    layers += [upsample('conv2_1_D', '1', scale=2)]
    layers += conv_bn_relu(bot='pool1_D',   suffix='1_2_D', nfilt=64)
    # output pixel labels
    layers += [convolution('conv1_2_D', 'conv1_1_D_output' + str(n_classes),
                           'conv1_1_D_output' + str(n_classes), nfilt=n_classes)]

    # 26 total convolutional layers
    # total_conv_layers = sum([layer.params['type'] == 'Convolution'
    #                          for layer in layers])
    learnable_layers = [layer for layer in layers if 'lr_mult' in layer.params]
    total_lr_layers = len(learnable_layers)

    import numpy as np
    freeze_before_ = (np.clip(freeze_before, -total_lr_layers, total_lr_layers) %
                      total_lr_layers)

    # Freeze learning in all layers before `freeze_before_`
    for count, layer in enumerate(learnable_layers, start=1):
        if count < freeze_before_:
            # Force layers before this point to have a learning rate of 0
            layer.params['lr_mult'] = 0
        if finetune_decay != 1:
            # Decay so earlier layers recieve exponentially less weight
            layer.params['lr_mult'] *= (finetune_decay ** (total_lr_layers - count))

    # Freeze the learning rate of any layer before the freeze points
    core_layer_auto = '\n'.join([p.format() for p in layers])

    if False:
        import utool as ut
        from pysseg.models import segnet_proper_orig
        core_layer_old = segnet_proper_orig.CORE_LAYERS.format(n_classes=n_classes)
        print(ut.color_diff_text(ut.get_textdiff(core_layer_old, core_layer_auto, num_context_lines=10)))
    return core_layer_auto


FIT_FOOTER = ub.codeblock(
    '''
    layer {{
      name: "loss"
      type: "SoftmaxWithLoss"
      bottom: "conv1_1_D_output{n_classes}"
      bottom: "label"
      top: "loss"
      softmax_param {{engine: CAFFE}}
      loss_param: {{
        weight_by_label_freqs: true
        {class_weights_text}
      }}
    }}
    layer {{
      name: "accuracy"
      type: "Accuracy"
      bottom: "conv1_1_D_output{n_classes}"
      bottom: "label"
      top: "accuracy"
      top: "per_class_accuracy"
    }}
    ''')


PREDICT_FOOTER = ub.codeblock(
    '''
    layer {{
      name: "prob"
      type: "Softmax"
      bottom: "conv1_1_D_output{n_classes}"
      top: "prob"
      softmax_param {{engine: CAFFE}}
    }}
    ''')
