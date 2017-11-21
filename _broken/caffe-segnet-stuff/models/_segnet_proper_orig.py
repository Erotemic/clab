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


def convolution(bot, top, name, nfilt, ksize=3):
    """
    References:
        http://caffe.berkeleyvision.org/tutorial/layers/convolution.html
    """
    return ub.codeblock(
        '''
        layer {{
          bottom: "{bot}"
          top: "{name}"
          name: "{name}"
          type: "Convolution"
          # First param is learning/decay rate for the filter weights
          param {{
            lr_mult: 1
            decay_mult: 1
          }}
          # Second param is learning/decay rate for the biases
          param {{
            lr_mult: 2
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
        ''').format(**locals())


def batch_norm(bot, top, name):
    """
    this is the SegNet version of BatchNorm, normal caffe has a different one

    References:
        https://github.com/alexgkendall/caffe-segnet/blob/288e1cb1ec3759e9e1a8fb2d7e3e314f7b35fe1e/src/caffe/proto/caffe.proto#L426
    """
    return ub.codeblock(
        '''
        layer {{
          bottom: "{bot}"
          top: "{top}"
          name: "{name}"
          type: "BN"
          # First param is learning/decay rate for the scale (weight)
          param {{
            lr_mult: 1
            decay_mult: 1
          }}
          # First param is learning/decay rate for the shift (bias)
          param {{
            lr_mult: 1
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
        ''').format(**locals())


def relu(bot, top, name):
    return ub.codeblock(
        '''
        layer {{
          bottom: "{bot}"
          top: "{top}"
          name: "{name}"
          type: "ReLU"
        }}
        ''').format(**locals())


def pool(bot, name, stride, ksize=2):
    return ub.codeblock(
        '''
        layer {{
          bottom: "{bot}"
          top: "{name}"
          top: "{name}_mask"
          name: "{name}"
          type: "Pooling"
          pooling_param {{
            pool: MAX
            kernel_size: {ksize}
            stride: {stride}
          }}
        }}
        ''').format(**locals())


def upsample(bot, suffix, scale, width=None, height=None):
    if width is not None:
        return ub.codeblock(
            '''
            layer {{
              name: "upsample{suffix}"
              type: "Upsample"
              bottom: "{bot}"
              top: "pool{suffix}_D"
              bottom: "pool{suffix}_mask"
              upsample_param {{
                scale: {scale}
                upsample_w: {width}
                upsample_h: {height}
              }}
            }}
            ''').format(**locals())
    else:
        return ub.codeblock(
            '''
            layer {{
              name: "upsample{suffix}"
              type: "Upsample"
              bottom: "{bot}"
              top: "pool{suffix}_D"
              bottom: "pool{suffix}_mask"
              upsample_param {{
                scale: {scale}
              }}
            }}
            ''').format(**locals())


def conv_bn_relu(bot, suffix, nfilt, ksize=3):
    conv_name = 'conv' + suffix
    bn_name = conv_name + '_bn'
    relu_name = 'relu' + suffix
    return [
        convolution(bot, conv_name, conv_name, nfilt=nfilt, ksize=ksize),
        batch_norm(conv_name, conv_name, bn_name),
        relu(conv_name, conv_name, relu_name),
    ]


def define_model():
    n_classes = 6

    layers = []

    # ==
    # conv1
    layers += conv_bn_relu(bot='data',    suffix='1_1', nfilt=64)
    layers += conv_bn_relu(bot='conv1_1', suffix='1_2', nfilt=64)
    layers += [pool('conv1_2', 'pool1')]

    # ====
    # conv2
    layers += conv_bn_relu(bot='pool1',   suffix='2_1', nfilt=128)
    layers += conv_bn_relu(bot='conv2_1', suffix='2_2', nfilt=128)
    layers += [pool('conv2_2', 'pool2')]

    # ======
    # conv3
    layers += conv_bn_relu(bot='pool2',   suffix='3_1', nfilt=256)
    layers += conv_bn_relu(bot='conv3_1', suffix='3_2', nfilt=256)
    layers += conv_bn_relu(bot='conv3_2', suffix='3_3', nfilt=256)
    layers += [pool('conv3_3', 'pool3')]

    # ========
    # conv4
    layers += conv_bn_relu(bot='pool3',     suffix='4_1', nfilt=512)
    layers += conv_bn_relu(bot='conv4_1',   suffix='4_2', nfilt=512)
    layers += conv_bn_relu(bot='conv4_2',   suffix='4_3', nfilt=512)
    layers += [pool('conv4_3', 'pool4')]

    # ==========
    # conv5
    layers += conv_bn_relu(bot='pool4',     suffix='5_1', nfilt=512)
    layers += conv_bn_relu(bot='conv5_1',   suffix='5_2', nfilt=512)
    layers += conv_bn_relu(bot='conv5_2',   suffix='5_3', nfilt=512)
    layers += [pool('conv5_3', 'pool5')]

    # ==========
    # up+conv5
    layers += [upsample('pool5', '5', 2, width=30, height=23)]
    layers += conv_bn_relu(bot='pool5_D',     suffix='5_3_D', nfilt=512)
    layers += conv_bn_relu(bot='conv5_3_D',   suffix='5_2_D', nfilt=512)
    layers += conv_bn_relu(bot='conv5_2_D',   suffix='5_1_D', nfilt=512)

    # ========
    # up+conv4
    layers += [upsample('conv5_1_D', '4', 2, width=60, height=45)]
    layers += conv_bn_relu(bot='pool4_D',     suffix='4_3_D', nfilt=512)
    layers += conv_bn_relu(bot='conv4_3_D',   suffix='4_2_D', nfilt=512)
    layers += conv_bn_relu(bot='conv4_2_D',   suffix='4_1_D', nfilt=256)

    # ======
    # up+conv3
    layers += [upsample('conv4_1_D', '3', 2)]
    layers += conv_bn_relu(bot='pool3_D',     suffix='3_3_D', nfilt=256)
    layers += conv_bn_relu(bot='conv3_3_D',   suffix='3_2_D', nfilt=256)
    layers += conv_bn_relu(bot='conv3_2_D',   suffix='3_1_D', nfilt=128)

    # ====
    # up+conv2
    layers += [upsample('conv3_1_D', '2', 2)]
    layers += conv_bn_relu(bot='pool2_D',     suffix='2_2_D', nfilt=128)
    layers += conv_bn_relu(bot='conv2_2_D',   suffix='2_1_D', nfilt=64)

    # ==
    # up+conv1
    layers += [upsample('conv2_1_D', '1', 2)]
    layers += conv_bn_relu(bot='pool1_D',   suffix='1_2_D', nfilt=64)

    # output mask
    layers += [convolution('conv1_2_D', 'conv1_1_D_output' + str(n_classes),
                           'conv1_1_D_output6', nfilt=n_classes)]

    core_layer_auto = '\n'.join(layers)

    core_layer_old = CORE_LAYERS.format(n_classes=n_classes)

    import utool as ut
    print(ut.color_diff_text(ut.get_textdiff(core_layer_old, core_layer_auto, num_context_lines=10)))


# Note the input layer is defined in __init__.py
CORE_LAYERS = ub.codeblock(
    '''
    layer {{
      bottom: "data"
      top: "conv1_1"
      name: "conv1_1"
      type: "Convolution"
      param {{
        lr_mult: 1
        decay_mult: 1
      }}
      param {{
        lr_mult: 2
        decay_mult: 0
      }}
      convolution_param {{
        weight_filler {{
          type: "msra"
        }}
        bias_filler {{
          type: "constant"
        }}
        num_output: 64
        pad: 1
        kernel_size: 3
      }}
    }}
    layer {{
      bottom: "conv1_1"
      top: "conv1_1"
      name: "conv1_1_bn"
      type: "BN"
      param {{
        lr_mult: 1
        decay_mult: 1
      }}
      param {{
        lr_mult: 1
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
    layer {{
      bottom: "conv1_1"
      top: "conv1_1"
      name: "relu1_1"
      type: "ReLU"
    }}
    layer {{
      bottom: "conv1_1"
      top: "conv1_2"
      name: "conv1_2"
      type: "Convolution"
      param {{
        lr_mult: 1
        decay_mult: 1
      }}
      param {{
        lr_mult: 2
        decay_mult: 0
      }}
      convolution_param {{
        weight_filler {{
          type: "msra"
        }}
        bias_filler {{
          type: "constant"
        }}
        num_output: 64
        pad: 1
        kernel_size: 3
      }}
    }}
    layer {{
      bottom: "conv1_2"
      top: "conv1_2"
      name: "conv1_2_bn"
      type: "BN"
      param {{
        lr_mult: 1
        decay_mult: 1
      }}
      param {{
        lr_mult: 1
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
    layer {{
      bottom: "conv1_2"
      top: "conv1_2"
      name: "relu1_2"
      type: "ReLU"
    }}
    layer {{
      bottom: "conv1_2"
      top: "pool1"
      top: "pool1_mask"
      name: "pool1"
      type: "Pooling"
      pooling_param {{
        pool: MAX
        kernel_size: 2
        stride: 2
      }}
    }}
    layer {{
      bottom: "pool1"
      top: "conv2_1"
      name: "conv2_1"
      type: "Convolution"
      param {{
        lr_mult: 1
        decay_mult: 1
      }}
      param {{
        lr_mult: 2
        decay_mult: 0
      }}
      convolution_param {{
        weight_filler {{
          type: "msra"
        }}
        bias_filler {{
          type: "constant"
        }}
        num_output: 128
        pad: 1
        kernel_size: 3
      }}
    }}
    layer {{
      bottom: "conv2_1"
      top: "conv2_1"
      name: "conv2_1_bn"
      type: "BN"
      param {{
        lr_mult: 1
        decay_mult: 1
      }}
      param {{
        lr_mult: 1
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
    layer {{
      bottom: "conv2_1"
      top: "conv2_1"
      name: "relu2_1"
      type: "ReLU"
    }}
    layer {{
      bottom: "conv2_1"
      top: "conv2_2"
      name: "conv2_2"
      type: "Convolution"
      param {{
        lr_mult: 1
        decay_mult: 1
      }}
      param {{
        lr_mult: 2
        decay_mult: 0
      }}
      convolution_param {{
        weight_filler {{
          type: "msra"
        }}
        bias_filler {{
          type: "constant"
        }}
        num_output: 128
        pad: 1
        kernel_size: 3
      }}
    }}
    layer {{
      bottom: "conv2_2"
      top: "conv2_2"
      name: "conv2_2_bn"
      type: "BN"
      param {{
        lr_mult: 1
        decay_mult: 1
      }}
      param {{
        lr_mult: 1
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
    layer {{
      bottom: "conv2_2"
      top: "conv2_2"
      name: "relu2_2"
      type: "ReLU"
    }}
    layer {{
      bottom: "conv2_2"
      top: "pool2"
      top: "pool2_mask"
      name: "pool2"
      type: "Pooling"
      pooling_param {{
        pool: MAX
        kernel_size: 2
        stride: 2
      }}
    }}
    layer {{
      bottom: "pool2"
      top: "conv3_1"
      name: "conv3_1"
      type: "Convolution"
      param {{
        lr_mult: 1
        decay_mult: 1
      }}
      param {{
        lr_mult: 2
        decay_mult: 0
      }}
      convolution_param {{
        weight_filler {{
          type: "msra"
        }}
        bias_filler {{
          type: "constant"
        }}
        num_output: 256
        pad: 1
        kernel_size: 3
      }}
    }}
    layer {{
      bottom: "conv3_1"
      top: "conv3_1"
      name: "conv3_1_bn"
      type: "BN"
      param {{
        lr_mult: 1
        decay_mult: 1
      }}
      param {{
        lr_mult: 1
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
    layer {{
      bottom: "conv3_1"
      top: "conv3_1"
      name: "relu3_1"
      type: "ReLU"
    }}
    layer {{
      bottom: "conv3_1"
      top: "conv3_2"
      name: "conv3_2"
      type: "Convolution"
      param {{
        lr_mult: 1
        decay_mult: 1
      }}
      param {{
        lr_mult: 2
        decay_mult: 0
      }}
      convolution_param {{
        weight_filler {{
          type: "msra"
        }}
        bias_filler {{
          type: "constant"
        }}
        num_output: 256
        pad: 1
        kernel_size: 3
      }}
    }}
    layer {{
      bottom: "conv3_2"
      top: "conv3_2"
      name: "conv3_2_bn"
      type: "BN"
      param {{
        lr_mult: 1
        decay_mult: 1
      }}
      param {{
        lr_mult: 1
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
    layer {{
      bottom: "conv3_2"
      top: "conv3_2"
      name: "relu3_2"
      type: "ReLU"
    }}
    layer {{
      bottom: "conv3_2"
      top: "conv3_3"
      name: "conv3_3"
      type: "Convolution"
      param {{
        lr_mult: 1
        decay_mult: 1
      }}
      param {{
        lr_mult: 2
        decay_mult: 0
      }}
      convolution_param {{
        weight_filler {{
          type: "msra"
        }}
        bias_filler {{
          type: "constant"
        }}
        num_output: 256
        pad: 1
        kernel_size: 3
      }}
    }}
    layer {{
      bottom: "conv3_3"
      top: "conv3_3"
      name: "conv3_3_bn"
      type: "BN"
      param {{
        lr_mult: 1
        decay_mult: 1
      }}
      param {{
        lr_mult: 1
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
    layer {{
      bottom: "conv3_3"
      top: "conv3_3"
      name: "relu3_3"
      type: "ReLU"
    }}
    layer {{
      bottom: "conv3_3"
      top: "pool3"
      top: "pool3_mask"
      name: "pool3"
      type: "Pooling"
      pooling_param {{
        pool: MAX
        kernel_size: 2
        stride: 2
      }}
    }}
    layer {{
      bottom: "pool3"
      top: "conv4_1"
      name: "conv4_1"
      type: "Convolution"
      param {{
        lr_mult: 1
        decay_mult: 1
      }}
      param {{
        lr_mult: 2
        decay_mult: 0
      }}
      convolution_param {{
        weight_filler {{
          type: "msra"
        }}
        bias_filler {{
          type: "constant"
        }}
        num_output: 512
        pad: 1
        kernel_size: 3
      }}
    }}
    layer {{
      bottom: "conv4_1"
      top: "conv4_1"
      name: "conv4_1_bn"
      type: "BN"
      param {{
        lr_mult: 1
        decay_mult: 1
      }}
      param {{
        lr_mult: 1
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
    layer {{
      bottom: "conv4_1"
      top: "conv4_1"
      name: "relu4_1"
      type: "ReLU"
    }}
    layer {{
      bottom: "conv4_1"
      top: "conv4_2"
      name: "conv4_2"
      type: "Convolution"
      param {{
        lr_mult: 1
        decay_mult: 1
      }}
      param {{
        lr_mult: 2
        decay_mult: 0
      }}
      convolution_param {{
        weight_filler {{
          type: "msra"
        }}
        bias_filler {{
          type: "constant"
        }}
        num_output: 512
        pad: 1
        kernel_size: 3
      }}
    }}
    layer {{
      bottom: "conv4_2"
      top: "conv4_2"
      name: "conv4_2_bn"
      type: "BN"
      param {{
        lr_mult: 1
        decay_mult: 1
      }}
      param {{
        lr_mult: 1
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
    layer {{
      bottom: "conv4_2"
      top: "conv4_2"
      name: "relu4_2"
      type: "ReLU"
    }}
    layer {{
      bottom: "conv4_2"
      top: "conv4_3"
      name: "conv4_3"
      type: "Convolution"
      param {{
        lr_mult: 1
        decay_mult: 1
      }}
      param {{
        lr_mult: 2
        decay_mult: 0
      }}
      convolution_param {{
        weight_filler {{
          type: "msra"
        }}
        bias_filler {{
          type: "constant"
        }}
        num_output: 512
        pad: 1
        kernel_size: 3
      }}
    }}
    layer {{
      bottom: "conv4_3"
      top: "conv4_3"
      name: "conv4_3_bn"
      type: "BN"
      param {{
        lr_mult: 1
        decay_mult: 1
      }}
      param {{
        lr_mult: 1
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
    layer {{
      bottom: "conv4_3"
      top: "conv4_3"
      name: "relu4_3"
      type: "ReLU"
    }}
    layer {{
      bottom: "conv4_3"
      top: "pool4"
      top: "pool4_mask"
      name: "pool4"
      type: "Pooling"
      pooling_param {{
        pool: MAX
        kernel_size: 2
        stride: 2
      }}
    }}
    layer {{
      bottom: "pool4"
      top: "conv5_1"
      name: "conv5_1"
      type: "Convolution"
      param {{
        lr_mult: 1
        decay_mult: 1
      }}
      param {{
        lr_mult: 2
        decay_mult: 0
      }}
      convolution_param {{
        weight_filler {{
          type: "msra"
        }}
        bias_filler {{
          type: "constant"
        }}
        num_output: 512
        pad: 1
        kernel_size: 3
      }}
    }}
    layer {{
      bottom: "conv5_1"
      top: "conv5_1"
      name: "conv5_1_bn"
      type: "BN"
      param {{
        lr_mult: 1
        decay_mult: 1
      }}
      param {{
        lr_mult: 1
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
    layer {{
      bottom: "conv5_1"
      top: "conv5_1"
      name: "relu5_1"
      type: "ReLU"
    }}
    layer {{
      bottom: "conv5_1"
      top: "conv5_2"
      name: "conv5_2"
      type: "Convolution"
      param {{
        lr_mult: 1
        decay_mult: 1
      }}
      param {{
        lr_mult: 2
        decay_mult: 0
      }}
      convolution_param {{
        weight_filler {{
          type: "msra"
        }}
        bias_filler {{
          type: "constant"
        }}
        num_output: 512
        pad: 1
        kernel_size: 3
      }}
    }}
    layer {{
      bottom: "conv5_2"
      top: "conv5_2"
      name: "conv5_2_bn"
      type: "BN"
      param {{
        lr_mult: 1
        decay_mult: 1
      }}
      param {{
        lr_mult: 1
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
    layer {{
      bottom: "conv5_2"
      top: "conv5_2"
      name: "relu5_2"
      type: "ReLU"
    }}
    layer {{
      bottom: "conv5_2"
      top: "conv5_3"
      name: "conv5_3"
      type: "Convolution"
      param {{
        lr_mult: 1
        decay_mult: 1
      }}
      param {{
        lr_mult: 2
        decay_mult: 0
      }}
      convolution_param {{
        weight_filler {{
          type: "msra"
        }}
        bias_filler {{
          type: "constant"
        }}
        num_output: 512
        pad: 1
        kernel_size: 3
      }}
    }}
    layer {{
      bottom: "conv5_3"
      top: "conv5_3"
      name: "conv5_3_bn"
      type: "BN"
      param {{
        lr_mult: 1
        decay_mult: 1
      }}
      param {{
        lr_mult: 1
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
    layer {{
      bottom: "conv5_3"
      top: "conv5_3"
      name: "relu5_3"
      type: "ReLU"
    }}
    layer {{
      bottom: "conv5_3"
      top: "pool5"
      top: "pool5_mask"
      name: "pool5"
      type: "Pooling"
      pooling_param {{
        pool: MAX
        kernel_size: 2
        stride: 2
      }}
    }}
    layer {{
      name: "upsample5"
      type: "Upsample"
      bottom: "pool5"
      top: "pool5_D"
      bottom: "pool5_mask"
      upsample_param {{
        scale: 2
        upsample_w: 30
        upsample_h: 23
      }}
    }}
    layer {{
      bottom: "pool5_D"
      top: "conv5_3_D"
      name: "conv5_3_D"
      type: "Convolution"
      param {{
        lr_mult: 1
        decay_mult: 1
      }}
      param {{
        lr_mult: 2
        decay_mult: 0
      }}
      convolution_param {{
        weight_filler {{
          type: "msra"
        }}
        bias_filler {{
          type: "constant"
        }}
        num_output: 512
        pad: 1
        kernel_size: 3
      }}
    }}
    layer {{
      bottom: "conv5_3_D"
      top: "conv5_3_D"
      name: "conv5_3_D_bn"
      type: "BN"
      param {{
        lr_mult: 1
        decay_mult: 1
      }}
      param {{
        lr_mult: 1
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
    layer {{
      bottom: "conv5_3_D"
      top: "conv5_3_D"
      name: "relu5_3_D"
      type: "ReLU"
    }}
    layer {{
      bottom: "conv5_3_D"
      top: "conv5_2_D"
      name: "conv5_2_D"
      type: "Convolution"
      param {{
        lr_mult: 1
        decay_mult: 1
      }}
      param {{
        lr_mult: 2
        decay_mult: 0
      }}
      convolution_param {{
        weight_filler {{
          type: "msra"
        }}
        bias_filler {{
          type: "constant"
        }}
        num_output: 512
        pad: 1
        kernel_size: 3
      }}
    }}
    layer {{
      bottom: "conv5_2_D"
      top: "conv5_2_D"
      name: "conv5_2_D_bn"
      type: "BN"
      param {{
        lr_mult: 1
        decay_mult: 1
      }}
      param {{
        lr_mult: 1
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
    layer {{
      bottom: "conv5_2_D"
      top: "conv5_2_D"
      name: "relu5_2_D"
      type: "ReLU"
    }}
    layer {{
      bottom: "conv5_2_D"
      top: "conv5_1_D"
      name: "conv5_1_D"
      type: "Convolution"
      param {{
        lr_mult: 1
        decay_mult: 1
      }}
      param {{
        lr_mult: 2
        decay_mult: 0
      }}
      convolution_param {{
        weight_filler {{
          type: "msra"
        }}
        bias_filler {{
          type: "constant"
        }}
        num_output: 512
        pad: 1
        kernel_size: 3
      }}
    }}
    layer {{
      bottom: "conv5_1_D"
      top: "conv5_1_D"
      name: "conv5_1_D_bn"
      type: "BN"
      param {{
        lr_mult: 1
        decay_mult: 1
      }}
      param {{
        lr_mult: 1
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
    layer {{
      bottom: "conv5_1_D"
      top: "conv5_1_D"
      name: "relu5_1_D"
      type: "ReLU"
    }}
    layer {{
      name: "upsample4"
      type: "Upsample"
      bottom: "conv5_1_D"
      top: "pool4_D"
      bottom: "pool4_mask"
      upsample_param {{
        scale: 2
        upsample_w: 60
        upsample_h: 45
      }}
    }}
    layer {{
      bottom: "pool4_D"
      top: "conv4_3_D"
      name: "conv4_3_D"
      type: "Convolution"
      param {{
        lr_mult: 1
        decay_mult: 1
      }}
      param {{
        lr_mult: 2
        decay_mult: 0
      }}
      convolution_param {{
        weight_filler {{
          type: "msra"
        }}
        bias_filler {{
          type: "constant"
        }}
        num_output: 512
        pad: 1
        kernel_size: 3
      }}
    }}
    layer {{
      bottom: "conv4_3_D"
      top: "conv4_3_D"
      name: "conv4_3_D_bn"
      type: "BN"
      param {{
        lr_mult: 1
        decay_mult: 1
      }}
      param {{
        lr_mult: 1
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
    layer {{
      bottom: "conv4_3_D"
      top: "conv4_3_D"
      name: "relu4_3_D"
      type: "ReLU"
    }}
    layer {{
      bottom: "conv4_3_D"
      top: "conv4_2_D"
      name: "conv4_2_D"
      type: "Convolution"
      param {{
        lr_mult: 1
        decay_mult: 1
      }}
      param {{
        lr_mult: 2
        decay_mult: 0
      }}
      convolution_param {{
        weight_filler {{
          type: "msra"
        }}
        bias_filler {{
          type: "constant"
        }}
        num_output: 512
        pad: 1
        kernel_size: 3
      }}
    }}
    layer {{
      bottom: "conv4_2_D"
      top: "conv4_2_D"
      name: "conv4_2_D_bn"
      type: "BN"
      param {{
        lr_mult: 1
        decay_mult: 1
      }}
      param {{
        lr_mult: 1
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
    layer {{
      bottom: "conv4_2_D"
      top: "conv4_2_D"
      name: "relu4_2_D"
      type: "ReLU"
    }}
    layer {{
      bottom: "conv4_2_D"
      top: "conv4_1_D"
      name: "conv4_1_D"
      type: "Convolution"
      param {{
        lr_mult: 1
        decay_mult: 1
      }}
      param {{
        lr_mult: 2
        decay_mult: 0
      }}
      convolution_param {{
        weight_filler {{
          type: "msra"
        }}
        bias_filler {{
          type: "constant"
        }}
        num_output: 256
        pad: 1
        kernel_size: 3
      }}
    }}
    layer {{
      bottom: "conv4_1_D"
      top: "conv4_1_D"
      name: "conv4_1_D_bn"
      type: "BN"
      param {{
        lr_mult: 1
        decay_mult: 1
      }}
      param {{
        lr_mult: 1
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
    layer {{
      bottom: "conv4_1_D"
      top: "conv4_1_D"
      name: "relu4_1_D"
      type: "ReLU"
    }}
    layer {{
      name: "upsample3"
      type: "Upsample"
      bottom: "conv4_1_D"
      top: "pool3_D"
      bottom: "pool3_mask"
      upsample_param {{
        scale: 2
      }}
    }}
    layer {{
      bottom: "pool3_D"
      top: "conv3_3_D"
      name: "conv3_3_D"
      type: "Convolution"
      param {{
        lr_mult: 1
        decay_mult: 1
      }}
      param {{
        lr_mult: 2
        decay_mult: 0
      }}
      convolution_param {{
        weight_filler {{
          type: "msra"
        }}
        bias_filler {{
          type: "constant"
        }}
        num_output: 256
        pad: 1
        kernel_size: 3
      }}
    }}
    layer {{
      bottom: "conv3_3_D"
      top: "conv3_3_D"
      name: "conv3_3_D_bn"
      type: "BN"
      param {{
        lr_mult: 1
        decay_mult: 1
      }}
      param {{
        lr_mult: 1
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
    layer {{
      bottom: "conv3_3_D"
      top: "conv3_3_D"
      name: "relu3_3_D"
      type: "ReLU"
    }}
    layer {{
      bottom: "conv3_3_D"
      top: "conv3_2_D"
      name: "conv3_2_D"
      type: "Convolution"
      param {{
        lr_mult: 1
        decay_mult: 1
      }}
      param {{
        lr_mult: 2
        decay_mult: 0
      }}
      convolution_param {{
        weight_filler {{
          type: "msra"
        }}
        bias_filler {{
          type: "constant"
        }}
        num_output: 256
        pad: 1
        kernel_size: 3
      }}
    }}
    layer {{
      bottom: "conv3_2_D"
      top: "conv3_2_D"
      name: "conv3_2_D_bn"
      type: "BN"
      param {{
        lr_mult: 1
        decay_mult: 1
      }}
      param {{
        lr_mult: 1
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
    layer {{
      bottom: "conv3_2_D"
      top: "conv3_2_D"
      name: "relu3_2_D"
      type: "ReLU"
    }}
    layer {{
      bottom: "conv3_2_D"
      top: "conv3_1_D"
      name: "conv3_1_D"
      type: "Convolution"
      param {{
        lr_mult: 1
        decay_mult: 1
      }}
      param {{
        lr_mult: 2
        decay_mult: 0
      }}
      convolution_param {{
        weight_filler {{
          type: "msra"
        }}
        bias_filler {{
          type: "constant"
        }}
        num_output: 128
        pad: 1
        kernel_size: 3
      }}
    }}
    layer {{
      bottom: "conv3_1_D"
      top: "conv3_1_D"
      name: "conv3_1_D_bn"
      type: "BN"
      param {{
        lr_mult: 1
        decay_mult: 1
      }}
      param {{
        lr_mult: 1
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
    layer {{
      bottom: "conv3_1_D"
      top: "conv3_1_D"
      name: "relu3_1_D"
      type: "ReLU"
    }}
    layer {{
      name: "upsample2"
      type: "Upsample"
      bottom: "conv3_1_D"
      top: "pool2_D"
      bottom: "pool2_mask"
      upsample_param {{
        scale: 2
      }}
    }}
    layer {{
      bottom: "pool2_D"
      top: "conv2_2_D"
      name: "conv2_2_D"
      type: "Convolution"
      param {{
        lr_mult: 1
        decay_mult: 1
      }}
      param {{
        lr_mult: 2
        decay_mult: 0
      }}
      convolution_param {{
        weight_filler {{
          type: "msra"
        }}
        bias_filler {{
          type: "constant"
        }}
        num_output: 128
        pad: 1
        kernel_size: 3
      }}
    }}
    layer {{
      bottom: "conv2_2_D"
      top: "conv2_2_D"
      name: "conv2_2_D_bn"
      type: "BN"
      param {{
        lr_mult: 1
        decay_mult: 1
      }}
      param {{
        lr_mult: 1
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
    layer {{
      bottom: "conv2_2_D"
      top: "conv2_2_D"
      name: "relu2_2_D"
      type: "ReLU"
    }}
    layer {{
      bottom: "conv2_2_D"
      top: "conv2_1_D"
      name: "conv2_1_D"
      type: "Convolution"
      param {{
        lr_mult: 1
        decay_mult: 1
      }}
      param {{
        lr_mult: 2
        decay_mult: 0
      }}
      convolution_param {{
        weight_filler {{
          type: "msra"
        }}
        bias_filler {{
          type: "constant"
        }}
        num_output: 64
        pad: 1
        kernel_size: 3
      }}
    }}
    layer {{
      bottom: "conv2_1_D"
      top: "conv2_1_D"
      name: "conv2_1_D_bn"
      type: "BN"
      param {{
        lr_mult: 1
        decay_mult: 1
      }}
      param {{
        lr_mult: 1
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
    layer {{
      bottom: "conv2_1_D"
      top: "conv2_1_D"
      name: "relu2_1_D"
      type: "ReLU"
    }}
    layer {{
      name: "upsample1"
      type: "Upsample"
      bottom: "conv2_1_D"
      top: "pool1_D"
      bottom: "pool1_mask"
      upsample_param {{
        scale: 2
      }}
    }}
    layer {{
      bottom: "pool1_D"
      top: "conv1_2_D"
      name: "conv1_2_D"
      type: "Convolution"
      param {{
        lr_mult: 1
        decay_mult: 1
      }}
      param {{
        lr_mult: 2
        decay_mult: 0
      }}
      convolution_param {{
        weight_filler {{
          type: "msra"
        }}
        bias_filler {{
          type: "constant"
        }}
        num_output: 64
        pad: 1
        kernel_size: 3
      }}
    }}
    layer {{
      bottom: "conv1_2_D"
      top: "conv1_2_D"
      name: "conv1_2_D_bn"
      type: "BN"
      param {{
        lr_mult: 1
        decay_mult: 1
      }}
      param {{
        lr_mult: 1
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
    layer {{
      bottom: "conv1_2_D"
      top: "conv1_2_D"
      name: "relu1_2_D"
      type: "ReLU"
    }}
    layer {{
      bottom: "conv1_2_D"
      top: "conv1_1_D_output{n_classes}"
      name: "conv1_1_D_output{n_classes}"
      type: "Convolution"
      param {{
        lr_mult: 1
        decay_mult: 1
      }}
      param {{
        lr_mult: 2
        decay_mult: 0
      }}
      convolution_param {{
        weight_filler {{
          type: "msra"
        }}
        bias_filler {{
          type: "constant"
        }}
        num_output: {n_classes}
        pad: 1
        kernel_size: 3
      }}
    }}
    ''')


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
