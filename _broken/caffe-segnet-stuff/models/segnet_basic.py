# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import ubelt as ub


# # See this for info on params
# https://github.com/BVLC/caffe/blob/master/src/caffe/proto/caffe.proto#L102
DEFAULT_HYPERPARAMS = {
    'base_lr': 0.001,
    'lr_policy': "step",
    'gamma': 1.0,
    'stepsize': 10000000,
    'momentum': 0.9,
    'weight_decay': 0.0005,
    'test_initialization': False,
    'test_iter': 1,
    'test_interval': 10000000,
    'display': 20,
    'max_iter': 10000,
    'snapshot': 1000,
}


# Note the input layer is defined in __init__.py

CORE_LAYERS = ub.codeblock(
    """
    layer {{
      name: "norm"
      type: "LRN"
      bottom: "data"
      top: "norm"
      lrn_param {{
        local_size: 5
        alpha: 0.0001
        beta: 0.75
      }}
    }}
    layer {{
      name: "conv1"
      type: "Convolution"
      bottom: "norm"
      top: "conv1"
      param {{
        lr_mult: 1
        decay_mult: 1
      }}
      param {{
        lr_mult: 2
        decay_mult: 0
      }}
      convolution_param {{
        num_output: 64
        kernel_size: 7
        pad: 3
        weight_filler {{
          type: "msra"
        }}
        bias_filler {{
          type: "constant"
        }}
      }}
    }}
    layer {{
      bottom: "conv1"
      top: "conv1"
      name: "conv1_bn"
      type: "BN"
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
      name: "relu1"
      type: "ReLU"
      bottom: "conv1"
      top: "conv1"
    }}
    layer {{
      name: "pool1"
      type: "Pooling"
      bottom: "conv1"
      top: "pool1"
      top: "pool1_mask"
      pooling_param {{
        pool: MAX
        kernel_size: 2
        stride: 2
      }}
    }}
    layer {{
      name: "conv2"
      type: "Convolution"
      bottom: "pool1"
      top: "conv2"
      param {{
        lr_mult: 1
        decay_mult: 1
      }}
      param {{
        lr_mult: 2
        decay_mult: 0
      }}
      convolution_param {{
        num_output: 64
        kernel_size: 7
        pad: 3
        weight_filler {{
          type: "msra"
        }}
        bias_filler {{
          type: "constant"
        }}
      }}
    }}
    layer {{
      bottom: "conv2"
      top: "conv2"
      name: "conv2_bn"
      type: "BN"
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
      name: "relu2"
      type: "ReLU"
      bottom: "conv2"
      top: "conv2"
    }}
    layer {{
      name: "pool2"
      type: "Pooling"
      bottom: "conv2"
      top: "pool2"
      top: "pool2_mask"
      pooling_param {{
        pool: MAX
        kernel_size: 2
        stride: 2
      }}
    }}
    layer {{
      name: "conv3"
      type: "Convolution"
      bottom: "pool2"
      top: "conv3"
      param {{
        lr_mult: 1
        decay_mult: 1
      }}
      param {{
        lr_mult: 2
        decay_mult: 0
      }}
      convolution_param {{
        num_output: 64
        kernel_size: 7
        pad: 3
        weight_filler {{
          type: "msra"
        }}
        bias_filler {{
          type: "constant"
        }}
      }}
    }}
    layer {{
      bottom: "conv3"
      top: "conv3"
      name: "conv3_bn"
      type: "BN"
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
      name: "relu3"
      type: "ReLU"
      bottom: "conv3"
      top: "conv3"
    }}
    layer {{
      name: "pool3"
      type: "Pooling"
      bottom: "conv3"
      top: "pool3"
      top: "pool3_mask"
      pooling_param {{
        pool: MAX
        kernel_size: 2
        stride: 2
      }}
    }}
    layer {{
      name: "conv4"
      type: "Convolution"
      bottom: "pool3"
      top: "conv4"
      param {{
        lr_mult: 1
        decay_mult: 1
      }}
      param {{
        lr_mult: 2
        decay_mult: 0
      }}
      convolution_param {{
        num_output: 64
        kernel_size: 7
        pad: 3
        weight_filler {{
          type: "msra"
        }}
        bias_filler {{
          type: "constant"
        }}
      }}
    }}
    layer {{
      bottom: "conv4"
      top: "conv4"
      name: "conv4_bn"
      type: "BN"
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
      name: "relu4"
      type: "ReLU"
      bottom: "conv4"
      top: "conv4"
    }}
    layer {{
      name: "pool4"
      type: "Pooling"
      bottom: "conv4"
      top: "pool4"
      top: "pool4_mask"
      pooling_param {{
        pool: MAX
        kernel_size: 2
        stride: 2
      }}
    }}
    layer {{
      name: "upsample4"
      type: "Upsample"
      bottom: "pool4"
      bottom: "pool4_mask"
      top: "upsample4"
      upsample_param {{
        scale: 2
        pad_out_h: true
      }}
    }}
    layer {{
      name: "conv_decode4"
      type: "Convolution"
      bottom: "upsample4"
      top: "conv_decode4"
      param {{
        lr_mult: 1
        decay_mult: 1
      }}
      param {{
        lr_mult: 2
        decay_mult: 0
      }}
      convolution_param {{
        num_output: 64
        kernel_size: 7
        pad: 3
        weight_filler {{
          type: "msra"
        }}
        bias_filler {{
          type: "constant"
        }}
      }}
    }}
    layer {{
      bottom: "conv_decode4"
      top: "conv_decode4"
      name: "conv_decode4_bn"
      type: "BN"
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
      name: "upsample3"
      type: "Upsample"
      bottom: "conv_decode4"
      bottom: "pool3_mask"
      top: "upsample3"
      upsample_param {{
        scale: 2
      }}
    }}
    layer {{
      name: "conv_decode3"
      type: "Convolution"
      bottom: "upsample3"
      top: "conv_decode3"
      param {{
        lr_mult: 1
        decay_mult: 1
      }}
      param {{
        lr_mult: 2
        decay_mult: 0
      }}
      convolution_param {{
        num_output: 64
        kernel_size: 7
        pad: 3
        weight_filler {{
          type: "msra"
        }}
        bias_filler {{
          type: "constant"
        }}
      }}
    }}
    layer {{
      bottom: "conv_decode3"
      top: "conv_decode3"
      name: "conv_decode3_bn"
      type: "BN"
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
      name: "upsample2"
      type: "Upsample"
      bottom: "conv_decode3"
      bottom: "pool2_mask"
      top: "upsample2"
      upsample_param {{
        scale: 2
      }}
    }}
    layer {{
      name: "conv_decode2"
      type: "Convolution"
      bottom: "upsample2"
      top: "conv_decode2"
      param {{
        lr_mult: 1
        decay_mult: 1
      }}
      param {{
        lr_mult: 2
        decay_mult: 0
      }}
      convolution_param {{
        num_output: 64
        kernel_size: 7
        pad: 3
        weight_filler {{
          type: "msra"
        }}
        bias_filler {{
          type: "constant"
        }}
      }}
    }}
    layer {{
      bottom: "conv_decode2"
      top: "conv_decode2"
      name: "conv_decode2_bn"
      type: "BN"
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
      name: "upsample1"
      type: "Upsample"
      bottom: "conv_decode2"
      bottom: "pool1_mask"
      top: "upsample1"
      upsample_param {{
        scale: 2
      }}
    }}
    layer {{
      name: "conv_decode1"
      type: "Convolution"
      bottom: "upsample1"
      top: "conv_decode1"
      param {{
        lr_mult: 1
        decay_mult: 1
      }}
      param {{
        lr_mult: 2
        decay_mult: 0
      }}
      convolution_param {{
        num_output: 64
        kernel_size: 7
        pad: 3
        weight_filler {{
          type: "msra"
        }}
        bias_filler {{
          type: "constant"
        }}
      }}
    }}
    layer {{
      bottom: "conv_decode1"
      top: "conv_decode1"
      name: "conv_decode1_bn"
      type: "BN"
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
      name: "conv_classifier_output{n_classes}"
      type: "Convolution"
      bottom: "conv_decode1"
      top: "conv_classifier_output{n_classes}"
      param {{
        lr_mult: 1
        decay_mult: 1
      }}
      param {{
        lr_mult: 2
        decay_mult: 0
      }}
      convolution_param {{
        num_output: {n_classes}
        kernel_size: 1
        weight_filler {{
          type: "msra"
        }}
        bias_filler {{
          type: "constant"
        }}
      }}
    }}
    """)


FIT_FOOTER = ub.codeblock(
    '''
    layer {{
      name: "loss"
      type: "SoftmaxWithLoss"
      bottom: "conv_classifier_output{n_classes}"
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
      bottom: "conv_classifier_output{n_classes}"
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
      bottom: "conv_classifier_output{n_classes}"
      top: "prob"
      softmax_param {{engine: CAFFE}}
    }}
    ''')
