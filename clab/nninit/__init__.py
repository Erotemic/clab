"""
python -c "import ubelt._internal as a; a.autogen_init('clab.nninit')"
python -m clab
"""
# flake8: noqa
from clab.nninit import base
from clab.nninit import lsuv
from clab.nninit.base import (HeNormal, KaimingNormal, KaimingUniform, NoOp,
                              Orthogonal, Pretrained, VGG16, apply_initializer,
                              constant, he_normal, he_uniform, init_he_normal,
                              kaiming_normal, kaiming_uniform,
                              load_partial_state, normal, orthogonal, shock,
                              shock_he, sparse, trainable_layers, uniform,
                              xavier_normal, xavier_uniform,)
from clab.nninit.lsuv import (LSUV, Orthonormal, svd_orthonormal,)
