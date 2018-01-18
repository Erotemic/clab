"""
python -c "import ubelt._internal as a; a.autogen_init('clab.augment')"
"""
# flake8: noqa
from clab.augment import augment_common
from clab.augment.augment_common import (PERTERB_AUG_KW, affine_around_mat2x3,
                                         affine_mat2x3, random_affine_args,)