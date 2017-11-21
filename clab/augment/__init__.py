"""
python -c "import ubelt._internal as a; a.autogen_init('clab.augment')"
"""
# flake8: noqa
from clab.augment import augment_common
from clab.augment import augment_numpy
from clab.augment import augment_numpy_online
from clab.augment import augment_pil_online
from clab.augment import augment_sseg_offline
from clab.augment.augment_common import (PERTERB_AUG_KW, affine_around_mat2x3,
                                         affine_mat2x3, random_affine_args,)

from clab.augment.augment_numpy_online import (SKIMAGE_INTERP_LOOKUP,
                                               online_affine_perterb_np,
                                               online_intensity_augment_np,)
from clab.augment.augment_pil_online import (online_affine_perterb,
                                             online_intensity_augment,)
from clab.augment.augment_sseg_offline import (SSegAugmentor,)