"""
python -c "import ubelt._internal as a; a.autogen_init('clab.augment')"
"""
# flake8: noqa
from .augment import augment_sseg_offline
from .augment import augment_numpy
from .augment import augment_common
from .augment import augment_pil_online
from .augment import augment_numpy_online
from .augment.augment_sseg_offline import (SSegAugmentor,)

from .augment.augment_common import (random_affine_args, affine_mat2x3,
                                           affine_around_mat2x3,)
from .augment.augment_pil_online import (online_affine_perterb,
                                               online_intensity_augment,)
from .augment.augment_numpy_online import (online_affine_perterb_np,
                                                 online_intensity_augment_np,)
