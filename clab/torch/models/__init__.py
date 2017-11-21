# -*- coding: utf-8 -*-
"""
python -c "import ubelt._internal as a; a.autogen_init('clab.torch.models')"
"""
# flake8: noqa
from __future__ import absolute_import, division, print_function, unicode_literals
from .torch.models import unet_aux
from .torch.models import unet
from .torch.models import siamese
from .torch.models import mnist_net
from .torch.models import fcn
from .torch.models import sseg_dummy
from .torch.models import segnet
from .torch.models import pspnet
from .torch.models import linknet
from .torch.models.unet_aux import (InputAux2,)
from .torch.models.unet import (UNet,)
from .torch.models.siamese import (SiameseLP, SiameseCLF,)
from .torch.models.mnist_net import (MnistNet,)
from .torch.models.fcn import (FCN32, FCN16, FNC8,)
from .torch.models.sseg_dummy import (SSegDummy,)
from .torch.models.segnet import (SegNet,)
from .torch.models.pspnet import (PSPNet,)
from .torch.models.linknet import (LinkNet,)