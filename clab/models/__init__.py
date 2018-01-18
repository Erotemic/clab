# -*- coding: utf-8 -*-
"""
python -c "import ubelt._internal as a; a.autogen_init('clab.models')"
"""
# flake8: noqa
from __future__ import absolute_import, division, print_function, unicode_literals
from clab.models import unet_aux
from clab.models import unet
from clab.models import siamese
from clab.models import mnist_net
from clab.models import fcn
from clab.models import sseg_dummy
from clab.models import segnet
from clab.models import pspnet
from clab.models import linknet
from clab.models.unet_aux import (InputAux2,)
from clab.models.unet import (UNet,)
from clab.models.siamese import (SiameseLP, SiameseCLF,)
from clab.models.mnist_net import (MnistNet,)
from clab.models.fcn import (FCN32, FCN16, FNC8,)
from clab.models.sseg_dummy import (SSegDummy,)
from clab.models.segnet import (SegNet,)
from clab.models.pspnet import (PSPNet,)
from clab.models.linknet import (LinkNet,)