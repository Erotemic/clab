# -*- coding: utf-8 -*-
"""
python -c "import ubelt._internal as a; a.autogen_init('clab.torch.models')"
"""
# flake8: noqa
from __future__ import absolute_import, division, print_function, unicode_literals
from clab.torch.models import unet_aux
from clab.torch.models import unet
from clab.torch.models import siamese
from clab.torch.models import mnist_net
from clab.torch.models import fcn
from clab.torch.models import sseg_dummy
from clab.torch.models import segnet
from clab.torch.models import pspnet
from clab.torch.models import linknet
from clab.torch.models.unet_aux import (InputAux2,)
from clab.torch.models.unet import (UNet,)
from clab.torch.models.siamese import (SiameseLP, SiameseCLF,)
from clab.torch.models.mnist_net import (MnistNet,)
from clab.torch.models.fcn import (FCN32, FCN16, FNC8,)
from clab.torch.models.sseg_dummy import (SSegDummy,)
from clab.torch.models.segnet import (SegNet,)
from clab.torch.models.pspnet import (PSPNet,)
from clab.torch.models.linknet import (LinkNet,)