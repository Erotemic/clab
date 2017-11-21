# -*- coding: utf-8 -*-
"""
python -c "import ubelt._internal as a; a.autogen_init('clab.tasks')"
"""
# flake8: noqa
from __future__ import absolute_import, division, print_function, unicode_literals
from .tasks import camvid
from .tasks import diva_v1
from .tasks import cityscapes
from .tasks.camvid import (CamVid,)
from .tasks.diva_v1 import (DivaV1,)
from .tasks.cityscapes import (CityScapes,)