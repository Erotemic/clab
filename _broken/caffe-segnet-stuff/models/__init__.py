# -*- coding: utf-8 -*-
"""
python -c "import ubelt._internal as a; a.autogen_init('pysseg.models', imports=['proto'])"
"""
# flake8: noqa
from __future__ import absolute_import, division, print_function, unicode_literals
from pysseg.models import proto
from pysseg.models.proto import (default_hyperparams, make_model_file,
                                 make_solver_file, make_input_file,
                                 make_solver, make_prototext,)