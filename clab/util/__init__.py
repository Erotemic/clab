"""
python -c "import ubelt._internal as a; a.autogen_init('clab.util')"
python -m clab
"""
# flake8: noqa
from __future__ import absolute_import, division, print_function, unicode_literals

__DYNAMIC__ = True
if __DYNAMIC__:
    from ubelt._internal import dynamic_import
    exec(dynamic_import(__name__))
else:
    # <AUTOGEN_INIT>
    from .util import colorutil
    from .util import nputil
    from .util import hashutil
    from .util import gpu_util
    from .util import fnameutil
    from .util import imutil
    from .util import jsonutil
    from .util import misc
    from .util import utildraw
    from .util.colorutil import (make_distinct_bgr01_colors,
                                       convert_hex_to_255, lookup_bgr255,)
    from .util.nputil import (iter_reduce_ufunc, isect_flags, atleast_nd,)
    from .util.hashutil import (hash_data, get_file_hash,)
    from .util.gpu_util import (have_gpu, find_unused_gpu, gpu_info,)
    from .util.fnameutil import (dumpsafe, shortest_unique_prefixes,
                                       shortest_unique_suffixes, align_paths,
                                       check_aligned,)
    from .util.imutil import (imscale, adjust_gamma, rectify_to_float01,
                                    make_channels_comparable, get_num_channels,
                                    overlay_alpha_images, ensure_alpha_channel,
                                    ensure_grayscale, convert_colorspace,
                                    overlay_colorized, load_image_paths, imread,
                                    imwrite, wide_strides_1d, image_slices,
                                    run_length_encoding,)
    from .util.jsonutil import (walk_json, JSONEncoder, write_json,)
    from .util.misc import (isiterable, super2, roundrobin,)
    from .util.utildraw import (figure, pandas_plot_matrix, axes_extent,
                                      extract_axes_extents, adjust_subplots,
                                      render_figure_to_image, savefig2,
                                      copy_figure_to_clipboard,)
    # </AUTOGEN_INIT>