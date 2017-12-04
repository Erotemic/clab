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
    from clab.util import colorutil
    from clab.util import fnameutil
    from clab.util import gpu_util
    from clab.util import hashutil
    from clab.util import imutil
    from clab.util import jsonutil
    from clab.util import misc
    from clab.util import mplutil
    from clab.util import nputil
    from clab.util.colorutil import (colorbar_image, convert_hex_to_255,
                                     lookup_bgr255, make_distinct_bgr01_colors,
                                     make_heatmask,)
    from clab.util.fnameutil import (align_paths, check_aligned, dumpsafe,
                                     shortest_unique_prefixes,
                                     shortest_unique_suffixes,)
    from clab.util.gpu_util import (find_unused_gpu, gpu_info, have_gpu, num_gpus,)
    from clab.util.hashutil import (get_file_hash, hash_data,)
    from clab.util.imutil import (CV2_INTERPOLATION_TYPES, InternalRunningStats,
                                  RunningStats, absdev, adjust_gamma,
                                  atleast_3channels, convert_colorspace,
                                  ensure_alpha_channel, ensure_float01,
                                  ensure_grayscale, get_num_channels, image_slices,
                                  imread, imscale, imwrite, load_image_paths,
                                  logger, make_channels_comparable,
                                  overlay_alpha_images, overlay_colorized, print,
                                  putMultiLineText, run_length_encoding,
                                  wide_strides_1d,)
    from clab.util.jsonutil import (JSONEncoder, NumpyAwareJSONEncoder,
                                    NumpyEncoder, json_numpy_obj_hook, read_json,
                                    walk_json, write_json,)
    from clab.util.misc import (cc_locs, compact_idstr, isiterable, read_arr,
                                read_h5arr, roundrobin, super2, write_arr,
                                write_h5arr,)
    from clab.util.mplutil import (adjust_subplots, axes_extent,
                                   copy_figure_to_clipboard, extract_axes_extents,
                                   figure, pandas_plot_matrix,
                                   render_figure_to_image, savefig2,)
    from clab.util.nputil import (apply_grouping, atleast_nd, group_indices,
                                  group_items, isect_flags, iter_reduce_ufunc,)
    # </AUTOGEN_INIT>