"""
mkinit clab.util
"""
# flake8: noqa
from __future__ import absolute_import, division, print_function, unicode_literals

__DYNAMIC__ = False
if __DYNAMIC__:
    import mkinit
    exec(mkinit.mkinit.dynamic_init(__name__))
else:
    # <AUTOGEN_INIT>
    from clab.util import colorutil
    from clab.util import coverage_kpts
    from clab.util import fnameutil
    from clab.util import gpu_util
    from clab.util import hashutil
    from clab.util import imutil
    from clab.util import jsonutil
    from clab.util import misc
    from clab.util import mplutil
    from clab.util import nputil
    from clab.util import nxutil
    from clab.util import priority_queue
    from clab.util import profiler
    from clab.util import util_affine
    from clab.util import util_alg
    from clab.util import util_averages
    from clab.util.colorutil import (colorbar_image, convert_hex_to_255,
                                     lookup_bgr255, make_distinct_bgr01_colors,
                                     make_distinct_bgr255_colors, make_heatmask,)
    from clab.util.coverage_kpts import (gaussian_patch, make_kpts_coverage_mask,
                                         make_kpts_heatmask,)
    from clab.util.fnameutil import (align_paths, check_aligned, dumpsafe,
                                     shortest_unique_prefixes,
                                     shortest_unique_suffixes,)
    from clab.util.gpu_util import (find_unused_gpu, gpu_info, have_gpu, num_gpus,)
    from clab.util.hashutil import (hash_data, hash_file,)
    from clab.util.imutil import (CV2_INTERPOLATION_TYPES, adjust_gamma,
                                  atleast_3channels, convert_colorspace,
                                  ensure_alpha_channel, ensure_float01,
                                  ensure_grayscale, get_num_channels,
                                  grab_test_imgpath, image_slices, imread, imscale,
                                  imwrite, load_image_paths, logger,
                                  make_channels_comparable, overlay_alpha_images,
                                  overlay_colorized, putMultiLineText,
                                  run_length_encoding, wide_strides_1d,)
    from clab.util.jsonutil import (JSONEncoder, NumpyAwareJSONEncoder,
                                    NumpyEncoder, json_numpy_obj_hook, read_json,
                                    walk_json, write_json,)
    from clab.util.misc import (Boxes, PauseTQDM, cc_locs,
                                clean_tensorboard_protobufs, compact_idstr,
                                ensure_rng, get_stack_frame, grab_test_image,
                                isiterable, make_idstr, make_short_idstr,
                                protect_print, random_indices, read_arr,
                                read_h5arr, roundrobin, scale_boxes, super2,
                                write_arr, write_h5arr,)
    from clab.util.mplutil import (Color, PlotNums, adjust_subplots, axes_extent,
                                   colorbar, copy_figure_to_clipboard,
                                   deterministic_shuffle, dict_intersection,
                                   distinct_colors, distinct_markers, draw_border,
                                   draw_boxes, draw_line_segments, ensure_fnum,
                                   extract_axes_extents, figure, imshow, legend,
                                   multi_plot, next_fnum, pandas_plot_matrix,
                                   qtensure, render_figure_to_image,
                                   reverse_colormap, save_parts, savefig2,
                                   scores_to_cmap, scores_to_color, set_figtitle,
                                   show_if_requested,)
    from clab.util.nputil import (apply_grouping, argsubmax, argsubmaxima,
                                  atleast_nd, group_indices, group_items,
                                  isect_flags, iter_reduce_ufunc,)
    from clab.util.nxutil import (dump_nx_ondisk, make_agraph,
                                  nx_delete_None_edge_attr, nx_delete_node_attr,
                                  nx_ensure_agraph_color, nx_sink_nodes,
                                  nx_source_nodes, patch_pygraphviz,)
    from clab.util.priority_queue import (PriorityQueue, SortedQueue,)
    from clab.util.profiler import (IS_PROFILING, KernprofParser,
                                    dump_global_profile_report, dynamic_profile,
                                    find_parent_class, find_pattern_above_row,
                                    find_pyclass_above_row, profile,
                                    profile_onthefly,)
    from clab.util.util_affine import (TRANSFORM_DTYPE, affine_around_mat3x3,
                                       affine_mat3x3, rotation_around_bbox_mat3x3,
                                       rotation_around_mat3x3, rotation_mat2x2,
                                       rotation_mat3x3, scale_around_mat3x3,
                                       scale_mat3x3, shear_mat3x3,
                                       transform_around, translation_mat3x3,)
    from clab.util.util_alg import (mincost_assignment,)
    from clab.util.util_averages import (CumMovingAve, ExpMovingAve,
                                         InternalRunningStats, MovingAve,
                                         RunningStats, WindowedMovingAve, absdev,)
    # </AUTOGEN_INIT>
