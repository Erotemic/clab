from __future__ import absolute_import, division, print_function
import numpy as np


def make_distinct_bgr01_colors(num):
    import matplotlib as mpl
    import matplotlib._cm  as _cm
    cm = mpl.colors.LinearSegmentedColormap.from_list(
        'gist_rainbow', _cm.datad['gist_rainbow'],
        mpl.rcParams['image.lut'])
    distinct_colors = [
        (np.array(cm(i / num)) * 255).astype(np.int).tolist()[0:3][::-1]
        for i in range(num)
    ]
    return distinct_colors


def convert_hex_to_255(hex_color):
    """
    hex_color = '#6A5AFFAF'
    """
    assert hex_color.startswith('#'), 'not a hex string %r' % (hex_color,)
    parts = hex_color[1:].strip()
    color255 = tuple(int(parts[i: i + 2], 16) for i in range(0, len(parts), 2))
    assert len(color255) in [3, 4], 'must be length 3 or 4'
    # # color = mcolors.hex2color(hex_color[0:7])
    # if len(hex_color) > 8:
    #     alpha_hex = hex_color[7:9]
    #     alpha_float = int(alpha_hex, 16) / 255.0
    #     color = color + (alpha_float,)
    return color255


def lookup_bgr255(key):
    from matplotlib import colors as mcolors
    return convert_hex_to_255(mcolors.CSS4_COLORS[key])[::-1]
