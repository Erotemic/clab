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


def make_heatmask(probs, cmap='plasma'):
    """
    Colorizes a single-channel intensity mask (with an alpha channel)
    """
    # import matplotlib as mpl
    # current_backend = mpl.get_backend()
    # for backend in ['Qt5Agg', 'Agg']:
    #     try:
    #         mpl.use(backend, warn=True, force=False)
    #         break
    #     except Exception:
    #         pass
    import matplotlib as mpl
    from clab.util import imutil
    assert len(probs.shape) == 2
    cmap_ = mpl.cm.get_cmap(cmap)
    probs = imutil.ensure_float01(probs)
    heatmask = cmap_(probs)
    heatmask[:, :, 0:3] = heatmask[:, :, 0:3][:, :, ::-1]
    heatmask[:, :, 3] = probs
    return heatmask


def colorbar_image(domain, cmap='plasma', dpi=96, shape=(200, 20)):
    """

    note that shape is approximate

    domain = np.linspace(-30, 200)
    cmap='plasma'
    dpi = 80
    dsize = (20, 200)

    """
    import matplotlib as mpl
    from clab.util import utildraw
    mpl.use('agg', force=False, warn=False)
    from matplotlib import pyplot as plt

    fig = plt.figure(dpi=dpi)

    w, h = shape[1] / dpi, shape[0] / dpi
    # w, h = 1, 10
    fig.set_size_inches(w, h)

    ax = fig.add_subplot('111')

    sm = plt.cm.ScalarMappable(cmap=plt.get_cmap(cmap))
    sm.set_array(domain)

    plt.colorbar(sm, cax=ax)

    cb = utildraw.render_figure_to_image(fig, dpi=dpi)

    plt.close(fig)

    return cb
    # from clab import util
    # util.imwrite('foo.png', img)
