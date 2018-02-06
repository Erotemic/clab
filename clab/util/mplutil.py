from __future__ import absolute_import, division, print_function
import cv2
import numpy as np
import six
import matplotlib as mpl
import ubelt as ub
from six.moves import zip_longest
from os.path import basename, splitext, join, dirname


def figure(fnum=None, pnum=(1, 1, 1), title=None, figtitle=None, doclf=False,
           docla=False, projection=None, **kwargs):
    """
    http://matplotlib.org/users/gridspec.html

    Args:
        fnum (int): fignum = figure number
        pnum (int, str, or tuple(int, int, int)): plotnum = plot tuple
        title (str):  (default = None)
        figtitle (None): (default = None)
        docla (bool): (default = False)
        doclf (bool): (default = False)

    Returns:
        mpl.Figure: fig

    CommandLine:
        python -m clab.util.mplutil figure:0 --show

    Example:
        >>> import matplotlib.pyplot as plt
        >>> fnum = 1
        >>> fig = figure(fnum, (2, 2, 1))
        >>> plt.gca().text(0.5, 0.5, "ax1", va="center", ha="center")
        >>> fig = figure(fnum, (2, 2, 2))
        >>> plt.gca().text(0.5, 0.5, "ax2", va="center", ha="center")
        >>> show_if_requested()

    Example:
        >>> import matplotlib.pyplot as plt
        >>> fnum = 1
        >>> fig = figure(fnum, (2, 2, 1))
        >>> plt.gca().text(0.5, 0.5, "ax1", va="center", ha="center")
        >>> fig = figure(fnum, (2, 2, 2))
        >>> plt.gca().text(0.5, 0.5, "ax2", va="center", ha="center")
        >>> fig = figure(fnum, (2, 4, (1, slice(1, None))))
        >>> plt.gca().text(0.5, 0.5, "ax3", va="center", ha="center")
        >>> show_if_requested()
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    def ensure_fig(fnum=None):
        if fnum is None:
            try:
                fig = plt.gcf()
            except Exception as ex:
                fig = plt.figure()
        else:
            try:
                fig = plt.figure(fnum)
            except Exception as ex:
                fig = plt.gcf()
        return fig

    def _convert_pnum_int_to_tup(int_pnum):
        # Convert pnum to tuple format if in integer format
        nr = int_pnum // 100
        nc = int_pnum // 10 - (nr * 10)
        px = int_pnum - (nr * 100) - (nc * 10)
        pnum = (nr, nc, px)
        return pnum

    def _pnum_to_subspec(pnum):
        if isinstance(pnum, six.string_types):
            pnum = list(pnum)
        nrow, ncols, plotnum = pnum
        # if kwargs.get('use_gridspec', True):
        # Convert old pnums to gridspec
        gs = gridspec.GridSpec(nrow, ncols)
        if isinstance(plotnum, (tuple, slice, list)):
            subspec = gs[plotnum]
        else:
            subspec = gs[plotnum - 1]
        return (subspec,)

    def _setup_subfigure(pnum):
        if isinstance(pnum, int):
            pnum = _convert_pnum_int_to_tup(pnum)
        if doclf:
            fig.clf()
        axes_list = fig.get_axes()
        if docla or len(axes_list) == 0:
            if pnum is not None:
                assert pnum[0] > 0, 'nRows must be > 0: pnum=%r' % (pnum,)
                assert pnum[1] > 0, 'nCols must be > 0: pnum=%r' % (pnum,)
                subspec = _pnum_to_subspec(pnum)
                ax = fig.add_subplot(*subspec, projection=projection)
                if len(axes_list) > 0:
                    ax.cla()
            else:
                ax = plt.gca()
        else:
            if pnum is not None:
                subspec = _pnum_to_subspec(pnum)
                ax = plt.subplot(*subspec)
            else:
                ax = plt.gca()

    fig = ensure_fig(fnum)
    if pnum is not None:
        _setup_subfigure(pnum)
    # Set the title / figtitle
    if title is not None:
        ax = plt.gca()
        ax.set_title(title)
    if figtitle is not None:
        fig.suptitle(figtitle)
    return fig


def multi_plot(xdata=None, ydata_list=[], **kwargs):
    r"""
    plots multiple lines, bars, etc...

    This is the big function that implements almost all of the heavy lifting in
    this file.  Any function not using this should probably find a way to use
    it. It is pretty general and relatively clean.

    Args:
        xdata (ndarray): can also be a list of arrays
        ydata_list (list of ndarrays): can also be a single array

    Kwargs:
        Misc:
            fnum, pnum, use_legend, legend_loc
        Labels:
            xlabel, ylabel, title, figtitle
            ticksize, titlesize, legendsize, labelsize
        Grid:
            gridlinewidth, gridlinestyle
        Ticks:
            num_xticks, num_yticks, tickwidth, ticklength, ticksize
        Data:
            xmin, xmax, ymin, ymax, spread_list
            # can append _list to any of these
            plot_kw_keys = ['label', 'color', 'marker', 'markersize',
                'markeredgewidth', 'linewidth', 'linestyle']
            kind = ['bar', 'plot', ...]
        if kind='plot':
            spread
        if kind='bar':
            stacked, width

    References:
        matplotlib.org/examples/api/barchart_demo.html

    CommandLine:
        python -m clab.util.mplutil multi_plot:0 --show
        python -m clab.util.mplutil multi_plot:1 --show

    Example:
        >>> xdata = [1, 2, 3, 4, 5]
        >>> ydata_list = [[1, 2, 3, 4, 5], [3, 3, 3, 3, 3], [5, 4, np.nan, 2, 1], [4, 3, np.nan, 1, 0]]
        >>> kwargs = {'label_list': ['spamΣ', 'eggs', 'jamµ', 'pram'],  'linestyle': '-'}
        >>> #fig = multi_plot(xdata, ydata_list, title='$\phi_1(\\vec{x})$', xlabel='\nfds', **kwargs)
        >>> fig = multi_plot(xdata, ydata_list, title='ΣΣΣµµµ', xlabel='\nfdsΣΣΣµµµ', **kwargs)
        >>> show_if_requested()

    Example:
        >>> fig1 = multi_plot([1, 2, 3], [4, 5, 6])
        >>> fig2 = multi_plot([1, 2, 3], [4, 5, 6], fnum=4)
        >>> show_if_requested()
    """
    from matplotlib import pyplot as plt

    if isinstance(ydata_list, dict):
        # Special case where ydata is a dictionary
        if isinstance(xdata, six.string_types):
            # Special-er case where xdata is specified in ydata
            xkey = xdata
            ykeys = set(ydata_list.keys()) - {xkey}
            xdata = ydata_list[xkey]
        else:
            ykeys = list(ydata_list.keys())
        # Normalize input
        ydata_list = list(ub.take(ydata_list, ykeys))
        kwargs['label_list'] = kwargs.get('label_list', ykeys)

    def is_listlike(data):
        flag = isinstance(data, (list, np.ndarray, tuple))
        flag &= hasattr(data, '__getitem__') and hasattr(data, '__len__')
        return flag

    def is_list_of_scalars(data):
        if is_listlike(data):
            if len(data) > 0 and not is_listlike(data[0]):
                return True
        return False

    def is_list_of_lists(data):
        if is_listlike(data):
            if len(data) > 0 and is_listlike(data[0]):
                return True
        return False

    # allow ydata_list to be passed without a container
    if is_list_of_scalars(ydata_list):
        ydata_list = [np.array(ydata_list)]

    if xdata is None:
        xdata = list(range(len(ydata_list[0])))

    num_lines = len(ydata_list)

    # Transform xdata into xdata_list
    if is_list_of_lists(xdata):
        xdata_list = [np.array(xd, copy=True) for xd in xdata]
    else:
        xdata_list = [np.array(xdata, copy=True)] * num_lines

    fnum = ensure_fnum(kwargs.get('fnum', None))
    pnum = kwargs.get('pnum', None)
    kind = kwargs.get('kind', 'plot')
    transpose = kwargs.get('transpose', False)

    def parsekw_list(key, kwargs, num_lines=num_lines):
        """ copies relevant plot commands into plot_list_kw """
        if key in kwargs:
            val_list = [kwargs[key]] * num_lines
        elif key + '_list' in kwargs:
            val_list = kwargs[key + '_list']
        elif key + 's' in kwargs:
            # hack, multiple ways to do something
            val_list = kwargs[key + 's']
        else:
            val_list = None
            #val_list = [None] * num_lines
        return val_list

    # Parse out arguments to ax.plot
    plot_kw_keys = ['label', 'color', 'marker', 'markersize',
                    'markeredgewidth', 'linewidth', 'linestyle', 'alpha']
    # hackish / extra args that dont go to plot, but help
    extra_plot_kw_keys = ['spread_alpha', 'autolabel', 'edgecolor', 'fill']
    plot_kw_keys += extra_plot_kw_keys
    plot_ks_vals = [parsekw_list(key, kwargs) for key in plot_kw_keys]
    plot_list_kw = dict([
        (key, vals)
        for key, vals in zip(plot_kw_keys, plot_ks_vals) if vals is not None
    ])

    if 'color' not in plot_list_kw:
        plot_list_kw['color'] = distinct_colors(num_lines)

    if kind == 'plot':
        if 'marker' not in plot_list_kw:
            plot_list_kw['marker'] = distinct_markers(num_lines)
        if 'spread_alpha' not in plot_list_kw:
            plot_list_kw['spread_alpha'] = [.2] * num_lines

    if kind == 'bar':
        # Remove non-bar kwargs
        for key in ['markeredgewidth', 'linewidth', 'marker', 'markersize', 'linestyle']:
            plot_list_kw.pop(key, None)

        stacked = kwargs.get('stacked', False)
        width_key = 'height' if transpose else 'width'
        if 'width_list' in kwargs:
            plot_list_kw[width_key] = kwargs['width_list']
        else:
            width = kwargs.get('width', .9)
            # if width is None:
            #     # HACK: need variable width
            #     # width = np.mean(np.diff(xdata_list[0]))
            #     width = .9
            if not stacked:
                width /= num_lines
            #plot_list_kw['orientation'] = ['horizontal'] * num_lines
            plot_list_kw[width_key] = [width] * num_lines

    spread_list = kwargs.get('spread_list', None)
    if spread_list is None:
        pass

    # nest into a list of dicts for each line in the multiplot
    valid_keys = list(set(plot_list_kw.keys()) - set(extra_plot_kw_keys))
    valid_vals = list(ub.dict_take(plot_list_kw, valid_keys))
    plot_kw_list = [dict(zip(valid_keys, vals)) for vals in zip(*valid_vals)]

    extra_kw_keys = [key for key in extra_plot_kw_keys if key in plot_list_kw]
    extra_kw_vals = list(ub.dict_take(plot_list_kw, extra_kw_keys))
    extra_kw_list = [dict(zip(extra_kw_keys, vals)) for vals in zip(*extra_kw_vals)]

    # Get passed in axes or setup a new figure
    ax = kwargs.get('ax', None)
    if ax is None:
        fig = figure(fnum=fnum, pnum=pnum, docla=False)
        ax = plt.gca()
    else:
        plt.sca(ax)
        fig = ax.figure

    # +---------------
    # Draw plot lines
    ydata_list = np.array(ydata_list)

    if transpose:
        if kind == 'bar':
            plot_func = ax.barh
        elif kind == 'plot':
            def plot_func(_x, _y, **kw):
                return ax.plot(_y, _x, **kw)
    else:
        plot_func = getattr(ax, kind)  # usually ax.plot

    assert len(ydata_list) > 0, 'no ydata'
    #assert len(extra_kw_list) == len(plot_kw_list), 'bad length'
    #assert len(extra_kw_list) == len(ydata_list), 'bad length'
    _iter = enumerate(zip_longest(xdata_list, ydata_list, plot_kw_list, extra_kw_list))
    for count, (_xdata, _ydata, plot_kw, extra_kw) in _iter:
        ymask = np.isfinite(_ydata)
        ydata_ = _ydata.compress(ymask)
        xdata_ = _xdata.compress(ymask)
        if kind == 'bar':
            if stacked:
                # Plot bars on top of each other
                xdata_ = xdata_
            else:
                # Plot bars side by side
                baseoffset = (width * num_lines) / 2
                lineoffset = (width * count)
                offset = baseoffset - lineoffset  # Fixeme for more histogram bars
                xdata_ = xdata_ - offset
            # width_key = 'height' if transpose else 'width'
            # plot_kw[width_key] = np.diff(xdata)
        objs = plot_func(xdata_, ydata_, **plot_kw)

        if kind == 'bar':
            if extra_kw is not None and 'edgecolor' in extra_kw:
                for rect in objs:
                    rect.set_edgecolor(extra_kw['edgecolor'])
            if extra_kw is not None and extra_kw.get('autolabel', False):
                # FIXME: probably a more cannonical way to include bar
                # autolabeling with tranpose support, but this is a hack that
                # works for now
                for rect in objs:
                    if transpose:
                        numlbl = width = rect.get_width()
                        xpos = width + ((_xdata.max() - _xdata.min()) * .005)
                        ypos = rect.get_y() + rect.get_height() / 2.
                        ha, va = 'left', 'center'
                    else:
                        numlbl = height = rect.get_height()
                        xpos = rect.get_x() + rect.get_width() / 2.
                        ypos = 1.05 * height
                        ha, va = 'center', 'bottom'
                    barlbl = '%.3f' % (numlbl,)
                    ax.text(xpos, ypos, barlbl, ha=ha, va=va)

        # print('extra_kw = %r' % (extra_kw,))
        if kind == 'plot' and extra_kw.get('fill', False):
            ax.fill_between(_xdata, ydata_, alpha=plot_kw.get('alpha', 1.0),
                            color=plot_kw.get('color', None))  # , zorder=0)

        if spread_list is not None:
            # Plots a spread around plot lines usually indicating standard
            # deviation
            _xdata = np.array(_xdata)
            spread = spread_list[count]
            ydata_ave = np.array(ydata_)
            y_data_dev = np.array(spread)
            y_data_max = ydata_ave + y_data_dev
            y_data_min = ydata_ave - y_data_dev
            ax = plt.gca()
            spread_alpha = extra_kw['spread_alpha']
            ax.fill_between(_xdata, y_data_min, y_data_max, alpha=spread_alpha,
                            color=plot_kw.get('color', None))  # , zorder=0)
    # L________________

    #max_y = max(np.max(y_data), max_y)
    #min_y = np.min(y_data) if min_y is None else min(np.min(y_data), min_y)

    ydata = _ydata  # HACK
    xdata = _xdata  # HACK
    if transpose:
        #xdata_list = ydata_list
        ydata = xdata
        # Hack / Fix any transpose issues
        def transpose_key(key):
            if key.startswith('x'):
                return 'y' + key[1:]
            elif key.startswith('y'):
                return 'x' + key[1:]
            elif key.startswith('num_x'):
                # hackier, fixme to use regex or something
                return 'num_y' + key[5:]
            elif key.startswith('num_y'):
                # hackier, fixme to use regex or something
                return 'num_x' + key[5:]
            else:
                return key
        kwargs = {transpose_key(key): val for key, val in kwargs.items()}

    # Setup axes labeling
    title      = kwargs.get('title', None)
    xlabel     = kwargs.get('xlabel', '')
    ylabel     = kwargs.get('ylabel', '')
    def none_or_unicode(text):
        return None if text is None else ub.ensure_unicode(text)

    xlabel = none_or_unicode(xlabel)
    ylabel = none_or_unicode(ylabel)
    title = none_or_unicode(title)

    # Initial integration with mpl rcParams standards
    mplrc = mpl.rcParams.copy()
    mplrc.update({
        # 'legend.fontsize': custom_figure.LEGEND_SIZE,
        # 'axes.titlesize': custom_figure.TITLE_SIZE,
        # 'axes.labelsize': custom_figure.LABEL_SIZE,
        # 'legend.facecolor': 'w',
        # 'font.family': 'sans-serif',
        # 'xtick.labelsize': custom_figure.TICK_SIZE,
        # 'ytick.labelsize': custom_figure.TICK_SIZE,
    })
    mplrc.update(kwargs.get('rcParams', {}))

    titlesize  = kwargs.get('titlesize',  mplrc['axes.titlesize'])
    labelsize  = kwargs.get('labelsize',  mplrc['axes.labelsize'])
    legendsize = kwargs.get('legendsize', mplrc['legend.fontsize'])
    xticksize = kwargs.get('ticksize', mplrc['xtick.labelsize'])
    yticksize = kwargs.get('ticksize', mplrc['ytick.labelsize'])
    family = kwargs.get('fontfamily', mplrc['font.family'])

    tickformat = kwargs.get('tickformat', None)
    ytickformat = kwargs.get('ytickformat', tickformat)
    xtickformat = kwargs.get('xtickformat', tickformat)

    # 'DejaVu Sans','Verdana', 'Arial'
    weight = kwargs.get('fontweight', None)
    if weight is None:
        weight = 'normal'

    labelkw = {
        'fontproperties': mpl.font_manager.FontProperties(
            weight=weight,
            family=family, size=labelsize)
    }
    ax.set_xlabel(xlabel, **labelkw)
    ax.set_ylabel(ylabel, **labelkw)

    tick_fontprop = mpl.font_manager.FontProperties(family=family,
                                                    weight=weight)

    if tick_fontprop is not None:
        for ticklabel in ax.get_xticklabels():
            ticklabel.set_fontproperties(tick_fontprop)
        for ticklabel in ax.get_yticklabels():
            ticklabel.set_fontproperties(tick_fontprop)
    if xticksize is not None:
        for ticklabel in ax.get_xticklabels():
            ticklabel.set_fontsize(xticksize)
    if yticksize is not None:
        for ticklabel in ax.get_yticklabels():
            ticklabel.set_fontsize(yticksize)

    if xtickformat is not None:
        # mpl.ticker.StrMethodFormatter  # newstyle
        # mpl.ticker.FormatStrFormatter  # oldstyle
        ax.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter(xtickformat))
    if ytickformat is not None:
        ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter(ytickformat))

    xtick_kw = ytick_kw = {
        'width': kwargs.get('tickwidth', None),
        'length': kwargs.get('ticklength', None),
    }
    xtick_kw = {k: v for k, v in xtick_kw.items() if v is not None}
    ytick_kw = {k: v for k, v in ytick_kw.items() if v is not None}
    ax.xaxis.set_tick_params(**xtick_kw)
    ax.yaxis.set_tick_params(**ytick_kw)

    #ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%d'))

    # Setup axes limits
    if 'xlim' in kwargs:
        xlim = kwargs['xlim']
        if xlim is not None:
            if 'xmin' not in kwargs and 'xmax' not in kwargs:
                kwargs['xmin'] = xlim[0]
                kwargs['xmax'] = xlim[1]
            else:
                raise ValueError('use xmax, xmin instead of xlim')
    if 'ylim' in kwargs:
        ylim = kwargs['ylim']
        if ylim is not None:
            if 'ymin' not in kwargs and 'ymay' not in kwargs:
                kwargs['ymin'] = ylim[0]
                kwargs['ymay'] = ylim[1]
            else:
                raise ValueError('use ymay, ymin instead of ylim')

    xmin = kwargs.get('xmin', ax.get_xlim()[0])
    xmax = kwargs.get('xmax', ax.get_xlim()[1])
    ymin = kwargs.get('ymin', ax.get_ylim()[0])
    ymax = kwargs.get('ymax', ax.get_ylim()[1])

    text_type = six.text_type

    if text_type(xmax) == 'data':
        xmax = max([xd.max() for xd in xdata_list])
    if text_type(xmin) == 'data':
        xmin = min([xd.min() for xd in xdata_list])

    # Setup axes ticks
    num_xticks = kwargs.get('num_xticks', None)
    num_yticks = kwargs.get('num_yticks', None)

    if num_xticks is not None:
        # TODO check if xdata is integral
        if xdata.dtype.kind == 'i':
            xticks = np.linspace(np.ceil(xmin), np.floor(xmax),
                                 num_xticks).astype(np.int32)
        else:
            xticks = np.linspace((xmin), (xmax), num_xticks)
        ax.set_xticks(xticks)
    if num_yticks is not None:
        if ydata.dtype.kind == 'i':
            yticks = np.linspace(np.ceil(ymin), np.floor(ymax),
                                 num_yticks).astype(np.int32)
        else:
            yticks = np.linspace((ymin), (ymax), num_yticks)
        ax.set_yticks(yticks)

    force_xticks = kwargs.get('force_xticks', None)
    if force_xticks is not None:
        xticks = np.array(sorted(ax.get_xticks().tolist() + force_xticks))
        ax.set_xticks(xticks)

    yticklabels = kwargs.get('yticklabels', None)
    if yticklabels is not None:
        # Hack ONLY WORKS WHEN TRANSPOSE = True
        # Overrides num_yticks
        ax.set_yticks(ydata)
        ax.set_yticklabels(yticklabels)

    xticklabels = kwargs.get('xticklabels', None)
    if xticklabels is not None:
        # Overrides num_xticks
        ax.set_xticks(xdata)
        ax.set_xticklabels(xticklabels)

    xtick_rotation = kwargs.get('xtick_rotation', None)
    if xtick_rotation is not None:
        [lbl.set_rotation(xtick_rotation)
         for lbl in ax.get_xticklabels()]
    ytick_rotation = kwargs.get('ytick_rotation', None)
    if ytick_rotation is not None:
        [lbl.set_rotation(ytick_rotation)
         for lbl in ax.get_yticklabels()]

    # Axis padding
    xpad = kwargs.get('xpad', None)
    ypad = kwargs.get('ypad', None)
    xpad_factor = kwargs.get('xpad_factor', None)
    ypad_factor = kwargs.get('ypad_factor', None)
    if xpad is None and xpad_factor is not None:
        xpad = (xmax - xmin) * xpad_factor
    if ypad is None and ypad_factor is not None:
        ypad = (ymax - ymin) * ypad_factor
    xpad = 0 if xpad is None else xpad
    ypad = 0 if ypad is None else ypad
    ypad_high = kwargs.get('ypad_high', ypad)
    ypad_low  = kwargs.get('ypad_low', ypad)
    xpad_high = kwargs.get('xpad_high', xpad)
    xpad_low  = kwargs.get('xpad_low', xpad)
    xmin, xmax = (xmin - xpad_low), (xmax + xpad_high)
    ymin, ymax = (ymin - ypad_low), (ymax + ypad_high)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    xscale          = kwargs.get('xscale', None)
    yscale          = kwargs.get('yscale', None)
    if yscale is not None:
        ax.set_yscale(yscale)
    if xscale is not None:
        ax.set_xscale(xscale)

    gridlinestyle = kwargs.get('gridlinestyle', None)
    gridlinewidth = kwargs.get('gridlinewidth', None)
    gridlines = ax.get_xgridlines() + ax.get_ygridlines()
    if gridlinestyle:
        for line in gridlines:
            line.set_linestyle(gridlinestyle)
    if gridlinewidth:
        for line in gridlines:
            line.set_linewidth(gridlinewidth)

    # Setup title
    if title is not None:
        titlekw = {
            'fontproperties': mpl.font_manager.FontProperties(
                family=family,
                weight=weight,
                size=titlesize)
        }
        ax.set_title(title, **titlekw)

    use_legend   = kwargs.get('use_legend', 'label' in valid_keys)
    legend_loc   = kwargs.get('legend_loc', 'best')
    legend_alpha = kwargs.get('legend_alpha', 1.0)
    if use_legend:
        legendkw = {
            'alpha': legend_alpha,
            'fontproperties': mpl.font_manager.FontProperties(
                family=family,
                weight=weight,
                size=legendsize)
        }
        legend(loc=legend_loc, ax=ax, **legendkw)

    figtitle = kwargs.get('figtitle', None)
    if figtitle is not None:
        set_figtitle(figtitle, fontfamily=family, fontweight=weight,
                     size=kwargs.get('figtitlesize'))

    use_darkbackground = kwargs.get('use_darkbackground', None)
    lightbg = kwargs.get('lightbg', None)
    if lightbg is None:
        lightbg = True
    if use_darkbackground is None:
        use_darkbackground = not lightbg
    if use_darkbackground:
        dark_background(force=use_darkbackground is True)
    # TODO: return better info
    return fig


def pandas_plot_matrix(df, rot=90, ax=None, grid=True, label=None,
                       zerodiag=False,
                       cmap='viridis', showvals=False, logscale=True):
    import matplotlib as mpl
    import copy
    from matplotlib import pyplot as plt
    if ax is None:
        fig = figure(fnum=1, pnum=(1, 1, 1))
        fig.clear()
        ax = plt.gca()
    ax = plt.gca()
    values = df.values
    if zerodiag:
        values = values.copy()
        values = values - np.diag(np.diag(values))

    # aximg = ax.imshow(values, interpolation='none', cmap='viridis')
    if logscale:
        from matplotlib.colors import LogNorm
        vmin = df[df > 0].min().min()
        norm = LogNorm(vmin=vmin, vmax=values.max())
    else:
        norm = None

    cmap = copy.copy(mpl.cm.get_cmap(cmap))  # copy the default cmap
    cmap.set_bad((0, 0, 0))

    aximg = ax.matshow(values, interpolation='none', cmap=cmap, norm=norm)
    # aximg = ax.imshow(values, interpolation='none', cmap='viridis', norm=norm)

    # ax.imshow(values, interpolation='none', cmap='viridis')
    ax.grid(False)
    cax = plt.colorbar(aximg, ax=ax)
    if label is not None:
        cax.set_label(label)

    ax.set_xticks(list(range(len(df.index))))
    ax.set_xticklabels([lbl[0:100] for lbl in df.index])
    for lbl in ax.get_xticklabels():
        lbl.set_rotation(rot)
    for lbl in ax.get_xticklabels():
        lbl.set_horizontalalignment('center')

    ax.set_yticks(list(range(len(df.columns))))
    ax.set_yticklabels([lbl[0:100] for lbl in df.columns])
    for lbl in ax.get_yticklabels():
        lbl.set_horizontalalignment('right')
    for lbl in ax.get_yticklabels():
        lbl.set_verticalalignment('center')

    # Grid lines around the pixels
    if grid:
        offset = -.5
        xlim = [-.5, len(df.columns)]
        ylim = [-.5, len(df.index)]
        segments = []
        for x in range(ylim[1]):
            xdata = [x + offset, x + offset]
            ydata = ylim
            segment = list(zip(xdata, ydata))
            segments.append(segment)
        for y in range(xlim[1]):
            xdata = xlim
            ydata = [y + offset, y + offset]
            segment = list(zip(xdata, ydata))
            segments.append(segment)
        bingrid = mpl.collections.LineCollection(segments, color='w', linewidths=1)
        ax.add_collection(bingrid)

    if showvals:
        x_basis = np.arange(len(df.columns))
        y_basis = np.arange(len(df.index))
        x, y = np.meshgrid(x_basis, y_basis)

        for c, r in zip(x.flatten(), y.flatten()):
            val = df.iloc[r, c]
            ax.text(c, r, val, va='center', ha='center', color='white')
    return ax


def axes_extent(axs, pad=0.0):
    """
    Get the full extent of a group of axes, including axes labels, tick labels,
    and titles.
    """
    import itertools as it
    import matplotlib as mpl
    def axes_parts(ax):
        yield ax
        for label in ax.get_xticklabels():
            if label.get_text():
                yield label
        for label in ax.get_yticklabels():
            if label.get_text():
                yield label
        xlabel = ax.get_xaxis().get_label()
        ylabel = ax.get_yaxis().get_label()
        for label in (xlabel, ylabel, ax.title):
            if label.get_text():
                yield label

    items = it.chain.from_iterable(axes_parts(ax) for ax in axs)
    extents = [item.get_window_extent() for item in items]
    #mpl.transforms.Affine2D().scale(1.1)
    extent = mpl.transforms.Bbox.union(extents)
    extent = extent.expanded(1.0 + pad, 1.0 + pad)
    return extent


def extract_axes_extents(fig, combine=False, pad=0.0):
    # Make sure we draw the axes first so we can
    # extract positions from the text objects
    import matplotlib as mpl
    fig.canvas.draw()

    # Group axes that belong together
    atomic_axes = []
    seen_ = set([])
    for ax in fig.axes:
        if ax not in seen_:
            atomic_axes.append([ax])
            seen_.add(ax)

    dpi_scale_trans_inv = fig.dpi_scale_trans.inverted()
    axes_bboxes_ = [axes_extent(axs, pad) for axs in atomic_axes]
    axes_extents_ = [extent.transformed(dpi_scale_trans_inv) for extent in axes_bboxes_]
    # axes_extents_ = axes_bboxes_
    if combine:
        # Grab include extents of figure text as well
        # FIXME: This might break on OSX
        # http://stackoverflow.com/questions/22667224/bbox-backend
        renderer = fig.canvas.get_renderer()
        for mpl_text in fig.texts:
            bbox = mpl_text.get_window_extent(renderer=renderer)
            extent_ = bbox.expanded(1.0 + pad, 1.0 + pad)
            extent = extent_.transformed(dpi_scale_trans_inv)
            # extent = extent_
            axes_extents_.append(extent)
        axes_extents = mpl.transforms.Bbox.union(axes_extents_)
    else:
        axes_extents = axes_extents_
    # if True:
    #     axes_extents.x0 = 0
    #     # axes_extents.y1 = 0
    return axes_extents


def adjust_subplots(left=None, right=None, bottom=None, top=None, wspace=None,
                    hspace=None, fig=None):
    """
    Kwargs:
        left (float): left side of the subplots of the figure
        right (float): right side of the subplots of the figure
        bottom (float): bottom of the subplots of the figure
        top (float): top of the subplots of the figure
        wspace (float): width reserved for blank space between subplots
        hspace (float): height reserved for blank space between subplots
    """
    from matplotlib import pyplot as plt
    kwargs = dict(left=left, right=right, bottom=bottom, top=top,
                  wspace=wspace, hspace=hspace)
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    if fig is None:
        fig = plt.gcf()
    subplotpars = fig.subplotpars
    adjust_dict = subplotpars.__dict__.copy()
    del adjust_dict['validate']
    adjust_dict.update(kwargs)
    fig.subplots_adjust(**adjust_dict)


def render_figure_to_image(fig, **savekw):
    import io
    import cv2
    import matplotlib as mpl
    axes_extents = extract_axes_extents(fig)
    extent = mpl.transforms.Bbox.union(axes_extents)
    with io.BytesIO() as stream:
        # This call takes 23% - 15% of the time depending on settings
        fig.savefig(stream, bbox_inches=extent, **savekw)
        # fig.savefig(stream, **savekw)
        stream.seek(0)
        data = np.fromstring(stream.getvalue(), dtype=np.uint8)
    im_bgra = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    return im_bgra


def savefig2(fig, fpath, **kwargs):
    """
    Does a tight layout and saves the figure with transparency
    """
    import matplotlib as mpl
    if 'transparent' not in kwargs:
        kwargs['transparent'] = True
    if 'extent' not in kwargs:
        axes_extents = extract_axes_extents(fig)
        extent = mpl.transforms.Bbox.union(axes_extents)
        kwargs['extent'] = extent
    fig.savefig(fpath, **kwargs)


def copy_figure_to_clipboard(fig):
    """
    References:
        https://stackoverflow.com/questions/17676373/python-matplotlib-pyqt-copy-image-to-clipboard
    """
    print('Copying figure %d to the clipboard' % fig.number)
    import matplotlib as mpl
    app = mpl.backends.backend_qt5.qApp
    QtGui = mpl.backends.backend_qt5.QtGui
    im_bgra = render_figure_to_image(fig, transparent=True)
    im_rgba = cv2.cvtColor(im_bgra, cv2.COLOR_BGRA2RGBA)
    im = im_rgba
    QImage = QtGui.QImage
    qim = QImage(im.data, im.shape[1], im.shape[0], im.strides[0], QImage.Format_RGBA8888)
    clipboard = app.clipboard()
    clipboard.setImage(qim)

    # size = fig.canvas.size()
    # width, height = size.width(), size.height()
    # qim = QtGui.QImage(fig.canvas.buffer_rgba(), width, height, QtGui.QImage.Format_ARGB32)

    # QtWidgets = mpl.backends.backend_qt5.QtWidgets
    # pixmap = QtWidgets.QWidget.grab(fig.canvas)
    # clipboard.setPixmap(pixmap)


def dict_intersection(dict1, dict2):
    r"""
    Args:
        dict1 (dict):
        dict2 (dict):

    Returns:
        dict: mergedict_

    CommandLine:
        python -m utool.util_dict --exec-dict_intersection

    Example:
        >>> # ENABLE_DOCTEST
        >>> dict1 = {'a': 1, 'b': 2, 'c': 3, 'd': 4}
        >>> dict2 = {'b': 2, 'c': 3, 'd': 5, 'e': 21, 'f': 42}
        >>> mergedict_ = dict_intersection(dict1, dict2)
        >>> print(ub.repr2(mergedict_, nl=0))
        {'b': 2, 'c': 3}
    """
    isect_keys = set(dict1.keys()).intersection(set(dict2.keys()))
    # maintain order if possible
    if isinstance(dict1, ub.odict):
        isect_keys_ = [k for k in dict1.keys() if k in isect_keys]
        _dict_cls = ub.odict
    else:
        isect_keys_ = isect_keys
        _dict_cls = dict
    dict_isect = _dict_cls(
        (k, dict1[k]) for k in isect_keys_ if dict1[k] == dict2[k]
    )
    return dict_isect


def dark_background(ax=None, doubleit=False, force=False):
    r"""
    Args:
        ax (None): (default = None)
        doubleit (bool): (default = False)

    CommandLine:
        python -m .draw_func2 --exec-dark_background --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> fig = figure()
        >>> dark_background()
        >>> show_if_requested()
    """
    from matplotlib import pyplot as plt

    def is_using_style(style):
        style_dict = mpl.style.library[style]
        return len(dict_intersection(style_dict, mpl.rcParams)) == len(style_dict)

    if force:
        from mpl_toolkits.mplot3d import Axes3D
        BLACK = np.array((  0,   0,   0, 255)) / 255.0
        # Should use mpl style dark background instead
        bgcolor = BLACK * .9
        if ax is None:
            ax = plt.gca()
        if isinstance(ax, Axes3D):
            ax.set_axis_bgcolor(bgcolor)
            ax.tick_params(colors='white')
            return
        xy, width, height = get_axis_xy_width_height(ax)
        if doubleit:
            halfw = (doubleit) * (width / 2)
            halfh = (doubleit) * (height / 2)
            xy = (xy[0] - halfw, xy[1] - halfh)
            width *= (doubleit + 1)
            height *= (doubleit + 1)
        rect = mpl.patches.Rectangle(xy, width, height, lw=0, zorder=0)
        rect.set_clip_on(True)
        rect.set_fill(True)
        rect.set_color(bgcolor)
        rect.set_zorder(-99999999999)
        rect = ax.add_patch(rect)


def get_axis_xy_width_height(ax=None, xaug=0, yaug=0, waug=0, haug=0):
    """ gets geometry of a subplot """
    from matplotlib import pyplot as plt
    if ax is None:
        ax = plt.gca()
    autoAxis = ax.axis()
    xy     = (autoAxis[0] + xaug, autoAxis[2] + yaug)
    width  = (autoAxis[1] - autoAxis[0]) + waug
    height = (autoAxis[3] - autoAxis[2]) + haug
    return xy, width, height


LEGEND_LOCATION = {
    'upper right':  1,
    'upper left':   2,
    'lower left':   3,
    'lower right':  4,
    'right':        5,
    'center left':  6,
    'center right': 7,
    'lower center': 8,
    'upper center': 9,
    'center':      10,
}


def set_figtitle(figtitle, subtitle='', forcefignum=True, incanvas=True,
                 size=None, fontfamily=None, fontweight=None,
                 fig=None):
    r"""
    Args:
        figtitle (?):
        subtitle (str): (default = '')
        forcefignum (bool): (default = True)
        incanvas (bool): (default = True)
        fontfamily (None): (default = None)
        fontweight (None): (default = None)
        size (None): (default = None)
        fig (None): (default = None)

    CommandLine:
        python -m .custom_figure set_figtitle --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> fig = figure(fnum=1, doclf=True)
        >>> result = set_figtitle(figtitle='figtitle', fig=fig)
        >>> # xdoc: +REQUIRES(--show)
        >>> show_if_requested()
    """
    from matplotlib import pyplot as plt
    if figtitle is None:
        figtitle = ''
    if fig is None:
        fig = plt.gcf()
    figtitle = ub.ensure_unicode(figtitle)
    subtitle = ub.ensure_unicode(subtitle)
    if incanvas:
        if subtitle != '':
            subtitle = '\n' + subtitle
        prop = {
            'family': fontfamily,
            'weight': fontweight,
            'size': size,
        }
        prop = {k: v for k, v in prop.items() if v is not None}
        sup = fig.suptitle(figtitle + subtitle)

        if prop:
            fontproperties = sup.get_fontproperties().copy()
            for key, val in prop.items():
                getattr(fontproperties, 'set_' + key)(val)
            sup.set_fontproperties(fontproperties)
            # fontproperties = mpl.font_manager.FontProperties(**prop)
    else:
        fig.suptitle('')
    # Set title in the window
    window_figtitle = ('fig(%d) ' % fig.number) + figtitle
    window_figtitle = window_figtitle.replace('\n', ' ')
    fig.canvas.set_window_title(window_figtitle)


def legend(loc='best', fontproperties=None, size=None, fc='w', alpha=1,
           ax=None, handles=None):
    r"""
    Args:
        loc (str): (default = 'best')
        fontproperties (None): (default = None)
        size (None): (default = None)

    Example:
        >>> # ENABLE_DOCTEST
        >>> loc = 'best'
        >>> xdata = np.linspace(-6, 6)
        >>> ydata = np.sin(xdata)
        >>> plt.plot(xdata, ydata, label='sin')
        >>> fontproperties = None
        >>> size = None
        >>> result = legend(loc, fontproperties, size)
        >>> print(result)
        >>> show_if_requested()
    """
    from matplotlib import pyplot as plt
    assert loc in LEGEND_LOCATION or loc == 'best', (
        'invalid loc. try one of %r' % (LEGEND_LOCATION,))
    if ax is None:
        ax = plt.gca()
    if fontproperties is None:
        prop = {}
        if size is not None:
            prop['size'] = size
        # prop['weight'] = 'normal'
        # prop['family'] = 'sans-serif'
    else:
        prop = fontproperties
    legendkw = dict(loc=loc)
    if prop:
        legendkw['prop'] = prop
    if handles is not None:
        legendkw['handles'] = handles
    legend = ax.legend(**legendkw)
    if legend:
        legend.get_frame().set_fc(fc)
        legend.get_frame().set_alpha(alpha)


def distinct_colors(N, brightness=.878, randomize=True, hue_range=(0.0, 1.0), cmap_seed=None):
    r"""
    Args:
        N (int):
        brightness (float):

    Returns:
        list: RGB_tuples

    CommandLine:
        python -m color_funcs --test-distinct_colors --N 2 --show --hue-range=0.05,.95
        python -m color_funcs --test-distinct_colors --N 3 --show --hue-range=0.05,.95
        python -m color_funcs --test-distinct_colors --N 4 --show --hue-range=0.05,.95
        python -m .color_funcs --test-distinct_colors --N 3 --show --no-randomize
        python -m .color_funcs --test-distinct_colors --N 4 --show --no-randomize
        python -m .color_funcs --test-distinct_colors --N 6 --show --no-randomize
        python -m .color_funcs --test-distinct_colors --N 20 --show

    References:
        http://blog.jianhuashao.com/2011/09/generate-n-distinct-colors.html

    CommandLine:
        python -m .color_funcs --exec-distinct_colors --show
        python -m .color_funcs --exec-distinct_colors --show --no-randomize --N 50
        python -m .color_funcs --exec-distinct_colors --show --cmap_seed=foobar

    Example:
        >>> # build test data
        >>> N = ub.smartcast(ut.get_argval('--N', default=2), int)
        >>> randomize = not ub.argflag('--no-randomize')
        >>> brightness = 0.878
        >>> # execute function
        >>> cmap_seed = ub.get_argval('--cmap_seed', default=None)
        >>> hue_range = ub.smartcast(ub.get_argval('--hue-range', default=(0.00, 1.0)), list)
        >>> RGB_tuples = distinct_colors(N, brightness, randomize, hue_range, cmap_seed=cmap_seed)
        >>> # verify results
        >>> assert len(RGB_tuples) == N
        >>> result = str(RGB_tuples)
        >>> print(result)
        >>> ut.quit_if_noshow()
        >>> color_list = RGB_tuples
        >>> testshow_colors(color_list)
        >>> show_if_requested()
    """
    # TODO: Add sin wave modulation to the sat and value
    # HACK for white figures
    from matplotlib import pyplot as plt
    import colorsys
    remove_yellow = True

    use_jet = False
    if use_jet:
        cmap = plt.cm.jet
        RGB_tuples = list(map(tuple, cmap(np.linspace(0, 1, N))))
    elif cmap_seed is not None:
        # Randomized map based on a seed
        #cmap_ = 'Set1'
        #cmap_ = 'Dark2'
        choices = [
            #'Set1', 'Dark2',
            'jet',
            #'gist_rainbow',
            #'rainbow',
            #'gnuplot',
            #'Accent'
        ]
        cmap_hack = ub.argval('--cmap-hack', default=None)
        ncolor_hack = ub.argval('--ncolor-hack', default=None)
        if cmap_hack is not None:
            choices = [cmap_hack]
        if ncolor_hack is not None:
            N = int(ncolor_hack)
            N_ = N
        seed = sum(list(map(ord, ub.hash_data(cmap_seed))))
        rng = np.random.RandomState(seed + 48930)
        cmap_str = rng.choice(choices, 1)[0]
        #print('cmap_str = %r' % (cmap_str,))
        cmap = plt.cm.get_cmap(cmap_str)
        #ut.hashstr27(cmap_seed)
        #cmap_seed = 0
        #pass
        jitter = (rng.randn(N) / (rng.randn(100).max() / 2)).clip(-1, 1) * ((1 / (N ** 2)))
        range_ = np.linspace(0, 1, N, endpoint=False)
        #print('range_ = %r' % (range_,))
        range_ = range_ + jitter
        #print('range_ = %r' % (range_,))
        while not (np.all(range_ >= 0) and np.all(range_ <= 1)):
            range_[range_ < 0] = np.abs(range_[range_ < 0] )
            range_[range_ > 1] = 2 - range_[range_ > 1]
        #print('range_ = %r' % (range_,))
        shift = rng.rand()
        range_ = (range_ + shift) % 1
        #print('jitter = %r' % (jitter,))
        #print('shift = %r' % (shift,))
        #print('range_ = %r' % (range_,))
        if ncolor_hack is not None:
            range_ = range_[0:N_]
        RGB_tuples = list(map(tuple, cmap(range_)))
    else:
        sat = brightness
        val = brightness
        hmin, hmax = hue_range
        if remove_yellow:
            hue_skips = [(.13, .24)]
        else:
            hue_skips = []
        hue_skip_ranges = [_[1] - _[0] for _ in hue_skips]
        total_skip = sum(hue_skip_ranges)
        hmax_ = hmax - total_skip
        hue_list = np.linspace(hmin, hmax_, N, endpoint=False, dtype=np.float)
        # Remove colors (like hard to see yellows) in specified ranges
        for skip, range_ in zip(hue_skips, hue_skip_ranges):
            hue_list = [hue if hue <= skip[0] else hue + range_ for hue in hue_list]
        HSV_tuples = [(hue, sat, val) for hue in hue_list]
        RGB_tuples = [colorsys.hsv_to_rgb(*x) for x in HSV_tuples]
    if randomize:
        deterministic_shuffle(RGB_tuples)
    return RGB_tuples


def distinct_markers(num, style='astrisk', total=None, offset=0):
    r"""
    Args:
        num (?):

    CommandLine:
        python -m .draw_func2 --exec-distinct_markers --show
        python -m .draw_func2 --exec-distinct_markers --style=star --show
        python -m .draw_func2 --exec-distinct_markers --style=polygon --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> style = ut.get_argval('--style', type_=str, default='astrisk')
        >>> marker_list = distinct_markers(10, style)
        >>> x_data = np.arange(0, 3)
        >>> for count, (marker) in enumerate(marker_list):
        >>>     plt.plot(x_data, [count] * len(x_data), marker=marker, markersize=10, linestyle='', label=str(marker))
        >>> legend()
        >>> show_if_requested()
    """
    num_sides = 3
    style_num = {
        'astrisk': 2,
        'star': 1,
        'polygon': 0,
        'circle': 3
    }[style]
    if total is None:
        total = num
    total_degrees = 360 / num_sides
    marker_list = [
        (num_sides, style_num,  total_degrees * (count + offset) / total)
        for count in range(num)
    ]
    return marker_list


def deterministic_shuffle(list_, rng=0):
    r"""
    Args:
        list_ (list):
        seed (int):

    Returns:
        list: list_

    Example:
        >>> list_ = [1, 2, 3, 4, 5, 6]
        >>> seed = 1
        >>> list_ = deterministic_shuffle(list_, seed)
        >>> result = str(list_)
        >>> print(result)
        [3, 2, 5, 1, 4, 6]
    """
    from clab import util
    rng = util.ensure_rng(rng)
    rng.shuffle(list_)
    return list_


BASE_FNUM = 9001


def next_fnum(new_base=None):
    global BASE_FNUM
    if new_base is not None:
        BASE_FNUM = new_base
    BASE_FNUM += 1
    return BASE_FNUM


def ensure_fnum(fnum):
    if fnum is None:
        return next_fnum()
    return fnum


def _save_requested(fpath_, save_parts):
    raise NotImplementedError('havent done this yet')
    # dpi = ut.get_argval('--dpi', type_=int, default=200)
    from os.path import expanduser
    from matplotlib import pyplot as plt
    dpi = 200
    fpath_ = expanduser(fpath_)
    print('Figure save was requested')
    # arg_dict = ut.get_arg_dict(prefix_list=['--', '-'],
    #                            type_hints={'t': list, 'a': list})
    arg_dict = {}
    # HACK
    arg_dict = {
        key: (val[0] if len(val) == 1 else '[' + ']['.join(val) + ']')
        if isinstance(val, list) else val
        for key, val in arg_dict.items()
    }
    fpath_ = fpath_.format(**arg_dict)
    fpath_ = fpath_.replace(' ', '').replace('\'', '').replace('"', '')
    dpath = ub.argval('--dpath', type_=str, default=None)
    if dpath is None:
        gotdpath = False
        dpath = '.'
    else:
        gotdpath = True

    fpath = join(dpath, fpath_)
    if not gotdpath:
        dpath = dirname(fpath_)
    print('dpath = %r' % (dpath,))

    fig = plt.gcf()
    fig.dpi = dpi

    fpath_strict = ub.truepath(fpath)
    CLIP_WHITE = ub.argflag('--clipwhite')

    if save_parts:
        import os
        # TODO: call save_parts instead, but we still need to do the
        # special grouping.

        # Group axes that belong together
        atomic_axes = []
        seen_ = set([])
        for ax in fig.axes:
            # div = pt.get_plotdat(ax, DF2_DIVIDER_KEY, None)
            # if div is not None:
            #     df2_div_axes = pt.get_plotdat_dict(ax).get('df2_div_axes', [])
            #     seen_.add(ax)
            #     seen_.update(set(df2_div_axes))
            #     atomic_axes.append([ax] + df2_div_axes)
            #     # TODO: pad these a bit
            # else:
            if ax not in seen_:
                atomic_axes.append([ax])
                seen_.add(ax)

        hack_axes_group_row = ub.argflag('--grouprows')
        if hack_axes_group_row:
            groupid_list = []
            for axs in atomic_axes:
                for ax in axs:
                    groupid = ax.colNum
                groupid_list.append(groupid)

            groups = ub.group_items(atomic_axes, groupid_list)
            new_groups = list(map(ub.flatten, groups.values()))
            atomic_axes = new_groups
            #[[(ax.rowNum, ax.colNum) for ax in axs] for axs in atomic_axes]
            # save all rows of each column

        subpath_list = save_parts(fig=fig, fpath=fpath_strict,
                                  grouped_axes=atomic_axes, dpi=dpi)
        absfpath_ = subpath_list[-1]
        fpath_list = [os.path.relpath(_, dpath) for _ in subpath_list]

        if CLIP_WHITE:
            for subpath in subpath_list:
                # remove white borders
                pass
                # vt.clipwhite_ondisk(subpath, subpath)
    else:
        savekw = {}
        # savekw['transparent'] = fpath.endswith('.png') and not noalpha
        savekw['transparent'] = ub.argflag('--alpha')
        savekw['dpi'] = dpi
        savekw['edgecolor'] = 'none'
        savekw['bbox_inches'] = extract_axes_extents(fig, combine=True)  # replaces need for clipwhite
        absfpath_ = ub.truepath(fpath)
        fig.savefig(absfpath_, **savekw)

        # if CLIP_WHITE:
        #     # remove white borders
        #     fpath_in = fpath_out = absfpath_
        #     vt.clipwhite_ondisk(fpath_in, fpath_out)

        fpath_list = [fpath_]

    # Print out latex info
    default_caption = '\n% ---\n' + basename(fpath).replace('_', ' ') + '\n% ---\n'
    default_label = splitext(basename(fpath))[0]  # [0].replace('_', '')
    caption_list = ub.argval('--caption', type_=str,
                             default=default_caption)
    if isinstance(caption_list, six.string_types):
        caption_str = caption_list
    else:
        caption_str = ' '.join(caption_list)
    #caption_str = ut.get_argval('--caption', type_=str,
    #default=basename(fpath).replace('_', ' '))
    label_str   = ub.get_argval('--label', default=default_label)
    width_str = ub.get_argval('--width', default=r'\textwidth')
    width_str = ub.get_argval('--width', default=r'\textwidth')
    print('width_str = %r' % (width_str,))
    height_str  = ub.argval('--height', type_=str, default=None)
    caplbl_str =  label_str

    RESHAPE = ub.argval('--reshape', type_=tuple, default=None)
    if RESHAPE:
        raise NotImplementedError('fixme')
        def list_reshape(list_, new_shape):
            for dim in reversed(new_shape):
                list_ = list(map(list, zip(*[list_[i::dim] for i in range(dim)])))
            return list_
        newshape = (2,)
        ut = None
        unflat_fpath_list = ut.list_reshape(fpath_list, newshape, trail=True)
        fpath_list = ut.flatten(ut.list_transpose(unflat_fpath_list))

    caption_str = '\caplbl{' + caplbl_str + '}' + caption_str
    figure_str  = ut.util_latex.get_latex_figure_str(fpath_list,
                                                     label_str=label_str,
                                                     caption_str=caption_str,
                                                     width_str=width_str,
                                                     height_str=height_str)
    #import sys
    #print(sys.argv)
    latex_block = figure_str
    latex_block = ut.latex_newcommand(label_str, latex_block)
    #latex_block = ut.codeblock(
    #    r'''
    #    \newcommand{\%s}{
    #    %s
    #    }
    #    '''
    #) % (label_str, latex_block,)
    try:
        import os
        import psutil
        import pipes
        #import shlex
        # TODO: separate into get_process_cmdline_str
        # TODO: replace home with ~
        proc = psutil.Process(pid=os.getpid())
        home = os.path.expanduser('~')
        cmdline_str = ' '.join([
            pipes.quote(_).replace(home, '~')
            for _ in proc.cmdline()])
        latex_block = ut.codeblock(
            r'''
            \begin{comment}
            %s
            \end{comment}
            '''
        ) % (cmdline_str,) + '\n' + latex_block
    except OSError:
        pass

    #latex_indent = ' ' * (4 * 2)
    latex_indent = ' ' * (0)

    latex_block_ = (ut.indent(latex_block, latex_indent))
    ut.print_code(latex_block_, 'latex')

    if 'append' in arg_dict:
        append_fpath = arg_dict['append']
        ut.write_to(append_fpath, '\n\n' + latex_block_, mode='a')

    if ub.argflag(('--diskshow', '--ds')):
        # show what we wrote
        ut.startfile(absfpath_)

    # Hack write the corresponding logfile next to the output
    log_fpath = ut.get_current_log_fpath()
    if ub.argflag('--savelog'):
        if log_fpath is not None:
            ut.copy(log_fpath, splitext(absfpath_)[0] + '.txt')
        else:
            print('Cannot copy log file because none exists')


def show_if_requested(N=1):
    """
    Used at the end of tests. Handles command line arguments for saving figures

    Referencse:
        http://stackoverflow.com/questions/4325733/save-a-subplot-in-matplotlib

    """
    import matplotlib.pyplot as plt
    # Process figures adjustments from command line before a show or a save

    # udpate_adjust_subplots()
    # if use_argv:
    #     # hack to take args from commandline
    #     adjust_dict = ut.parse_dict_from_argv(adjust_dict)
    # adjust_subplots(use_argv=True)

    # def update_figsize():
    #     """ updates figsize based on command line """
    #     figsize = ut.get_argval('--figsize', type_=list, default=None)
    #     if figsize is not None:
    #         # Enforce inches and DPI
    #         fig = plt.gcf()
    #         figsize = [eval(term) if isinstance(term, str) else term
    #                    for term in figsize]
    #         figw, figh = figsize[0], figsize[1]
    #         print('get_size_inches = %r' % (fig.get_size_inches(),))
    #         print('fig w,h (inches) = %r, %r' % (figw, figh))
    #         fig.set_size_inches(figw, figh)
    #         #print('get_size_inches = %r' % (fig.get_size_inches(),))
    # update_figsize()

    save_parts = ub.argflag('--saveparts')

    fpath_ = ub.argval('--save', default=None)
    if fpath_ is None:
        fpath_ = ub.argval('--saveparts', default=None)
        save_parts = True

    if fpath_ is not None:
        _save_requested(fpath_, save_parts)
    elif ub.argflag('--cmd'):
        pass
    elif ub.argflag('--show'):
        # if ub.argflag('--tile'):
        #     if ut.get_computer_name().lower() in ['hyrule']:
        #         fig_presenter.all_figures_tile(percent_w=.5, monitor_num=0)
        #     else:
        #         fig_presenter.all_figures_tile()
        # if ub.argflag('--present'):
        #     fig_presenter.present()
        # for fig in fig_presenter.get_all_figures():
        #     fig.set_dpi(80)
        plt.show()


def save_parts(fig, fpath, grouped_axes=None, dpi=None):
    """
    FIXME: this works in mpl 2.0.0, but not 2.0.2

    Args:
        fig (?):
        fpath (str):  file path string
        dpi (None): (default = None)

    Returns:
        list: subpaths

    CommandLine:
        python -m draw_func2 save_parts

    Example:
        >>> # DISABLE_DOCTEST
        >>> import matplotlib as mpl
        >>> import matplotlib.pyplot as plt
        >>> def testimg(fname):
        >>>     return plt.imread(mpl.cbook.get_sample_data(fname))
        >>> fnames = ['grace_hopper.png', 'ada.png'] * 4
        >>> fig = plt.figure(1)
        >>> for c, fname in enumerate(fnames, start=1):
        >>>     ax = fig.add_subplot(3, 4, c)
        >>>     ax.imshow(testimg(fname))
        >>>     ax.set_title(fname[0:3] + str(c))
        >>>     ax.set_xticks([])
        >>>     ax.set_yticks([])
        >>> ax = fig.add_subplot(3, 1, 3)
        >>> ax.plot(np.sin(np.linspace(0, np.pi * 2)))
        >>> ax.set_xlabel('xlabel')
        >>> ax.set_ylabel('ylabel')
        >>> ax.set_title('title')
        >>> fpath = 'test_save_parts.png'
        >>> adjust_subplots(fig=fig, wspace=.3, hspace=.3, top=.9)
        >>> subpaths = save_parts(fig, fpath, dpi=300)
        >>> fig.savefig(fpath)
        >>> ut.startfile(subpaths[0])
        >>> ut.startfile(fpath)
    """
    if dpi:
        # Need to set figure dpi before we draw
        fig.dpi = dpi
    # We need to draw the figure before calling get_window_extent
    # (or we can figure out how to set the renderer object)
    # if getattr(fig.canvas, 'renderer', None) is None:
    fig.canvas.draw()

    # Group axes that belong together
    if grouped_axes is None:
        grouped_axes = []
        for ax in fig.axes:
            grouped_axes.append([ax])

    subpaths = []
    _iter = enumerate(grouped_axes, start=0)
    _iter = ub.ProgIter(list(_iter), label='save subfig')
    for count, axs in _iter:
        subpath = ub.augpath(fpath, suffix=chr(count + 65))
        extent = axes_extent(axs).transformed(fig.dpi_scale_trans.inverted())
        savekw = {}
        savekw['transparent'] = ub.argflag('--alpha')
        if dpi is not None:
            savekw['dpi'] = dpi
        savekw['edgecolor'] = 'none'
        fig.savefig(subpath, bbox_inches=extent, **savekw)
        subpaths.append(subpath)
    return subpaths


def qtensure():
    try:
        __IPYTHON__
    except NameError:
        return False
    else:
        import sys
        import IPython
        ipython = IPython.get_ipython()
        if ipython is None:
            # we must have exited ipython at some point
            return
        if 'PyQt4' in sys.modules:
            #IPython.get_ipython().magic('pylab qt4')
            ipython.magic('pylab qt4 --no-import-all')
        else:
            # if gt.__PYQT__.GUITOOL_PYQT_VERSION == 5:
            ipython.magic('pylab qt5 --no-import-all')


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m clab.util.mplutil
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
