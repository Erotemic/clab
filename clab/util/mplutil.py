from __future__ import absolute_import, division, print_function
import cv2
import numpy as np
import six


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
        python -m plottool.custom_figure --exec-figure:0 --show
        python -m plottool.custom_figure --exec-figure:1 --show

    Example:
        >>> fnum = 1
        >>> fig = figure(fnum, (2, 2, 1))
        >>> gca().text(0.5, 0.5, "ax1", va="center", ha="center")
        >>> fig = figure(fnum, (2, 2, 2))
        >>> gca().text(0.5, 0.5, "ax2", va="center", ha="center")
        >>> ut.show_if_requested()

    Example:
        >>> fnum = 1
        >>> fig = figure(fnum, (2, 2, 1))
        >>> gca().text(0.5, 0.5, "ax1", va="center", ha="center")
        >>> fig = figure(fnum, (2, 2, 2))
        >>> gca().text(0.5, 0.5, "ax2", va="center", ha="center")
        >>> fig = figure(fnum, (2, 4, (1, slice(1, None))))
        >>> gca().text(0.5, 0.5, "ax3", va="center", ha="center")
        >>> ut.show_if_requested()
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
