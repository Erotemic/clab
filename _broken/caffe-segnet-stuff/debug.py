from __future__ import absolute_import, division, print_function
from os.path import join
import pandas as pd
import cv2
import glob
import json
import itertools as it
import numpy as np
import ubelt as ub

from pysseg.util import utildraw

from pysseg import getLogger
logger = getLogger(__name__)
print = logger.info


def dump_polygon_overlap_analysis(task, scene_unique, scene_raw, total_isect):
    import matplotlib as mpl
    mpl.style.use('ggplot')
    from matplotlib import pyplot as plt

    total_raw = scene_raw.sum(axis=1)

    mpl.rcParams['ytick.labelsize'] = 20
    mpl.rcParams['xtick.labelsize'] = 20
    mpl.rcParams['axes.labelsize'] = 20
    mpl.rcParams['axes.titlesize'] = 20
    mpl.rcParams['legend.fontsize'] = 18
    mpl.rcParams['legend.loc'] = 'lower right'

    ub.ensuredir('plots')

    #### CLASS FREQUENCY (SCENES)
    sortx = total_raw.sort_values().index
    scene_raw = scene_raw.loc[sortx, :]
    ax = scene_raw.plot.barh(stacked=True, colormap='viridis', rot=0)
    ax.legend(*ax.get_legend_handles_labels(), loc='lower right', prop={'size': 20})
    ax.set_xlabel('area')
    ax.set_title('Scene Element Polygon Area (per scene)')
    ax.figure.set_size_inches(12, 8)
    utildraw.adjust_subplots(bottom=.1, left=.1, right=.95, top=.9)
    fig = ax.figure
    cv2.imwrite('plots/freq_scene.png', utildraw.render_figure_to_image(fig, transparent=True))

    train, test = task.load_predefined_train_test()

    #### CLASS FREQUENCY (XVAL)
    split_freq = pd.DataFrame({
        'train': scene_raw.loc[:, train].sum(axis=1),
        'test': scene_raw.loc[:, test].sum(axis=1),
    })
    sortx = scene_raw.sum(axis=1).sort_values().index
    split_freq = split_freq.loc[sortx, :]
    ax = split_freq.plot.barh(stacked=True, colormap='viridis', rot=0)
    ax.legend(*ax.get_legend_handles_labels(), loc='lower right', prop={'size': 20})
    ax.set_xlabel('area')
    ax.set_title('Scene Element Polygon Area (test/train)')
    ax.figure.set_size_inches(12, 8)
    utildraw.adjust_subplots(bottom=.1, left=.1, right=.95, top=.9)
    fig.canvas.draw()
    fig = ax.figure
    cv2.imwrite('plots/freq_xval.png', utildraw.render_figure_to_image(fig, transparent=True))

    #### CLASS INTERSECTION (matrix)
    df = total_isect
    sortx = df.sum(axis=0).sort_values().index[::-1]
    df = df.loc[sortx].T.loc[sortx]
    print(df.to_string(float_format='{:.2f}'.format))

    fig = plt.figure()
    fig.clear()
    ax = fig.add_subplot(111)
    ax = utildraw.pandas_plot_matrix(df, rot=80, ax=ax, label='intersecting area')
    ax.set_xlabel('Scene Element Polygon Overlap')
    ax.figure.set_size_inches(10, 8)
    utildraw.adjust_subplots(bottom=.1, left=.1, right=.9, top=.7)
    fig = ax.figure
    cv2.imwrite('plots/overlap_matrix.png', utildraw.render_figure_to_image(fig, transparent=True))

    #### CLASS INTERSECTION (bar)
    fig = plt.figure()
    flat_df = pd.Series(np.triu(df.values).ravel(),
                        index=list(it.product(df.index, df.columns)))
    flat_df = flat_df.sort_values(ascending=False)
    top_flat = flat_df.iloc[:10][::-1]
    ax = top_flat.plot.barh(colormap='viridis', rot=0)
    # ax.legend(patches, labels, loc='lower right', prop={'size': 20})
    utildraw.adjust_subplots(bottom=.1, left=.3, right=.95, top=.9)
    ax.figure.set_size_inches(12, 8)
    ax.set_xlabel('intersecting area')
    ax.set_title('Highest Overlap Class Pairs (Area)')
    fig = ax.figure
    cv2.imwrite('plots/overlap_pairs.png', utildraw.render_figure_to_image(fig, transparent=True))

    fig = plt.figure()
    denom = [min(*ub.take(total_raw.loc, p)) for p in flat_df.index]
    percent_intersect = flat_df / denom
    top_flat = percent_intersect.sort_values(ascending=False).iloc[:10][::-1]
    ax = top_flat.plot.barh(colormap='viridis', rot=0)
    utildraw.adjust_subplots(bottom=.1, left=.3, right=.95, top=.9)
    ax.figure.set_size_inches(12, 8)
    ax.set_xlabel('overlap(c1, c2) / min(c1.area, c2.area)')
    ax.set_title('Highest Overlap Class Pairs (Percent)')
    fig = ax.figure
    cv2.imwrite('plots/overlap_pairs_percent.png', utildraw.render_figure_to_image(fig, transparent=True))

    #### UNIQUE CLASS FREQUENCY (SCENES)
    fig = plt.figure()
    sortx = scene_unique.sum(axis=1).sort_values().index
    scene_unique = scene_unique.loc[sortx, :]
    ax = scene_unique.plot.barh(stacked=True, colormap='viridis', rot=0)
    ax.set_xlabel('area')
    ax.set_title('Unique Scene Element Area (per scene)')
    utildraw.adjust_subplots(bottom=.25, left=.25, right=.9, top=.9)
    ax.figure.set_size_inches(13, 10)
    fig = ax.figure
    cv2.imwrite('plots/unique_area_total.png', utildraw.render_figure_to_image(fig, transparent=True))

    fig = plt.figure()
    total_unique = scene_unique.sum(axis=1)
    unique = total_unique / total_raw
    sortx = unique.sort_values().index
    unique = unique.loc[sortx]
    ax = unique.plot.barh(stacked=True, colormap='viridis', rot=0)
    ax.set_xlabel('fraction')
    ax.set_title('Fraction of Unique Scene Elements')
    utildraw.adjust_subplots(bottom=.25, left=.25, right=.9, top=.9)
    ax.figure.set_size_inches(13, 10)
    fig = ax.figure
    cv2.imwrite('plots/unique_area_percent.png', utildraw.render_figure_to_image(fig, transparent=True))


class DebugFuncs(object):
    def preparse_classes(self, task):
        """
        CommandLine:
            python -m pysseg.tasks DebugFuncs.preparse_classes

        Example:
            >>> from pysseg.tasks import *
            >>> task = DivaV1()
            >>> ut.qtensure()
            >>> DebugFuncs().preparse_classes(task)
        """

        def scene_overlap(frame_elems, labels):
            from shapely.geometry import Polygon
            frame_polys = {frameid: [(k, Polygon(v)) for k, v in elems]
                           for frameid, elems in frame_elems.items()}

            isect = pd.DataFrame(0, columns=labels, index=labels)
            diag = pd.Series(0, index=labels)
            unique = pd.Series(0, index=labels)

            for frameid, polys in frame_polys.items():
                # Union all polys of the same class
                class_polys = {}
                polys = sorted(polys, key=lambda x: x[0])
                for k, poly in polys:
                    if k not in class_polys:
                        class_polys[k] = poly.buffer(0)
                    else:
                        class_polys[k] = poly.buffer(0).union(class_polys[k])

                # How much area does each class have?
                for k1 in class_polys.keys():
                    poly1 = class_polys[k1].buffer(0)
                    # isect.loc[(k1, k1)] = poly1.area
                    diag.loc[k1] += poly1.area

                # How much intersecting area does each class have?
                for k1, k2 in it.combinations(class_polys.keys(), 2):
                    poly1 = class_polys[k1].buffer(0)
                    poly2 = class_polys[k2].buffer(0)
                    poly_isect = poly1.intersection(poly2)
                    isect.loc[(k1, k2)] += poly_isect.area

                # How much unique area does each class have?
                for k1 in class_polys.keys():
                    poly1 = class_polys[k1].buffer(0)
                    for k2 in class_polys.keys():
                        if k1 != k2:
                            poly2 = class_polys[k2].buffer(0)
                            poly1 = poly1.difference(poly2).buffer(0)
                    unique.loc[k1] += poly1.area
            return diag, isect, unique

        labels = set()
        for scene in task.scene_ids:
            scene_path = join(task.scene_base, scene, 'static')
            scene_json_fpath = join(scene_path, 'static.json')
            with open(scene_json_fpath) as data_file:
                data = json.load(data_file)
            tracks = data['tracks']
            for track in tracks:
                if 'label' in track:
                    classname = track['label']
                    labels.add(classname)
        print('labels = {!r}'.format(labels))

        labels = sorted(labels)
        scene_raw = pd.DataFrame(0, columns=task.scene_ids, index=labels)
        scene_unique = pd.DataFrame(0, columns=task.scene_ids, index=labels)
        total_isect = pd.DataFrame(0, columns=labels, index=labels)
        for scene in task.scene_ids:
            scene_path = join(task.scene_base, scene, 'static')
            scene_json_fpath = join(scene_path, 'static.json')
            frame_elems = task.parse_scene_elements(scene_json_fpath)
            # Determine scene element intersection area
            diag, isect, unique = scene_overlap(frame_elems, labels)
            total_isect += isect
            scene_raw.loc[:, scene] = diag
            scene_unique.loc[:, scene] = unique

        total_isect = (total_isect + total_isect.T) / 2
        print(total_isect.to_string(float_format='{:.2f}'.format))

        m = total_isect.values
        o = m - np.diag(np.diag(m))
        od = pd.DataFrame(o, columns=labels, index=labels)
        sortx = od.sum(axis=0).sort_values().index[::-1]
        od = od.loc[sortx].T.loc[sortx]
        print(od.to_string(float_format='{:.2f}'.format))

        dump_polygon_overlap_analysis(task, scene_unique, scene_raw, total_isect)

    @staticmethod
    def full_freq(task):
        labels = set()
        train_gt_paths = []
        for scene in task.scene_ids:
            scene_gtfull_dpath = task.datasubdir('gt' + 'full', scene)
            for path in glob.glob(join(scene_gtfull_dpath, '*.png')):
                train_gt_paths.append(path)

        labels = np.arange(len(task.classnames))
        # bins are edges
        bins = np.arange(len(task.classnames) + 1)
        class_freq = np.zeros(len(labels))
        class_freq = pd.Series(class_freq, index=task.classnames)

        for path in ub.ProgIter(train_gt_paths, label='computing weights'):
            y_true = cv2.imread(path, flags=0).ravel()
            freq, _ = np.histogram(y_true, bins=bins)
            class_freq += freq
        print('class_freq = {!r}'.format(class_freq))


def viz_overlay_layers(task):
    """
    >>> from pysseg.tasks import *
    >>> task = DivaV1(clean=0)
    """
    for scene in ub.ProgIter(task.scene_ids, label='scene', verbose=3):
        scene_path = join(task.scene_base, scene, 'static')
        frame_image_fpaths = sorted(glob.glob(join(scene_path, '*.png')))
        scene_json_fpath = join(scene_path, 'static.json')

        frame_to_class_coords = task.parse_scene_elements(scene_json_fpath)
        from pysseg.util import imutil

        def new_layer(shape, classname, poly_coords):
            coords = np.round(np.array([poly_coords])).astype(np.int)
            alpha = int(.5 * 255)
            color = list(task.class_colors[classname]) + [alpha]
            # Initialize groundtruth image
            layer = np.full((shape[0], shape[1], 4), fill_value=0, dtype=np.uint8)
            layer = cv2.fillPoly(layer, coords, color)
            layer = imutil.rectify_to_float01(layer)
            yield layer
            # outline to see more clearly
            alpha = int(.95 * 255)
            color = list(task.class_colors[classname]) + [alpha]
            layer = np.full((shape[0], shape[1], 4), fill_value=0, dtype=np.uint8)
            layer = cv2.drawContours(layer, [coords], -1, color, 3)
            layer = imutil.rectify_to_float01(layer)
            yield layer

        priority = ['Crosswalk', 'Intersection', 'Trees', 'Grass', 'Parking_Lot']

        for frame_id, class_coords in frame_to_class_coords.items():
            frame_fpath = frame_image_fpaths[0]
            frame = cv2.imread(frame_fpath)
            shape = frame.shape[:2]
            # {c[0] for c in class_coords}
            layers = []
            boarder_layers = []
            class_coords = sorted(class_coords, key=lambda t: 900 if t[0] not in priority else priority.index(t[0]))
            classnames = set([p[0] for p in class_coords])
            for classname, poly_coords in reversed(class_coords):
                layer, layer_border = list(new_layer(shape, classname, poly_coords))
                layers.append(layer)
                boarder_layers.append(layer_border)

            layers = boarder_layers + layers

            topdown = layers[0]
            for layer in ub.ProgIter(layers[1:], label='blending'):
                topdown = imutil.overlay_alpha_images(topdown, layer)

            blend = imutil.overlay_alpha_images(topdown, imutil.ensure_grayscale(frame))

            import plottool as pt
            import matplotlib.patches as patches
            import matplotlib.pyplot as plt
            import matplotlib as mpl

            mpl.rcParams['legend.fontsize'] = 20
            mpl.rcParams['legend.loc'] = 'center'
            mpl.rcParams['axes.titlesize'] = 20
            mpl.rcParams['figure.titlesize'] = 20

            handles = [
                patches.Patch(color=np.array(bgr[::-1]) / 255, label=classname)
                for classname, bgr in ub.dict_subset(task.class_colors, classnames).items()
            ]
            n_cols = 5
            n = 1
            pt.imshow(blend, pnum=(1, n_cols, slice(0, n_cols - n)), fnum=1)
            ax = pt.gca()
            ax.set_title('Scene {}, frame {}'.format(scene, frame_id))

            pt.figure(fnum=1, pnum=(1, n_cols, slice(n_cols - n, n_cols)))
            ax = pt.gca()
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])
            plt.legend(handles=handles)
            utildraw.adjust_subplots(top=.9, bottom=0, left=0, right=1, wspace=.01)

            fig = pt.gcf()
            inches = np.array(blend.shape[:2][::-1]) / fig.dpi
            fig.set_size_inches(*inches)

            ub.ensuredir('scene_plots')
            cv2.imwrite('scene_plots/scene_{}_{}.png'.format(scene, frame_id),
                        utildraw.render_figure_to_image(fig, dpi=100,
                                                        transparent=True))
