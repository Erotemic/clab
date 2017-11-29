# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
from os.path import join, expanduser, basename, splitext, abspath, exists  # NOQA
import pandas as pd
import cv2
import glob
import ubelt as ub
import numpy as np
from clab.tasks._sseg import SemanticSegmentationTask
from clab import util
from clab.util import imutil
from clab.util import colorutil
from clab.util import fnameutil  # NOQA
from clab import inputs
from clab import preprocess
from clab import getLogger
import parse


if True:
    logger = getLogger(__name__)
    print = logger.info


def _imshow_dtm(image):
    """
    out_data2 = imutil.imread(out_path)
    out_data == out_data2

    image = out_data2

    """
    import copy
    import matplotlib as mpl
    from matplotlib.colors import Normalize
    import plottool as pt
    UNKNOWN = -32767
    vmin = image[image != UNKNOWN].min()
    vmax = image.max()
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = copy.copy(mpl.cm.get_cmap('viridis'))
    cmap.set_bad((0, 0, 0))
    pt.imshow(image, cmap=cmap, norm=norm, fnum=1)


class UrbanMapper3D(SemanticSegmentationTask):
    """
    References:
        https://community.topcoder.com/longcontest/?module=ViewProblemStatement&compid=57607&rd=17007

    Example:
        >>> from clab.tasks.urban_mapper_3d import *
        >>> task = UrbanMapper3D(root='~/remote/aretha/data/UrbanMapper3D',
        >>>                      workdir='~/data/work/urban_mapper3', boundary=True)
        >>> print(task.classnames)
        >>> task.prepare_fullres_inputs(force=True)
        >>> print(task.classnames)
        >>> task.preprocess()
        >>> (train, test), = task.xval_splits()
        >>> inputs_base = ub.ensuredir((task.workdir, 'inputs'))
        >>> train.base_dpath = inputs_base
        >>> train.prepare_images(force=True)
        >>> train.prepare_input()
        >>> gtstats = train.prepare_gtstats(task)
        >>> nan_value = -32767.0  # hack: specific number for DTM
        >>> center_stats = self.inputs.prepare_center_stats(
        >>>     task, nan_value=nan_value, colorspace='RGB')
    """
    def __init__(task, root=None, workdir=None, boundary=False):
        if root is None:
            assert False
        task.workdir = expanduser(workdir)
        task.root = expanduser(root)
        task.target_shape = (360, 480)
        task.input_shape = (360, 480)

        # the challenge training set (split into train test for our evaluation)
        task.fullres = None
        task.input_modes = {}
        task.augment_modes = {}

        task.eval_fullres = None  # the challenge evaluation dataset

        task.classnames = [
            'non-building',
            'building',
            'uncertain',
        ]
        task.null_classname = 'uncertain'
        super(UrbanMapper3D, task).__init__()
        task.boundary_mode_enabled = boundary

        if task.boundary_mode_enabled:
            task._boundary_mode()
        else:
            task.set_classnames(task.classnames, task.null_classname)

    def _boundary_mode(task):
        task.boundary_mode_enabled = True
        # update the task to reflect the updated gt labels
        classnames = [
            'non-building',
            'inner_building',
            'outer_building',
            'uncertain',
        ]
        null_classname = 'uncertain'
        task.set_classnames(classnames, null_classname, {})

    def customize_colors(task):
        # called by set_classnames
        lookup_bgr255 = colorutil.lookup_bgr255
        task.class_colors['non-building'] = lookup_bgr255('black')

    def preprocess(task):
        task.prepare_fullres_inputs()
        datadir = ub.ensuredir((task.workdir, 'data'))
        prep = preprocess.Preprocessor(datadir)
        prep.part_config['overlap'] = .75

        prep.ignore_label = task.ignore_label

        clear = 0
        fullres = task.fullres
        task.input_modes['lowres'] = prep.make_lowres(fullres, clear=clear)
        task.input_modes['part-scale1'] = prep.make_parts(
            fullres, scale=1, clear=clear)

        # for k, v in task.input_modes.items():
        #     # old code needed for caffe
        #     task.augment_modes[k] = prep.make_augment_inputs(v, rng='determ', clear=clear)

    def load_fullres_inputs(task, subdir='training'):
        """
        Loads the source data into the Inputs format for further processing.

        Example:
            >>> from clab.tasks.urban_mapper_3d import *
            >>> task = UrbanMapper3D(root='~/remote/aretha/data/UrbanMapper3D',
            >>>                      workdir='~/data/work/urban_mapper')
            >>> task.load_fullres_inputs()
            >>> subdir = 'training'

        """
        tagged_paths = {
            'gt': glob.glob(join(task.root, subdir, '*_GTL.tif')),
            'im': glob.glob(join(task.root, subdir, '*_RGB.tif')),

            'gti': glob.glob(join(task.root, subdir, '*_GTI.tif')),

            # digital terrain model
            'dtm': glob.glob(join(task.root, subdir, '*_DTM.tif')),
            # digital surface model
            'dsm': glob.glob(join(task.root, subdir, '*_DSM.tif')),
        }

        def extract_primary_key_info(paths, tag):
            if not paths:
                return pd.DataFrame()
            infos = [parse.parse('{site_id}_Tile_{N}_{type}.tif', p).named
                     for p in map(basename, paths)]
            df = pd.DataFrame(infos)
            df = df.rename(columns={'type': tag + 'type'})
            df[tag] = paths
            df = df.set_index(['site_id', 'N'], drop=False).sort_index()
            return df

        train = pd.DataFrame()
        for tag, paths in tagged_paths.items():
            _df = extract_primary_key_info(paths, tag)
            if len(_df):
                for pk in ['N', 'site_id']:
                    if pk not in train.columns:
                        train[pk] = _df[pk]
                train[tag] = _df[tag]

        null_idxs = list(set(np.where(pd.isnull(train))[0]))
        if null_idxs:
            raise ValueError(('MISSING DATA FOR {}'.format(
                [train.index[i] for i in null_idxs])))

        for tag, paths in tagged_paths.items():
            pass

        metadata = train[['site_id', 'N']].reset_index(drop=True)
        dump_im_names = ['{site_id}_Tile_{N}.tif'.format(**d)
                         for d in metadata.to_dict(orient='records')]

        # train_gts = list(train['gt'].values)
        # train_rgb = list(train['im'].values)
        # train_dtm = list(train['dtm'].values)
        # train_dsm = list(train['dsm'].values)

        # train_gts = sorted(train_gts)
        # train_rgb = fnameutil.align_paths(train_gts, train_rgb)
        # train_dtm = fnameutil.align_paths(train_gts, train_dtm)
        # train_dsm = fnameutil.align_paths(train_gts, train_dsm)
        # dump_im_names = ['{site_id}_Tile_{N}.tif'.format(**d) for d in infos]

        kw = train.drop(['N', 'site_id'], axis=1).to_dict(orient='list')
        fullres = inputs.Inputs.from_paths(**kw)

        # aux_paths = {'dtm': train_dtm, 'dsm': train_dsm}
        # fullres = {'im': train_rgb, 'gt': train_gts, 'aux': aux}

        fullres.dump_im_names = dump_im_names
        fullres.metadata = metadata

        # fullres.aux_paths = {}
        fullres.tag = 'fullres'
        return fullres

    def rebase_groundtruth(task, fullres, force=False):
        """
        Inplace / lazy modification of groundtruth labels

        hacky.
        """

        # Remap the original three labels to [0, 1, 2]
        orig_labels = [2, 6, 65]
        mapping = np.full(max(orig_labels) + 1, fill_value=-1)
        mapping[orig_labels] = np.arange(len(orig_labels))

        datadir = ub.ensuredir((task.workdir, 'data'))
        dpath = ub.ensuredir((datadir, 'gt', 'full'))

        new_gt_paths = []
        for ix in ub.ProgIter(range(len(fullres.paths['gt'])), label='rebase'):
            path = fullres.paths['gt'][ix]
            name = fullres.dump_im_names[ix]
            out_dpath = join(dpath, name)
            # Hacky cache
            if force or not exists(out_dpath):
                in_data = imutil.imread(path)
                out_data = mapping[in_data]
                imutil.imwrite(out_dpath, out_data)
            new_gt_paths.append(out_dpath)
        fullres.paths['gt'] = new_gt_paths
        return fullres

    def create_boundary_groundtruth(task, fullres, force=False):
        # Hack: transform task into boundary mode
        task._boundary_mode()

        NON_BUILDING = task.classname_to_id['non-building']
        INNER_INSTANCE = task.classname_to_id['inner_building']
        OUTER_INSTANCE = task.classname_to_id['outer_building']
        UNKNOWN = task.classname_to_id['uncertain']

        def instance_boundary(gti_data, gtl_data):
            """
                gti_data = np.array([
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 5, 5, 5, 5, 5, 0, 0, 0],
                  [0, 0, 5, 5, 5, 5, 5, 2, 2, 2],
                  [0, 0, 5, 5, 5, 5, 5, 2, 2, 2],
                  [0, 5, 5, 5, 5, 5, 5, 2, 2, 2],
                  [0, 5, 5, 5, 5, 5, 5, 0, 0, 0],
                  [0, 5, 5, 5, 5, 5, 5, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 3, 3, 0, 4, 0],
                  [1, 0, 0, 0, 0, 3, 3, 0, 0, 0],
                ], dtype=np.uint8)

                kernel = np.ones((2, 2))
                cv2.erode(im, kernel)
            """
            out_data = np.full(gti_data.shape, dtype=np.uint8,
                               fill_value=NON_BUILDING)

            # Map old unknown value to the a new one
            out_data[gtl_data == 65] = UNKNOWN

            kernel = np.ones((3, 3))
            touching_labels = touching_ccs(gti_data)

            # for each touching label
            simple_gti = gti_data.copy()
            for label in touching_labels:
                touching_mask = (gti_data == label)
                # remove it from the simple ones
                simple_gti[touching_mask] = 0
                # and it by itself
                ccmask = touching_mask.astype(np.uint8)
                inner = cv2.erode(ccmask, kernel, iterations=2)
                outer = ccmask - inner
                out_data[inner > 0] = INNER_INSTANCE
                out_data[outer > 0] = OUTER_INSTANCE

            # Then remove everything that is disjoint at once
            # (probably can do better in terms of speed)
            ccmask = (simple_gti > 0).astype(np.uint8)
            inner = cv2.erode(ccmask, kernel, iterations=2)
            outer = ccmask - inner
            out_data[inner > 0] = INNER_INSTANCE
            out_data[outer > 0] = OUTER_INSTANCE
            return out_data

        # Augment the groundtruth so the network must also predict instance
        # boundaries
        datadir = ub.ensuredir((task.workdir, 'data'))
        dpath = ub.ensuredir((datadir, 'gtb', 'full'))

        new_gt_paths = []
        for ix in ub.ProgIter(range(len(fullres.paths['gti'])), label='boundary'):
            path = fullres.paths['gti'][ix]
            name = fullres.dump_im_names[ix]
            out_dpath = join(dpath, name)
            if force or not exists(out_dpath):
                gtl_data = imutil.imread(fullres.paths['gt'][ix])
                gti_data = imutil.imread(path)
                out_data = instance_boundary(gti_data, gtl_data)
                imutil.imwrite(out_dpath, out_data)
            new_gt_paths.append(out_dpath)

        fullres.paths['gt'] = new_gt_paths
        return fullres

    def prepare_fullres_inputs(task, force=False):
        if not task.fullres:
            fullres = task.load_fullres_inputs('training')
            if task.boundary_mode_enabled:
                fullres = task.create_boundary_groundtruth(fullres, force=force)
            else:
                fullres = task.rebase_groundtruth(fullres, force=force)

            # aux can currently only contain the real aux channels
            del fullres.paths['gti']

            task.fullres = fullres

    def xval_splits(task, test_keys=None):
        import parse
        import logging
        parse.log.setLevel(logging.INFO)

        train_scene = 'JAX'
        test_scene = 'TAM'

        def primary_key_info(paths):
            infos = [parse.parse('{site_id}_Tile_{N}{junk}', p).named
                     for p in map(basename, paths)]
            df = pd.DataFrame(infos)
            return df

        train_inputs = []
        test_inputs = []

        # THESE PATHS MUST BE GENERATED IN THE SAME ORDER EACH TIME
        for k, v in sorted(task.input_modes.items()):
            df = primary_key_info(v.im_paths)
            parts = dict(list(df.groupby(['site_id'])))

            train_idx = parts[train_scene].index
            train_inputs.append(v.take(train_idx))

            if k in ['part-scale1']:
                test_idx = parts[test_scene].index
                test_inputs.append(v.take(test_idx))

        for k, v in sorted(task.augment_modes.items()):
            df = primary_key_info(v.im_paths)
            parts = dict(list(df.groupby(['site_id'])))
            train_idx = parts[train_scene].index
            train_inputs.append(v.take(train_idx))

        for v in train_inputs:
            assert not ub.find_duplicates(v.im_paths)
        for v in test_inputs:
            assert not ub.find_duplicates(v.im_paths)

        train = inputs.Inputs.union_all(*train_inputs)
        test = inputs.Inputs.union_all(*test_inputs)
        train.tag = 'train'
        test.tag = 'test'
        xval_split = (train, test)

        assert not bool(ub.find_duplicates(test.im_paths))
        assert not bool(ub.find_duplicates(train.im_paths))

        yield xval_split

    def restitch(task, output_dpath, part_paths, blend=None):

        def stitch_tiles(rc_locs, tiles):
            """
            Recombine parts back into an entire image
            (TODO: blending / averaging to remove boundry effects)
            """
            shapes = [t.shape[0:2] for t in tiles]
            n_channels = 1 if len(tiles[0].shape) == 2 else tiles[0].shape[2]
            bboxes = np.array([
                (r, c, r + h, c + w)
                for ((r, c), (h, w)) in zip(rc_locs, shapes)
            ])
            stiched_shape = tuple(bboxes.T[2:4].max(axis=1))
            if n_channels > 1:
                stiched_shape = stiched_shape + (n_channels,)
            stiched_pred = np.zeros(stiched_shape)
            for bbox, tile in zip(bboxes, tiles):
                r1, c1, r2, c2 = bbox
                stiched_pred[r1:r2, c1:c2] = tile
            return stiched_pred

        def stitch_tiles_ave(rc_locs, tiles):
            """
            Recombine parts back into an entire image
            (TODO: blending / averaging to remove boundry effects)

            Example:
                >>> rc_locs = [(0, 0), (0, 5), (0, 10)]
                >>> tiles = [np.ones((1, 7, 3)) + i for i in range(len(rc_locs))]
                >>> tiles = [np.ones((1, 7)) + i for i in range(len(rc_locs))]
            """
            shapes = [t.shape[0:2] for t in tiles]
            n_channels = 1 if len(tiles[0].shape) == 2 else tiles[0].shape[2]
            bboxes = np.array([
                (r, c, r + h, c + w)
                for ((r, c), (h, w)) in zip(rc_locs, shapes)
            ])
            stiched_wh = tuple(bboxes.T[2:4].max(axis=1))
            stiched_shape = stiched_wh
            if n_channels > 1:
                stiched_shape = stiched_wh + (n_channels,)
            sums = np.zeros(stiched_shape)
            nums = np.zeros(stiched_wh)
            for bbox, tile in zip(bboxes, tiles):
                r1, c1, r2, c2 = bbox
                sums[r1:r2, c1:c2] += tile
                nums[r1:r2, c1:c2] += 1

            if len(sums.shape) == 2:
                stiched_pred = sums / nums
            else:
                stiched_pred = sums / nums[:, :, None]
            return stiched_pred

        def stitch_tiles_vote(rc_locs, tiles):
            """
            Recombine parts back into an entire image
            (TODO: blending / averaging to remove boundry effects)
            """
            shapes = [t.shape[0:2] for t in tiles]
            n_channels = 1 if len(tiles[0].shape) == 2 else tiles[0].shape[2]
            assert n_channels == 1
            bboxes = np.array([
                (r, c, r + h, c + w)
                for ((r, c), (h, w)) in zip(rc_locs, shapes)
            ])
            stiched_shape = tuple(bboxes.T[2:4].max(axis=1))
            n_classes = len(task.classnames)
            votes = np.zeros((n_classes,) + stiched_shape)
            for bbox, tile in zip(bboxes, tiles):
                r1, c1, r2, c2 = bbox
                for i in range(n_classes):
                    votes[i, r1:r2, c1:c2][tile == i] += 1
            stiched_pred = votes.argmax(axis=0)
            return stiched_pred

        def _extract_part_grid(paths):
            # hack to use filenames to extract upper left locations of tiles in
            # the larger image.
            rc_locs = [[int(x) for x in basename(p).split('.')[0].split('_')[-2:]]
                       for p in paths]
            return rc_locs

        # Group parts by base id
        groupid = [basename(p).split('_part')[0] for p in part_paths]
        new_paths = []
        for tileid, paths in ub.ProgIter(list(ub.group_items(part_paths, groupid).items())):
            # Read all parts belonging to an original group
            tiles = [imutil.imread(p) for p in paths]
            # Find their relative positions and restitch them
            rc_locs = _extract_part_grid(paths)
            if blend == 'vote':
                stiched_pred = stitch_tiles_vote(rc_locs, tiles)
            elif blend == 'ave':
                stiched_pred = stitch_tiles_ave(rc_locs, tiles)
            elif blend is None:
                stiched_pred = stitch_tiles(rc_locs, tiles)
            else:
                raise KeyError(blend)
            # Write them to disk.
            fpath = join(output_dpath, tileid + '.png')
            imutil.imwrite(fpath, stiched_pred)
            new_paths.append(fpath)
        return new_paths

    def instance_label(task, pred, k=15, n_iters=1, dist_thresh=5, watershed=False):
        """
        Do some postprocessing to label instances instead of classes
        """
        mask = pred

        # noise removal
        if k > 0 and n_iters > 0:
            kernel = np.ones((k, k), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel,
                                    iterations=n_iters)

        if watershed:
            from clab.torch import filters
            mask = filters.watershed_filter(mask, dist_thresh=dist_thresh)

        mask = mask.astype(np.uint8)
        n_ccs, cc_labels = cv2.connectedComponents(mask, connectivity=4)
        return cc_labels


def script_overlay_aux():
    """
    """
    task = UrbanMapper3D(root='~/remote/aretha/data/UrbanMapper3D',
                         workdir='~/data/work/urban_mapper')
    fullres = task.load_fullres_inputs(subdir='training')
    NAN_VAL = -32767

    for paths in ub.ProgIter(fullres, adjust=False, freq=1):
        bgr = util.imread(paths['im'])

        gtl = util.imread(paths['gt'])
        gti = util.imread(paths['gt'].replace('GTL', 'GTI'))

        dsm = util.imread(paths['aux']['dsm'])
        dsm[(NAN_VAL == dsm)] = np.nan

        dtm = util.imread(paths['aux']['dtm'])
        dtm[(NAN_VAL == dtm)] = np.nan

        diff = dtm - dsm

        def normalize(chan):
            min_val = np.nanmax(chan)
            max_val = np.nanmin(chan)
            is_nan = np.isnan(chan)
            norm = chan.copy()
            norm[~is_nan] = (norm[~is_nan] - min_val) / (max_val - min_val)
            norm[is_nan] = 0
            return norm, min_val, max_val

        def colorize(chan):
            norm, min_val, max_val = normalize(chan)

            domain = np.linspace(min_val, max_val)

            shape = (norm.shape[0], norm.shape[0] / 25)
            cb = colorutil.colorbar_image(domain, dpi=200, shape=shape)[:, :, 0: 3]
            sfy = norm.shape[0] / cb.shape[0]
            cb = imutil.imscale(cb, scale=(sfy, sfy))[0]

            color_chan = util.make_heatmask(norm)[:, :, 0:3]
            color_chan[np.isnan(chan)] = [[0, 0, 1]]
            blend_chan = util.overlay_colorized(color_chan, bgr, alpha=.2)[:, :, 0:3]

            blend_gt_chan = draw_instance_contours(blend_chan, gti, gtl,
                                                   thickness=2, alpha=.3)

            # Add a colorbar
            blend_chan = np.hstack([blend_chan, cb])
            blend_gt_chan = np.hstack([blend_gt_chan, cb])

            return blend_chan, blend_gt_chan

        blend_dsm, blend_gt_dsm = colorize(dsm)
        blend_dtm, blend_gt_dtm = colorize(dtm)
        blend_diff, blend_gt_diff = colorize(diff)

        base_dpath = ub.ensuredir(join(task.workdir, 'viz'))

        outputs = {
            'blend_diff': blend_diff,
            'blend_dsm': blend_dsm,
            'blend_dtm': blend_dtm,

            'blend_gt_diff': blend_gt_diff,
            'blend_gt_dsm': blend_gt_dsm,
            'blend_gt_dtm': blend_gt_dtm,
        }

        for key, val in outputs.items():
            out_dpath = ub.ensuredir((base_dpath, key))
            out_fpath = join(out_dpath, ub.augpath(paths['dump_fname'], ext='.png'))
            util.imwrite(out_fpath, val)
        # util.overlay


def draw_mask_contours(img, gt, thickness=2, alpha=1):
    import cv2

    gt = gt.astype(np.uint8)
    border = cv2.copyMakeBorder(gt, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0 )
    _, contours, hierarchy = cv2.findContours(border, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE, offset=(-1, -1))

    BGR_GREEN = (0, 255, 0)
    img = util.ensure_float01(img)
    base = np.ascontiguousarray((255 * img[:, :, 0:3]).astype(np.uint8))
    if alpha >= 1:
        draw_img = cv2.drawContours(
            image=base, contours=contours, contourIdx=-1, color=BGR_GREEN,
            thickness=thickness)
    else:
        # Draw an image to overlay first
        draw_img = cv2.drawContours(
            image=np.zeros(base.shape, dtype=np.uint8), contours=contours,
            contourIdx=-1, color=BGR_GREEN, thickness=thickness)
        contour_overlay = util.ensure_alpha_channel(draw_img, alpha=0)
        contour_overlay.T[3].T[draw_img.sum(axis=2) > 0] = alpha

        # zero out the edges to avoid visualization errors
        contour_overlay[0:thickness, :, :] = 0
        contour_overlay[-thickness:, :, :] = 0
        contour_overlay[:, 0:thickness, :] = 0
        contour_overlay[:, -thickness:, :] = 0

        draw_img = util.overlay_alpha_images(contour_overlay, base)
        draw_img = np.ascontiguousarray((255 * draw_img[:, :, 0:3]).astype(np.uint8))
    return draw_img


def draw_instance_contours(img, gti, gtl=None, thickness=2, alpha=1):
    """

    img = util.imread('/home/joncrall/remote/aretha/data/UrbanMapper3D/training/TAM_Tile_003_RGB.tif')
    gti = util.imread('/home/joncrall/remote/aretha/data/UrbanMapper3D/training/TAM_Tile_003_GTI.tif')
    gtl = util.imread('/home/joncrall/remote/aretha/data/UrbanMapper3D/training/TAM_Tile_003_GTL.tif')
    thickness = 2
    alpha = 1

    """
    import cv2

    rc_locs = np.where(gti > 0)
    grouped_cc_rcs = util.group_items(
        np.ascontiguousarray(np.vstack(rc_locs).T),
        gti[rc_locs], axis=0
    )

    def bounding_box(rcs):
        rc1 = rcs.min(axis=0)
        rc2 = rcs.max(axis=0)
        return rc1, rc2

    # slice out a bounding region around each instance, detect the contour and
    # then offset it back into image coordinates
    grouped_contours = {}
    for label, rcs in grouped_cc_rcs.items():
        rc1, rc2 = bounding_box(rcs)
        sl = (slice(rc1[0], rc2[0] + 2), slice(rc1[1], rc2[1] + 2))
        subimg = (gti[sl] == label).astype(np.uint8)

        xy_offset = rc1[::-1]
        offset = xy_offset + [-2, -2]

        border = cv2.copyMakeBorder(subimg, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=0 )
        _, contors, hierarchy = cv2.findContours(border, cv2.RETR_TREE,
                                                 cv2.CHAIN_APPROX_SIMPLE,
                                                 offset=tuple(offset))
        """
        offset = [0, 0]
        BGR_GREEN = (0, 255, 0)
        x = np.ascontiguousarray(util.ensure_alpha_channel(subimg)[:, :, 0:3]).astype(np.uint8)
        draw_img = cv2.drawContours(
            image=x, contours=contors,
            contourIdx=-1, color=BGR_GREEN, thickness=2)
        """
        # note when len(contours > 1, there is a hole in the building)
        # assert len(contors) == 1
        grouped_contours[label] = contors[0]

    if gtl is not None:
        unknown_labels = set(np.unique(gti[gtl == 65]))
    else:
        unknown_labels = set()

    known_labels = set(grouped_contours.keys()) - unknown_labels

    BGR_GREEN = (0, 255, 0)
    BGR_BLUE = (255, 0, 0)
    img = util.ensure_float01(img)
    base = np.ascontiguousarray((255 * img[:, :, 0:3]).astype(np.uint8))

    # Draw an image to overlay first
    draw_img = np.zeros(base.shape, dtype=np.uint8)

    known_contours = list(ub.take(grouped_contours, known_labels))
    draw_img = cv2.drawContours(
        image=draw_img, contours=known_contours,
        contourIdx=-1, color=BGR_GREEN, thickness=thickness)

    if unknown_labels:
        unknown_contours = list(ub.take(grouped_contours, unknown_labels))
        draw_img = cv2.drawContours(
            image=draw_img, contours=unknown_contours,
            contourIdx=-1, color=BGR_BLUE, thickness=thickness)

    contour_overlay = util.ensure_alpha_channel(draw_img, alpha=0)
    contour_overlay.T[3].T[draw_img.sum(axis=2) > 0] = alpha

    # zero out the edges to avoid visualization errors
    contour_overlay[0:thickness, :, :] = 0
    contour_overlay[-thickness:, :, :] = 0
    contour_overlay[:, 0:thickness, :] = 0
    contour_overlay[:, -thickness:, :] = 0

    draw_img = util.overlay_alpha_images(contour_overlay, base)
    draw_img = np.ascontiguousarray((255 * draw_img[:, :, 0:3]).astype(np.uint8))
    return draw_img


def touching_ccs(instance_markers):
    locs = np.where(instance_markers)
    shape = instance_markers.shape
    high = [np.minimum(locs[i] + 1, shape[i] - 1) for i in range(2)]
    low  = [np.maximum(locs[i] - 1, 0) for i in range(2)]

    labels = instance_markers[locs]

    neighors8 = np.hstack([
        # important to use 8-cc here
        instance_markers[(high[0], locs[1])][:, None],
        instance_markers[( low[0], locs[1])][:, None],
        instance_markers[(locs[0], high[1])][:, None],
        instance_markers[(locs[0],  low[1])][:, None],
        instance_markers[(high[0], high[1])][:, None],
        instance_markers[( low[0],  low[1])][:, None],
        instance_markers[( low[0], high[1])][:, None],
        instance_markers[(high[0],  low[1])][:, None],
    ])

    ignore_labels = set([0])
    unique_neighbors = [set(s) - ignore_labels  - {l}
                        for s, l in zip(neighors8, labels)]

    import itertools as it
    touching_labels = set(it.chain.from_iterable(unique_neighbors))
    return touching_labels


# def discover_classes(prep, fullres):
#     import pandas as pd
#     from six.moves import reduce
#     class_freqs = []
#     for path in ub.ProgIter(fullres.paths['gt']):
#         true = imutil.imread(path)
#         freq = pd.value_counts(true.ravel())
#         class_freqs.append(freq)
#     total_freq = reduce(sum, class_freqs)
#     print('total_freq = {!r}'.format(total_freq))


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m clab.tasks.urban_mapper_3d
    """
    # import utool as ut
    # ut.doctest_funcs()
    import xdoctest
    xdoctest.doctest_module()
