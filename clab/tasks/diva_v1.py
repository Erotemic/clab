# -*- coding: utf-8 -*-
"""
Handles transforming the diva-v1 dataset into caffe format

SeeAlso:
    http://mi.eng.cam.ac.uk/projects/segnet/tutorial.html

Developer:
    ln -s ~/code/baseline-algorithms/Semantic_Seg ~/sseg
    cd ~/sseg/virat
"""
from __future__ import absolute_import, division, print_function
from os.path import join, expanduser, basename, splitext, abspath, exists
import ubelt as ub
import os
import cv2
import glob
import json
import shutil
import itertools as it
import numpy as np
from PIL import Image
from . import inputs
from .util import fnameutil
from .util import gpu_util
from .util import imutil
from .util import colorutil
from .util import hashutil
from .util import jsonutil
from . import util
from . import augment
from .tasks._sseg import SemanticSegmentationTask

from . import getLogger
logger = getLogger(__name__)
print = logger.info


class DivaV1(SemanticSegmentationTask):
    """
    Diva v-1 anotations

    CommandLine:
        python -m clab.tasks DivaV1

    Example:
        >>> from clab.tasks import *
        >>> task = DivaV1()
        >>> print(task.classnames)
    """

    def __init__(task, repo=None, workdir=None, clean=2):
        # TODO: generalize
        if repo is None:
            task.repo = expanduser('~/data/v1-annotations')
        else:
            task.repo = expanduser(repo)
        # TODO: generate this automatically
        classnames = [
            'Sky',
            'Building',
            'Street',
            'Parking_Lot',
            'Trees',
            'Crosswalk',
            'Grass',
            'Ground',
            'Intersection',
            'Shadows',
            'Sidewalk',
            'Stairs',
            'Other',
            'Background',
            'Unannotated'
        ]
        null_classname = 'Unannotated'

        if clean <= 0:
            alias = {}
        elif clean == 1:
            alias = {
                # 'Sidewalk': 'Unannotated',
                # 'Grass': 'Ground',
                'Shadows': 'Unannotated',
                'Stairs': 'Unannotated',
                'Background': 'Unannotated',
                'Other': 'Unannotated',
            }
        else:
            # These roughly correspond to shruti's classes
            alias = {
                # Remove Shadows / Background / Other
                'Shadows': 'Unannotated',
                'Background': 'Unannotated',
                'Other': 'Unannotated',

                # Tenous: Remove Sky, Ground, Stairs
                'Sky': 'Unannotated',
                'Ground': 'Unannotated',
                'Stairs': 'Unannotated',

                # VERY Tenous: Remove Sidewalk
                'Sidewalk': 'Unannotated',

                # Intersection and crosswalk become street
                'Intersection': 'Street',
                'Crosswalk': 'Street',
            }
        task.clean = clean

        util.super2(DivaV1, task).__init__(classnames, null_classname, alias)
        task._data_id = None

        # Special colors
        task.customize_colors()

        # Work directory
        if workdir is None:
            task.workdir = expanduser('~/data/work/diva/')
        else:
            task.workdir = expanduser(workdir)

        # Define data preprocessingg
        task.target_shape = (360, 480)
        task.input_shape = (360, 480)
        task.part_overlap = .5
        task.part_keepbound = True
        # task.part_overlap = 0
        # task.part_keepbound = False

        # task.enable_augment = True
        # task.enable_augment = False
        task.enable_augment = True
        # task.aug_params = {
        #     'axis_flips': [0, 1],
        #     'gammas':  [2.5],
        #     'blur':  [10],
        #     'n_rand_zoom':  1,
        #     'n_rand_occl': 1,
        # }
        task.aug_params = {
            'axis_flips': [0, 1],
            'gammas':  [.5, 2.5],
            'blur':  [5, 20],
            'n_rand_zoom':  2,
            'n_rand_occl': 2,
        }

        task.base_modes = ['lowres', 'part-scale1', 'part-scale2']
        # task.base_modes = ['lowres']

        if task.enable_augment:
            task.modes = task.base_modes + [m + '-aug' for m in task.base_modes]
        else:
            task.modes = task.base_modes[:]
        task.scene_base = join(task.repo, 'active')

    @property
    def preproc_data_id(task):
        if task._data_id is None:
            # We actually don't need to regenerate every time one of these
            # changes, but it makes things safer for now.
            depends = [
                task.enable_augment and task.aug_params,
                task.input_shape,  # FIXME: really a model spec, but needed here
                task.part_overlap,
                task.part_keepbound,
                task.base_modes,
                task.classnames,
            ]
            # In the future, the only thing that really matters for everything
            # is task.input_shape.

            # Different modes will have different dependencies, and those
            # should be accounted for on a per-mode basis. The same should be
            # done with augmentation.

            # lowres does not have additional dependencies

            # The part-scaleX modes depend on:
            # task.part_overlap and task.part_keepbound

            # aug modes have task.aug_params as a dependency.

            # The groundtruth should also be kept in another further-up
            # directory, and they have a strong dependency on task.classnames.
            task._data_id = hashutil.hash_data(ub.repr2(depends), hashlen=8)
        return task._data_id

    @property
    def datadir(task):
        # Make directory depend on our cleaning strategy
        # need to also take into account augmentation / parts vs full

        data_id = task.preproc_data_id
        return join(task.workdir, 'data_{}_{}'.format(task.clean, data_id))

    def datasubdir(task, *args, **kw):
        """
        use for making directories where we will store train/test data
        """
        dname = join(*((task.datadir,) + args))
        if kw.get('ensure', True):
            ub.ensuredir(dname)
        return dname

    def customize_colors(task):
        lookup_bgr255 = colorutil.lookup_bgr255
        task.class_colors[task.null_classname] = lookup_bgr255('black')
        task.class_colors['Sky']               = lookup_bgr255('skyblue')
        task.class_colors['Trees']             = lookup_bgr255('forestgreen')
        task.class_colors['Sidewalk']          = lookup_bgr255('plum')
        task.class_colors['Intersection']      = lookup_bgr255('red')
        task.class_colors['Ground']            = lookup_bgr255('rosybrown')
        task.class_colors['Crosswalk']         = lookup_bgr255('royalblue')
        task.class_colors['Grass']             = lookup_bgr255('palegreen')
        task.class_colors['Building']          = lookup_bgr255('orangered')
        task.class_colors['Parking_Lot']       = lookup_bgr255('yellow')
        task.class_colors['Street']            = lookup_bgr255('orange')

    def extend_data_from(task, other, force=False):
        """
        Map the images / groundtruth from another similar task to this one

        >>> from clab.tasks import *
        >>> other = CamVid(repo=expanduser('~/sseg/SegNet'))
        >>> task = DivaV1(clean=2)
        >>> force = False
        """
        src_fpaths = other.all_gt_paths()
        dst_dir = task.datasubdir('extern', other.name)
        dst_fpaths = [join(dst_dir, p) for p in fnameutil.dumpsafe(src_fpaths)]

        need_convert = True

        if not force:
            existing_fpaths = imutil.load_image_paths(dst_dir, ext='.png')
            existing_fpaths == dst_fpaths

            try:
                existing_fpaths = fnameutil.align_paths(dst_fpaths, existing_fpaths)
            except Exception:
                pass
            else:
                dst_fpaths = existing_fpaths
                need_convert = False

        if need_convert:
            from_other_label = task.convert_labels_from(other)
            for src_fpath, dst_fpath in ub.ProgIter(zip(src_fpaths, dst_fpaths),
                                                    length=len(src_fpaths),
                                                    label='converting ' + other.name):
                src_gt = cv2.imread(src_fpath, flags=cv2.IMREAD_UNCHANGED)
                dst_gt = from_other_label[src_gt]
                cv2.imwrite(dst_fpath, dst_gt)

        im_fpaths = other.all_im_paths()
        task.extern_train_im_paths += im_fpaths
        task.extern_train_gt_paths += dst_fpaths

        assert fnameutil.check_aligned(task.extern_train_im_paths,
                                       task.extern_train_gt_paths), (
                                           'should be aligned. unknown error')

    def convert_labels_from(task, other):
        from_other = task.convert_classnames_from(other.name)

        from_other_label = np.full(other.labels.max() + 1, fill_value=task.ignore_label)
        for other_name, us_name in from_other.items():
            other_label = other.classname_to_id[other_name]
            us_label = task.classname_to_id[us_name]
            from_other_label[other_label] = us_label
        return from_other_label

    def convert_classnames_from(task, other):
        """
        Map labels from this task onto labels from another
        """
        if other == 'CamVid' or other.name == 'CamVid':
            from_other = {
                'Sky'      : 'Sky',
                'Road'     : 'Street',
                'Pavement' : 'Parking_Lot' ,
                'Tree'     : 'Trees',
                'Building' : 'Building',
            }

            # to_other = {
            #     'Sky': 'Sky',
            #     'Street': 'Road',
            #     'Parking_Lot': 'Pavement',
            #     'Tree': 'Tree',
            #     'Building': 'Building',
            # }
        else:
            assert False, 'unknown task'

        # Apply the current aliases
        from_other = {
            k: task.classname_alias.get(v, v)
            for k, v in from_other.items()
        }
        return from_other

    @property
    def scene_ids(task):
        scene_ids = sorted(os.listdir(task.scene_base))
        scene_ids = [p for p in scene_ids if p != '0501-5hz']
        return scene_ids

    def convert_scene_elements(task):
        for scene in ub.ProgIter(task.scene_ids, label='convert full scene', verbose=3):
            task.convert_full_scene(scene)

    def preprocess_lowres(task):
        for scene in ub.ProgIter(task.scene_ids, label='preproc lowres scene', verbose=3):
            task.make_lowres_scene(scene)

    def preprocess_parts(task, scale=1):
        for scene in ub.ProgIter(task.scene_ids, label='preproc part scene scale={}'.format(scale), verbose=3):
            task.make_part_scene(scene, scale=scale)

    def preprocess_augment(task, modes=None):
        assert task.enable_augment
        if modes is None:
            modes = task.base_modes
        for scene in ub.ProgIter(task.scene_ids, label='preproc augment scene', verbose=3):
            for mode in ub.ProgIter(modes, label=' * mode', verbose=0):
                task.make_augment_scene(mode, scene, rng='determ')

    def create_groundtruth(task, force=False):
        """
        CommandLine:
            python -m clab.tasks DivaV1.create_groundtruth

        Example:
            >>> from clab.tasks import *
            >>> task = DivaV1()
            >>> task.create_groundtruth(force=True)
        """
        assert exists(task.repo), 'v1 repo does not exist'
        assert exists(task.scene_base), 'v1 repo does not have correct structure'

        print('[task] Ensure groundtruth exists in task.datadir = {!r}'.format(task.datadir))
        checkdir = task.datasubdir('gt' + 'full')
        if exists(checkdir) and len(os.listdir(checkdir)) > 0 and not force:
            # only do this if it hasn't been done.
            # FIXME: need better cache mechanism
            print('[task] diva groundtruth already setup')
            return

        task.convert_scene_elements()

        # TODO: use the clab.preprocess for the majority of this logic
        if 'lowres' in task.base_modes:
            task.preprocess_lowres()
            if task.enable_augment:
                task.preprocess_augment(modes=['lowres'])

        if 'part-scale1' in task.base_modes:
            task.preprocess_parts(scale=1)
            if task.enable_augment:
                task.preprocess_augment(modes=['part-scale1'])

        if 'part-scale2' in task.base_modes:
            task.preprocess_parts(scale=2)
            if task.enable_augment:
                task.preprocess_augment(modes=['part-scale2'])

        overlay = True
        if overlay:
            task.draw_data_overlay()

    def _parse_scene_elements(task, scene_json_fpath):
        """
        Parses the polygon points in the first and last frame of a scene
        """
        with open(scene_json_fpath) as data_file:
            data = json.load(data_file)
        tracks = data['tracks']

        all_frame_ids = {val for key, val in jsonutil.walk_json(data)
                         if key == 'frame_id'}
        # Find the first and last frame numbers
        # Only keep min and max frame
        valid_frame_ids = {min(all_frame_ids), max(all_frame_ids)}

        # Collect the labeled bounding polygons for each class
        frame_to_class_coords = {f: [] for f in valid_frame_ids}
        for track in tracks:
            if 'label' in track:
                classname = track['label']
                for frame in track['frames']:
                    frame_id = frame['frame_id']
                    if frame_id in valid_frame_ids:
                        poly_coords = [
                            (point['image_x'], point['image_y'])
                            for point in frame['polygon']
                        ]
                        frame_to_class_coords[frame_id].append(
                            (classname, poly_coords))
        return frame_to_class_coords

    def _scene_elements_to_pixel_labels(task, frame_image_fpaths,
                                        frame_to_class_coords):
        """
        Worker function. Converts the layered scene elemenents into pixel
        labels suitable for semantic segmentation.
        """
        seg_images = {}
        for frame_id, class_coords in frame_to_class_coords.items():
            # {c[0] for c in class_coords}

            # Initialize groundtruth image
            fpath = frame_image_fpaths[frame_id]
            pil_img = Image.open(fpath)
            # FIXME: Needs dtype change if we use more than 255 classes
            # that wil also require working with cv2 function like
            # fillPoly and imwrite. (might need to store as float32)
            gt_img = np.full(
                (pil_img.height, pil_img.width),
                fill_value=task.ignore_label, dtype=np.uint8)
            pil_img.close()
            for classname, poly_coords in reversed(class_coords):
                label = task.classname_to_id[classname]
                if label not in task.ignore_labels:
                    # might be a python 2/3 issue with this opencv call
                    # I think this is inplace in python2
                    coords = np.round(np.array([poly_coords])).astype(np.int)
                    gt_img = cv2.fillPoly(gt_img, coords, label)

            for priority_classname in ['Crosswalk', 'Intersection', 'Trees', 'Grass', 'Parking_Lot'][::-1]:
                # HACK ONE MORE TIME FOR PRIORITY CLASSNAMES
                for classname, poly_coords in reversed(class_coords):
                    if classname == priority_classname:
                        label = task.classname_to_id[classname]
                        if label not in task.ignore_labels:
                            # might be a python 2/3 issue with this opencv call
                            # I think this is inplace in python2
                            coords = np.round(np.array([poly_coords])).astype(np.int)
                            gt_img = cv2.fillPoly(gt_img, coords, label)

            DEBUG_CONTOURS = False
            if DEBUG_CONTOURS:
                # HACK DRAW CONTOURS AROUND EVERYTHING to see what is occluded
                for classname, poly_coords in reversed(class_coords):
                    label = task.classname_to_id[classname]
                    if label not in task.ignore_labels:
                        # might be a python 2/3 issue with this opencv call
                        # I think this is inplace in python2
                        coords = np.round(np.array([poly_coords])).astype(np.int)
                        gt_img = cv2.drawContours(gt_img, [coords], -1, label, 5)
            seg_images[frame_id] = gt_img
        return seg_images

    def convert_full_scene(task, scene, overlay=True):
        """
        Creates the full resolution groundtruth images from the annotations.
        Converts the layered scene elemenents into pixel labels suitable for
        semantic segmentation.

        Example:
            >>> from clab.tasks import *
            >>> task = DivaV1()
            >>> scene = task.scene_ids[0]
            >>> scene = '0401'
            >>> scene = '0400'
        """
        scene_path = join(task.scene_base, scene, 'static')
        frame_image_fpaths = sorted(glob.glob(join(scene_path, '*.png')))
        scene_json_fpath = join(scene_path, 'static.json')

        frame_to_class_coords = task._parse_scene_elements(scene_json_fpath)
        seg_images = task._scene_elements_to_pixel_labels(frame_image_fpaths,
                                                          frame_to_class_coords)

        if overlay:
            scene_overlay_dir = task.datasubdir('overlay', 'full')

        # Write the images to disk
        scene_gtfull_dpath = task.datasubdir('gt' + 'full', scene)
        scene_imfull_dpath = task.datasubdir('im' + 'full', scene)
        for frame_id, gt_img in seg_images.items():
            frame_fpath = frame_image_fpaths[frame_id]
            fname = basename(frame_fpath)
            gt_fpath = join(scene_gtfull_dpath, fname)
            cv2.imwrite(gt_fpath, gt_img)
            # print('gt_fpath = {!r}'.format(gt_fpath))
            # Take the source data
            shutil.copy(frame_fpath, join(scene_imfull_dpath, fname))

            if overlay:
                # Make a nice visualization
                frame_fpath = frame_image_fpaths[frame_id]
                fname = basename(frame_fpath)
                # fpath = join(scene_overlay_dir, fname + '.overlay.png')
                fpath = join(scene_overlay_dir, scene + '-' + fname + '.overlay.png')
                gt_color = task.colorize(gt_img)
                gt_overlay = imutil.overlay_colorized(gt_color, cv2.imread(frame_fpath))
                # print('fpath = {!r}'.format(fpath))
                cv2.imwrite(fpath, gt_overlay)

    def make_augment_scene(task, mode, scene, rng=None):
        """
        Augments data in a scene of a specific "mode"

        mode = 'part-scale1'
        scene = '0000'
        rng = 'determ'

        gtdir = task.datasubdir('gtpart', scene))
        imdir = task.datasubdir('impart', scene))
        """
        assert task.enable_augment

        if rng == 'determ':
            # Make a determenistic seed based on the scene and mode
            seed = int(hashutil.hash_data([scene, mode], alphabet='hex'), 16)
            seed = seed % (2 ** 32 - 1)
            rng = np.random.RandomState(seed)

        auger = augment.SSegAugmentor(rng=rng, ignore_label=task.ignore_label)
        auger.params = task.aug_params

        # rng = np.random.RandomState(0)
        imdir = task.datasubdir('im' + mode, scene)
        gtdir = task.datasubdir('gt' + mode, scene)
        im_fpaths = sorted(glob.glob(join(imdir, '*.png')))
        gt_fpaths = sorted(glob.glob(join(gtdir, '*.png')))

        # Define the output path for the augmentation of this mode
        key = mode + '-aug'
        scene_imout_dpath = task.datasubdir('im' + key, scene)
        scene_gtout_dpath = task.datasubdir('gt' + key, scene)

        # Start fresh. Remove existing files
        ub.delete(scene_gtout_dpath, verbose=False)
        ub.delete(scene_imout_dpath, verbose=False)
        ub.ensuredir(scene_gtout_dpath)
        ub.ensuredir(scene_imout_dpath)

        for impath, gtpath in ub.ProgIter(list(zip(im_fpaths, gt_fpaths)), label='   * augment mode={}'.format(mode)):
            fname_we = splitext(basename(impath))[0]
            im = cv2.imread(impath, flags=cv2.IMREAD_UNCHANGED)
            gt = cv2.imread(gtpath, flags=cv2.IMREAD_UNCHANGED)
            aug_gen = auger.augment(im, gt)
            for augx, aug_data in enumerate(aug_gen):
                (im_aug, gt_aug) = aug_data[0:2]
                fname = '{}_aug{:0=4d}.png'.format(fname_we, augx)
                cv2.imwrite(join(scene_imout_dpath, fname), im_aug)
                cv2.imwrite(join(scene_gtout_dpath, fname), gt_aug)
        return scene_imout_dpath, scene_gtout_dpath

    def make_lowres_scene(task, scene):
        scene_gtfull_dpath = task.datasubdir('gt' + 'full', scene)
        scene_imfull_dpath = task.datasubdir('im' + 'full', scene)
        gt_fpaths = sorted(glob.glob(join(scene_gtfull_dpath, '*.png')))
        im_fpaths = sorted(glob.glob(join(scene_imfull_dpath, '*.png')))

        # Define the output path for this preprocessing mode
        mode = 'lowres'
        scene_gtout_dpath = task.datasubdir('gt' + mode, scene)
        scene_imout_dpath = task.datasubdir('im' + mode, scene)

        # Start fresh. Remove existing files
        ub.delete(scene_gtout_dpath, verbose=False)
        ub.delete(scene_imout_dpath, verbose=False)
        ub.ensuredir(scene_gtout_dpath)
        ub.ensuredir(scene_imout_dpath)

        target_dsize = tuple(task.input_shape[::-1])
        for impath, gtpath in zip(im_fpaths, gt_fpaths):
            im = cv2.imread(impath, flags=cv2.IMREAD_UNCHANGED)
            gt = cv2.imread(gtpath, flags=cv2.IMREAD_UNCHANGED)
            im_lowres = cv2.resize(im, target_dsize, interpolation=cv2.INTER_LANCZOS4)
            gt_lowres = cv2.resize(gt, target_dsize, interpolation=cv2.INTER_NEAREST)

            fname = basename(impath)
            cv2.imwrite(join(scene_imout_dpath, fname), im_lowres)
            cv2.imwrite(join(scene_gtout_dpath, fname), gt_lowres)
        return scene_imout_dpath, scene_gtout_dpath

    def make_part_scene(task, scene, scale=1):
        """
        Slices the full scene into smaller parts that fit into the network but
        are at the original resolution (or higher).

        >>> scene = '0001'
        >>> scale = 1
        """
        if task.part_overlap < 0 or task.part_overlap >= 1:
            raise ValueError(('part overlap was {}, but it must be '
                              'in the range [0, 1)').format(task.part_overlap))

        input_shape = task.input_shape
        overlap = task.part_overlap
        keepbound = task.part_keepbound

        scene_gtfull_dpath = task.datasubdir('gt' + 'full', scene)
        scene_imfull_dpath = task.datasubdir('im' + 'full', scene)
        gt_fpaths = sorted(glob.glob(join(scene_gtfull_dpath, '*.png')))
        im_fpaths = sorted(glob.glob(join(scene_imfull_dpath, '*.png')))

        # Define the output path for this preprocessing mode
        mode = 'part-scale{}'.format(scale)
        scene_gtout_dpath = task.datasubdir('gt' + mode, scene)
        scene_imout_dpath = task.datasubdir('im' + mode, scene)
        # Start fresh. Remove existing files
        ub.delete(scene_gtout_dpath, verbose=False)
        ub.delete(scene_imout_dpath, verbose=False)
        ub.ensuredir(scene_gtout_dpath)
        ub.ensuredir(scene_imout_dpath)

        for impath, gtpath in zip(im_fpaths, gt_fpaths):
            im = cv2.imread(impath, flags=cv2.IMREAD_UNCHANGED)
            gt = cv2.imread(gtpath, flags=cv2.IMREAD_UNCHANGED)

            if scale != 1.0:
                im = imutil.imscale(im, scale, cv2.INTER_LANCZOS4)[0]
                gt = imutil.imscale(gt, scale, cv2.INTER_NEAREST)[0]

            assert gt.max() <= task.labels.max(), (
                'be careful not to change gt labels')

            fname_we = splitext(basename(impath))[0]
            sl_gen = imutil.image_slices(im.shape[0:2], input_shape, overlap,
                                         keepbound)
            for idx, rc_slice in enumerate(sl_gen):
                # encode the slice in the image name?
                fname = '{}_part{:0=4d}.png'.format(fname_we, idx)
                im_part = im[rc_slice]
                gt_part = gt[rc_slice]

                cv2.imwrite(join(scene_imout_dpath, fname), im_part)
                cv2.imwrite(join(scene_gtout_dpath, fname), gt_part)
        return scene_imout_dpath, scene_gtout_dpath

    def load_predefined_train_test(task):
        """
        the v1-annotations dataset has a predefined train / test split
        The training set can be used however you want (cross-validation wise)
        The test may only be used for final evaluation.
        """
        # TODO: generalize path spec
        # HACK
        train_scenes = sorted(['0000', '0002', '0101', '0102', '0401', '0503'])
        test_scenes = sorted(['0001', '0100', '0400', '0500', '0501', '0502'])
        return train_scenes, test_scenes

        # FIXME
        dpath = expanduser('~/code/baseline-algorithms/DIVA/splits/VIRAT/videos/new_v1_annotation')

        def parse_scene(fname):
            """
            ~~All~~ (most of) the filenames are formatted as follows:
            VIRAT_S_XXYYZZ_KK_SSSSSS_TTTTTT.mp4
            """
            import parse
            # virat_format = 'VIRAT_S_{group:DD}{scene:DD}{seq:DD}_{segmentid}_{start}_{stop}.mp4'
            virat_format = 'VIRAT_S_{group:DD}{scene:DD}{therest}.mp4'
            extra_types = {'DD': parse.with_pattern(r'\d\d')(lambda x: x)}
            result = parse.parse(virat_format, fname, extra_types)
            if result:
                return result.named
        train_scenes = set()
        test_scenes = set()
        for fpath in glob.glob(join(dpath, 'Validation_*')):
            paths = [p for p in ub.readfrom(fpath).split('\n') if p]
            info = [parse_scene(p) for p in paths]
            info = [p for p in info if p]
            scenes = {p['group'] + p['scene'] for p in info}
            train_scenes.update(scenes)

        for fpath in glob.glob(join(dpath, 'test_*')):
            paths = [p for p in ub.readfrom(fpath).split('\n') if p]
            info = [parse_scene(p) for p in paths]
            info = [p for p in info if p]
            scenes = {p['group'] + p['scene'] for p in info}
            test_scenes.update(scenes)
        # Ensure determenism
        train_scenes = sorted(train_scenes)
        test_scenes = sorted(test_scenes)
        return train_scenes, test_scenes

    def _scene_data_subset(task, scenes, keys):
        """
        Loads image/groundtruth paths for specific scenes processed with
        certain modes (e.g. lowres, part-scale1, lowres-aug).

        >>> scenes = task.load_predefined_train_test()[1]
        >>> keys = task.base_modes
        """
        scene_im_paths, scene_gt_paths = task._load_all_scene_paths()
        im_paths = []
        gt_paths = []
        for scene in scenes:
            im_modes = scene_im_paths[scene]
            gt_modes = scene_gt_paths[scene]
            for key in keys:
                im_paths += im_modes[key]
                gt_paths += gt_modes[key]

        return im_paths, gt_paths

    def _all_scene_dpaths(task):
        """
        Returns the directories that the train testing data will exist in
        """
        scene_im_dpaths = ub.AutoDict()
        scene_gt_dpaths = ub.AutoDict()

        keys = task._preprocessing_keys()

        for scene, key in it.product(task.scene_ids, keys):
            im_dpath = task.datasubdir('im' + key, scene)
            gt_dpath = task.datasubdir('gt' + key, scene)
            scene_im_dpaths[scene][key] = im_dpath
            scene_gt_dpaths[scene][key] = gt_dpath

        return scene_im_dpaths, scene_gt_dpaths

    def _startfresh(task):
        scene_im_dpaths, scene_gt_dpaths = task._all_scene_dpaths()
        keys = task._preprocessing_keys()
        for scene, key in it.product(task.scene_ids, keys):
            im_dpath = task.datasubdir('im' + key, scene)
            gt_dpath = task.datasubdir('gt' + key, scene)
            ub.delete(gt_dpath)
            ub.delete(im_dpath)

    def _check_datas(task):
        scene_im_paths, scene_gt_paths = task._load_all_scene_paths()
        keys = task._preprocessing_keys()
        key_to_num = ub.ddict(list)
        for scene, key in it.product(task.scene_ids, keys):
            im_paths = scene_im_paths[scene][key]
            gt_paths = scene_gt_paths[scene][key]
            assert len(im_paths) == len(gt_paths)
            assert len(im_paths) > 0
            key_to_num[key] += [len(im_paths)]

        for key, ns in key_to_num.items():
            ns_set = set(ns)
            if len(ns_set) != 1:
                print('key    = {!r}'.format(key))
                print('ns_set = {!r}'.format(ns_set))
                print('--')

    def _preprocessing_keys(task, augment=True):
        """ keys used to identify how each scene was modified """
        keys = task.base_modes
        if augment and task.enable_augment:
            keys = task.base_modes + [m + '-aug' for m in task.base_modes]
        return keys

    def _load_all_scene_paths(task):
        """
        Parses scene paths into dictionaries that organize it by scenes
        suitable for cross validation.
        """
        scene_im_paths = ub.AutoDict()
        scene_gt_paths = ub.AutoDict()

        keys = task._preprocessing_keys()

        for scene, key in it.product(task.scene_ids, keys):
            im_dpath = task.datasubdir('im' + key, scene)
            gt_dpath = task.datasubdir('gt' + key, scene)

            im_paths = imutil.load_image_paths(im_dpath, ext='.png')
            gt_paths = imutil.load_image_paths(gt_dpath, ext='.png')

            im_paths = list(map(abspath, im_paths))
            gt_paths = list(map(abspath, gt_paths))

            scene_im_paths[scene][key] = im_paths
            scene_gt_paths[scene][key] = gt_paths

        scene_im_paths = scene_im_paths.to_dict()
        scene_gt_paths = scene_gt_paths.to_dict()
        return scene_im_paths, scene_gt_paths

    def draw_data_overlay(task, sl=None):
        """
            >>> from clab.tasks import *
            >>> import clab
            >>> task = DivaV1(clean=2)
            >>> arch = 'segnet_proper'

            >>> # Use external dataset to increase the amount of training data
            >>> tutorial_dir = './SegNet-Tutorial'
            >>> task.extend_data_from(clab.tasks.CamVid(tutorial_dir))
            >>> task.draw_data_overlay()
        """
        keys = task._preprocessing_keys()
        scenes = task.scene_ids[:]
        keys = keys + ['extern']
        for key in ub.ProgIter(keys, label='overlay', verbose=3):
            scene_overlay_dir = task.datasubdir('overlay', key)
            if key == 'extern':
                # HACK
                im_paths = task.extern_train_im_paths
                gt_paths = task.extern_train_gt_paths
            else:
                im_paths, gt_paths = task._scene_data_subset(scenes, [key])
            gt_paths = fnameutil.align_paths(im_paths, gt_paths)

            overlay_fnames = fnameutil.dumpsafe(im_paths)

            if sl is not None:
                im_paths = im_paths[sl]
                gt_paths = gt_paths[sl]
                overlay_fnames = overlay_fnames[sl]

            prog = ub.ProgIter(zip(im_paths, gt_paths, overlay_fnames),
                               length=len(im_paths), label='overlay key={}'.format(key))
            for impath, gtpath, safename in prog:
                # Make a nice visualization
                fpath = join(scene_overlay_dir, safename)
                gt_img = cv2.imread(gtpath, cv2.IMREAD_UNCHANGED)
                im_img = cv2.imread(impath, cv2.IMREAD_UNCHANGED)
                gt_color = task.colorize(gt_img)
                gt_overlay = imutil.overlay_colorized(gt_color, im_img)
                cv2.imwrite(fpath, gt_overlay)

    def xval_splits(task, xval_method='predef', test_keys=None):
        """
        Generate the list of inputs in each test/train split.

        Currently does the simple thing which is train on all training
        data and test on all testing data. Logic exists for leave one out,
        but it is disabled.

        Yields:
            tuple(inputs.Inputs, inputs.Inputs): train / test inputs

        >>> (train_ims, train_gts), train = next(task.xval_splits())
        """
        # Parse the prepared data and prepare to split it into test / train
        task.create_groundtruth(force=False)
        scene_im_paths, scene_gt_paths = task._load_all_scene_paths()

        # Per scene xval generator
        def leave_k_out_xval(k=2):
            for test_scenes in ub.chunks(task.scene_ids, chunksize=k):
                # Simple leave one out
                train_scenes = list(task.scene_ids)
                for test_scene in test_scenes:
                    train_scenes.remove(test_scene)
                print('test_scenes = {!r}'.format(test_scenes))
                print('train_scenes = {!r}'.format(train_scenes))
                yield train_scenes, test_scenes

        def predef_single_xval():
            train_scenes, test_scenes = task.load_predefined_train_test()
            yield train_scenes, test_scenes

        if xval_method == 'predef':
            xval_iter = predef_single_xval()
        elif xval_method == 'k=2':
            xval_iter = leave_k_out_xval(k=2)
        else:
            raise KeyError(xval_method)

        train_keys = task._preprocessing_keys()
        if test_keys is None:
            test_keys = ['lowres']

        # Given a per scene split, map it to a split on an per-image basis
        flatten = it.chain.from_iterable
        for train_scenes, test_scenes in xval_iter:

            train_im_paths = list(flatten(
                [scene_im_paths[s][k]
                 for (s, k) in it.product(train_scenes, train_keys)]
            )) + task.extern_train_im_paths

            train_gt_paths = list(flatten(
                [scene_gt_paths[s][k]
                 for (s, k) in it.product(train_scenes, train_keys)]
            )) + task.extern_train_gt_paths

            test_im_paths = list(flatten(
                [scene_im_paths[s][k]
                 for (s, k) in it.product(test_scenes, test_keys)]
            ))
            test_gt_paths = list(flatten(
                [scene_gt_paths[s][k]
                 for (s, k) in it.product(test_scenes, test_keys)]
            ))

            train = inputs.Inputs.from_paths(train_im_paths, train_gt_paths,
                                             tag='train')
            test = inputs.Inputs.from_paths(test_im_paths, test_gt_paths,
                                            tag='test')

            assert not bool(ub.find_duplicates(test.im_paths))
            assert not bool(ub.find_duplicates(train.im_paths))

            xval_split = (train, test)
            yield xval_split

    def harness_from_xval(task, idx, arch='segnet_proper', pretrained=None,
                          hyperparams={}, xval_method='predef',
                          test_keys=None):
        from clab import harness
        xval_base = abspath(task.exptdir('xval'))
        splits = list(task.xval_splits(xval_method, test_keys=test_keys))
        (train_data, test_data) = splits[idx]

        xval_dpath = ub.ensuredir((xval_base, 'split_{:0=2}'.format(idx)))
        harn = harness.Harness(workdir=xval_dpath, arch=arch)
        harn.train.im_paths = train_data[0]
        harn.train.gt_paths = train_data[1]

        harn.test.im_paths  = test_data[0]
        harn.test.gt_paths  = test_data[1]

        assert len(harn.train.im_paths) > 0
        assert len(harn.train.gt_paths) > 0
        assert len(harn.test.im_paths) > 0
        assert len(harn.test.gt_paths) > 0

        harn.params.update(hyperparams)
        harn.init_pretrained_fpath = pretrained
        return harn

    def run_xval_evaluation(task, arch='segnet_proper', pretrained=None,
                            hyperparams={}, xval_method='predef',
                            test_keys=None, fit=True):
        """
        Writes test/train data files containing the image paths that will be
        used for testing and training as well as the appropriate solvers and
        prediction models.

        Args:
            fit (bool): if False, we will only evaluate models that have been
                trained so far (useful for getting results from existing while a
                model is not done training)

        CommandLine:
            export PYTHONPATH=$HOME/code/fletch/build-py3/install/lib/python3.5/site-packages:$PYTHONPATH
            python -m clab.tasks DivaV1.run_xval_evaluation
            python -m clab.tasks DivaV1.run_xval_evaluation --batch-size=1
            python -m clab.tasks DivaV1.run_xval_evaluation

            rsync -avpr aretha:sseg/sseg-data/xval-solvers ~/sseg/sseg-data/xval-solvers

        Example:
            >>> from clab.tasks import *
            >>> import clab
            >>> task = DivaV1(clean=2)
            >>> arch = 'segnet_proper'
            >>> pretrained =  '/home/local/KHQ/jon.crall/store/segnet-exact/SegNet-Tutorial/Models/Training/segnet_iter_30000.caffemodel'
            >>> pretrained = 'segnet_proper_camvid.caffemodel'
            >>> hyperparams = {'freeze_before': -23, 'max_iter': 10000}
            >>> # Use external dataset to increase the amount of training data
            >>> tutorial_dir = './SegNet-Tutorial'
            >>> task.extend_data_from(clab.tasks.CamVid(tutorial_dir))
            >>> task.run_xval_evaluation()
        """

        from clab import harness
        # from clab import models

        xval_base = abspath(task.exptdir('xval'))

        xval_results = []
        for idx, xval_split in enumerate(task.xval_splits(xval_method,
                                                          test_keys=test_keys)):
            print(ub.color_text('XVAL iteration {}'.format(idx), 'blue'))

            # harn = task.harness_from_xval(idx, arch=arch,
            #                               pretrained=pretrained,
            #                               hyperparams=hyperparams,
            #                               xval_method=xval_method,
            #                               test_keys=test_keys)

            # (train_data, test_data) = xval_split
            xval_dpath = ub.ensuredir((xval_base, 'split_{:0=2}'.format(idx)))

            (train, test) = xval_split
            harn = harness.Harness(workdir=xval_dpath, arch=arch)
            harn.set_inputs(train, test)

            # harn.train.im_paths = train_data[0]
            # harn.train.gt_paths = train_data[1]

            # harn.test.im_paths  = test_data[0]
            # harn.test.gt_paths  = test_data[1]

            # assert len(harn.train.im_paths) > 0
            # assert len(harn.train.gt_paths) > 0
            # assert len(harn.test.im_paths) > 0
            # assert len(harn.test.gt_paths) > 0

            harn.params.update(hyperparams)
            harn.init_pretrained_fpath = pretrained

            harn.test.prepare_images()
            harn.train.prepare_images()

            harn.gpu_num = gpu_util.find_unused_gpu(min_memory=6000)
            print('harn.gpu_num = {!r}'.format(harn.gpu_num))
            if harn.gpu_num is not None:
                avail_mb = gpu_util.gpu_info()[harn.gpu_num]['mem_avail']
                # Estimate how much we can fit in memory
                # TODO: estimate this from the model arch instead.
                # (90 is a mgic num corresponding to segnet_proper)
                harn.train_batch_size = int((avail_mb * 90) // np.prod(task.input_shape))
                harn.train_batch_size = int(harn.train_batch_size)
                if harn.train_batch_size == 0:
                    raise MemoryError('not enough GPU memory to train the model')
            else:
                # not sure what the best to do on CPU is. Probably nothing.
                harn.train_batch_size = 4

            harn.prepare_solver()

            # Check if we can resume a previous training state
            print(ub.color_text('Checking for previous snapshot states', 'blue'))
            previous_states = harn.snapshot_states()
            print('Found {} previous_states'.format(len(previous_states)))

            from clab.backend import iface_caffe as iface
            solver_info = iface.parse_solver_info(harn.solver_fpath)
            prev_iter = 0
            if previous_states:
                prev_state = previous_states[-1]
                prev_iter = iface.snapshot_iterno(prev_state)

            if prev_iter == 0:
                if fit:
                    print(ub.color_text('Starting a fresh training session', 'blue'))
                    # harn.fit()
                    list(harn.fit2())
                else:
                    print(ub.color_text('Would start a fresh training session', 'yellow'))
            elif prev_iter < solver_info['max_iter']:
                if fit:
                    # continue from the previous iteration
                    print(ub.color_text('Resume training from iter {}'.format(prev_iter), 'blue'))
                    # harn.fit(prevstate_fpath=prev_state)
                    list(harn.fit2(prevstate_fpath=prev_state))
                else:
                    print(ub.color_text('Would resume training from iter {}'.format(prev_iter), 'yellow'))
            else:
                print(ub.color_text('Already finished training this model', 'blue'))

            for _ in harn.deploy_trained_for_testing():
                # hack to evaulate while deploying
                harn.evaulate_all()
            xval_results.append(list(harn._test_results_fpaths()))
        return xval_results
