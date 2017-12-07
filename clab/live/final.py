from os.path import join, splitext, exists, basename, relpath, dirname  # NOQA
import glob
import itertools as it
import pickle
import numpy as np
import os  # NOQA
import sys
import torch
import torch  # NOQA
import torchvision  # NOQA
import tqdm  # NOQA
import ubelt as ub
from bayes_opt import BayesianOptimization
import pandas as pd
from clab import preprocess
from clab import util
from clab.live import fit_harn2
from clab.live import unet2
from clab.live import unet3
from clab.live.urban_metrics import instance_fscore
from clab.live.urban_pred import seeded_instance_label_from_probs
from clab.tasks.urban_mapper_3d import UrbanMapper3D
from clab.torch import criterions
from clab.torch import hyperparams
from clab.torch import im_loaders
from clab.torch import metrics
from clab.torch import models
from clab.torch import transforms
from clab.torch import xpu_device
from clab.torch.transforms import (ImageCenterScale, DTMCenterScale, ZipTransforms)
from clab.torch.transforms import (RandomWarpAffine, RandomGamma, RandomBlur,)

print = util.protect_print(print)

DEBUG = ub.argflag('--debug')


def package_pretrained_submission():
    """
    Gather the models trained during phase 1 and output them in a format
    useable by the phase 2 solution. Note: remember to put the output folder
    into docker.
    """
    # model1 = '/home/local/KHQ/jon.crall/data/work/urban_mapper2/test/input_26400-sotwptrx/solver_52200-fqljkqlk_unet2_ybypbjtw_smvuzfkv_a=1,c=RGB,n_ch=6,n_cl=4/_epoch_00000000/stitched'

    # model2 = '/home/local/KHQ/jon.crall/data/work/urban_mapper4/test/input_26400-fgetszbh/solver_25800-phpjjsqu_dense_unet_mmavmuou_zeosddyf_a=1,c=RGB,n_ch=6,n_cl=4/_epoch_00000026/stitched'

    # Localize the data

    # epoch
    # info_dpath = join(train_dpath, 'train_info.json')

    # Both of these seem to have the same centering stats
    # unet2_weights = '/home/local/KHQ/jon.crall/data/work/urban_mapper2/arch/unet2/train/input_52200-fqljkqlk/solver_52200-fqljkqlk_unet2_ybypbjtw_smvuzfkv_a=1,c=RGB,n_ch=6,n_cl=4/torch_snapshots/_epoch_00000000.pt'

    # dense_unet_weights = '/home/local/KHQ/jon.crall/remote/aretha/data/work/urban_mapper4/arch/dense_unet/train/input_25800-phpjjsqu/solver_25800-phpjjsqu_dense_unet_mmavmuou_zeosddyf_a=1,c=RGB,n_ch=6,n_cl=4/torch_snapshots/_epoch_00000026.pt'

    max_params = {'alpha': 0.8800, 'mask_thresh': 0.7870,
                  'min_seed_size': 85.1641, 'min_size': 64.0634,
                  'seed_thresh': 0.4320}

    max_epochs = {
        'unet2': 0,
        'dense_unet': 26,
    }

    arch_to_train_dpath = {
        'unet2': ub.truepath(
            '~/data/work/urban_mapper2/arch/unet2/train/input_52200-fqljkqlk/'
            'solver_52200-fqljkqlk_unet2_ybypbjtw_smvuzfkv_a=1,c=RGB,n_ch=6,n_cl=4'),
        'dense_unet': ub.truepath(
            '~/data/work/urban_mapper4/arch/dense_unet/train/input_25800-phpjjsqu/'
            'solver_25800-phpjjsqu_dense_unet_mmavmuou_zeosddyf_a=1,c=RGB,n_ch=6,n_cl=4'),
    }

    arches = ['unet2', 'dense_unet']

    def localize_model(train_dpath, epoch, local_path):
        import shutil
        local_snap_path = ub.ensuredir((local_path, 'torch_snapshots'))
        orig_load_path = fit_harn2.get_snapshot(train_dpath, epoch=epoch)
        orig_info_fpath = join(train_dpath, 'train_info.json')
        local_info_fpath = join(local_path, 'train_info.json')
        local_load_path = join(local_snap_path, basename(orig_load_path))

        shutil.copy2(orig_info_fpath, local_info_fpath)
        shutil.copy2(orig_load_path, local_load_path)

    local_arch_to_train_dpath = {}

    final_dpath = ub.ensuredir(ub.truepath('~/docker/final_model'))
    for arch in arches:
        local_path = ub.ensuredir((final_dpath, arch))
        epoch = max_epochs[arch]
        train_dpath = arch_to_train_dpath[arch]
        localize_model(train_dpath, epoch, local_path)
        rel_local_train_dpath = relpath(local_path, final_dpath)
        local_arch_to_train_dpath[arch] = rel_local_train_dpath

    solution = {
        'max_params': max_params,
        'max_epochs': max_epochs,
        'arch_to_train_dpath': local_arch_to_train_dpath,
        'arches': arches,
    }
    soln_fpath = join(final_dpath, 'trained_soln.pkl')
    print('write trained model/param info to = {!r}'.format(soln_fpath))
    with open(soln_fpath, 'wb') as file:
        file.write(pickle.dumps(solution))


class UrbanDataset(torch.utils.data.Dataset):
    """
    Ignore:
        >>> from clab.live.urban_train import *
        >>> task = get_task('urban_mapper_3d')
        >>> learn, test = next(task.xval_splits())
        >>> inputs = learn
        >>> self = SSegInputsWrapper(inputs, task)
        >>> im, gt = self[0]

        for i in range(len(self)):
            im = self[i]
            dtm = im[:, 3, :, :]
            dsm = im[:, 4, :, :]

        np.bincount(gt.cpu().numpy().ravel())

    """
    def __init__(self, inputs, task):

        self.inputs = inputs
        self.task = task
        self.colorspace = 'RGB'

        self.loader = im_loaders.np_loader
        self.rng = np.random.RandomState(432432)

        inputs_base = ub.ensuredir((task.workdir, 'inputs'))
        inputs.base_dpath = inputs_base
        if len(inputs):
            inputs.prepare_images(force=True)
            inputs.prepare_input()
            self.input_id = inputs.input_id
            self.with_gt = self.inputs.gt_paths
        else:
            self.input_id = ''

        self.augment = None
        self.im_augment = torchvision.transforms.Compose([
            RandomGamma(rng=self.rng),
            RandomBlur(rng=self.rng),
        ])
        self.rand_aff = RandomWarpAffine(self.rng)

        if self.inputs.aux_paths:
            self.aux_keys = sorted(self.inputs.aux_paths.keys())
        else:
            self.aux_keys = []

        self.center_inputs = None
        self.use_aux_diff = True
        self.use_dual_gt = True

        self.epoch_shrink = False

    def _make_normalizer(self):
        transforms = []
        # hack: specific number for DTM
        nan_value = -32767.0
        if len(self.inputs):
            # HACK IN CONSTANT VALS
            im_mean = .5
            im_scale = .75
            print('im_mean = {!r}'.format(im_mean))
            print('im_scale = {!r}'.format(im_scale))

            im_center = ImageCenterScale(im_mean, im_scale)
            transforms.append(im_center)

            if self.aux_keys:
                scale = 4.7431301577290377
                # zero the median on a per-chip basis, but use
                # the global internal_std to normalize extent
                # aux_std =
                print('aux scale = {!r}'.format(scale))
                aux_center = DTMCenterScale(scale, nan_value=nan_value,
                                            fill='median')
                transforms.append(aux_center)

        center_inputs = ZipTransforms(transforms)
        self.center_inputs = center_inputs
        return center_inputs

    def dataset_metadata(self):
        """ metadata that should be saved along with each model snapshot """
        meta = {
            'colorspace': 'RGB',
            'center': [(x.__class__.__name__, x.__getstate__())
                       for x in self.center_inputs.transforms],
            'n_classes': self.n_classes,
            'n_channels': self.n_channels,
            'classnames': self.task.classnames,
            'ignore_label': self.ignore_label,
            # 'class_weights': self.class_weights(),
        }
        return  meta

    def __len__(self):
        if self.epoch_shrink:
            return len(self.inputs) // self.epoch_shrink
        else:
            return len(self.inputs)

    def from_tensor(self, im, gt=None):
        if len(im.shape) == 3:
            im = im.cpu().numpy().transpose(2, 0, 1)
        else:
            im = im.cpu().numpy().transpose(0, 2, 3, 1)
        if gt is not None:
            gt = gt.cpu().numpy()
        return im, gt

    def to_tensor(self, input_tuple, gt):
        # NHWC -> NCHW
        input_tuple = [im_loaders.image_to_float_tensor(data)
                       for data in input_tuple]
        if gt is None:
            gt_tensor = None
        else:
            gt_tensor = im_loaders.label_to_long_tensor(gt)
        return input_tuple, gt_tensor

    def load_inputs(self, index):
        if self.epoch_shrink:
            # Randomly subsample epochs
            offset = int(self.rng.rand() * self.epoch_shrink)
            index = (index * self.epoch_shrink) + offset

        im_fpath = self.inputs.im_paths[index]

        if self.inputs.gt_paths:
            gt_fpath = self.inputs.gt_paths[index]
            gt_hwc = self.loader(gt_fpath, colorspace=None)
        else:
            gt_hwc = None

        # Load in RGB for now, we will convert right before we center the data
        im = self.loader(im_fpath, colorspace='RGB')

        aux_channels = []
        if self.aux_keys:
            aux_paths = [self.inputs.aux_paths[k][index]
                         for k in self.aux_keys]
            aux_channel = np.dstack([
                self.loader(p, colorspace=None)
                for p in aux_paths
            ])
            aux_channels = [aux_channel]

        if self.augment:
            # Image augmentation must be done in RGB
            # Augment intensity independently
            im = self.im_augment(im)
            # Augment geometry consistently
            im, aux_channels, gt_hwc = self.rand_aff.sseg_warp(
                im, aux_channels, gt_hwc)

        # Do centering of inputs
        input_tuple_hwc = [im] + aux_channels
        input_tuple_hwc = self.center_inputs(input_tuple_hwc)

        if self.use_aux_diff:
            # add residual between dtm and dsm
            dtm_dsm = input_tuple_hwc[-1]
            residual = dtm_dsm[:, :, 0:1] - dtm_dsm[:, :, 1:2]
            input_tuple_hwc += [residual]

        # gt_tuple_hwc = [], if gt_hwc is None else [gt_hwc]

        return input_tuple_hwc, gt_hwc

    def __getitem__(self, index):
        """

        Ignore:
            >>> from clab.live.urban_train import *
            >>> from clab.tasks.urban_mapper_3d import UrbanMapper3D
            >>> self = load_task_dataset('urban_mapper_3d')['train']
            >>> self.augment = True
            >>> index = 0
            >>> self.center_inputs = self._make_normalizer()
            >>> self.use_aux_diff = True
            >>> im, gt = self[0]
        """
        input_tuple_hwc, gt_hwc = self.load_inputs(index)
        input_tuple, gt_tensor = self.to_tensor(input_tuple_hwc, gt_hwc)

        data_tensor = torch.cat(input_tuple, dim=0)

        if self.with_gt:
            if self.use_dual_gt:
                mask = gt_tensor >= 2
                gt_tensor_alt = gt_tensor.clone()
                gt_tensor_alt[mask] = gt_tensor_alt[mask] - 1
                labels = [gt_tensor, gt_tensor_alt]
                inputs = [data_tensor]
                return inputs, labels

            return data_tensor, gt_tensor
        else:
            return data_tensor

    @property
    def n_channels(self):
        if self.aux_keys:
            c = 3 + len(self.aux_keys)
        else:
            c = 3
        return c + int(self.use_aux_diff)
        # return c + 1

    @property
    def n_classes(self):
        return int(self.task.labels.max() + 1)

    @property
    def ignore_label(self):
        return self.task.ignore_label

    def class_weights(self):
        """
            >>> from clab.live.urban_train import *
            >>> self = load_task_dataset('urban_mapper_3d')['train']
            >>> self.class_weights()
        """
        # HACK
        class_weights = np.array([ 0.05496113,  0.67041818,  1.96697962,  0. ])
        print('class_weights = {!r}'.format(class_weights))
        print('class_names   = {!r}'.format(self.task.classnames))
        return class_weights

        # # Handle class weights
        # print('prep class weights')
        # gtstats = self.inputs.prepare_gtstats(self.task)
        # gtstats = self.inputs.gtstats
        # # Take class weights (ensure they are in the same order as labels)
        # mfweight_dict = gtstats['mf_weight'].to_dict()
        # class_weights = np.array(list(ub.take(mfweight_dict, self.task.classnames)))
        # class_weights[self.task.ignore_labels] = 0

        # if 'inner-building' in self.task.classnames:
        #     # increase weight of inner building
        #     class_weights[1] *= 2

        # # HACK
        # # class_weights[0] = 1.0
        # # class_weights[1] = 0.7
        # print('class_weights = {!r}'.format(class_weights))
        # print('class_names   = {!r}'.format(self.task.classnames))
        # return class_weights


def _make_scorable_objective(arch_to_paths, arches, train_data_path):
    """
    Makes objective evaluation function that caches loaded data
    """

    @ub.memoize
    def memo_read_arr(fpath):
        return util.read_arr(fpath)

    @ub.memoize
    def gt_info_from_path(pred_fpath):
        # Hack to read UrbanMapper specific data
        gtl_fname = ub.augpath(basename(pred_fpath), suffix='_GTL', ext='.tif')
        gti_fname = ub.augpath(basename(pred_fpath), suffix='_GTI', ext='.tif')
        dsm_fname = ub.augpath(basename(pred_fpath), suffix='_DSM', ext='.tif')
        gtl_fpath = join(train_data_path, gtl_fname)
        gti_fpath = join(train_data_path, gti_fname)
        dsm_fpath = join(train_data_path, dsm_fname)

        gti = util.imread(gti_fpath)
        gtl = util.imread(gtl_fpath)
        dsm = util.imread(dsm_fpath)
        uncertain = (gtl == 65)
        return gti, uncertain, dsm

    # Search for good hyperparams for this config (boost if more than 2)
    def preload():
        n_paths = len(arch_to_paths[arches[0]]['probs'])
        for ix in tqdm.trange(n_paths, leave=True, desc='preload'):
            path = arch_to_paths[arches[0]]['probs'][ix]
            gt_info_from_path(path)

            for arch in arches:
                memo_read_arr(arch_to_paths[arch]['probs'][ix])
                memo_read_arr(arch_to_paths[arch]['probs1'][ix])

    def seeded_objective(**params):
        # CONVERT PROBABILITIES TO INSTANCE PREDICTIONS

        alpha = params.pop('alpha', .88)

        fscores = []
        # params = {'mask_thresh': 0.7664, 'min_seed_size': 48.5327, 'min_size': 61.8757, 'seed_thresh': 0.4090}
        n_paths = len(arch_to_paths[arches[0]]['probs'])
        for ix in tqdm.trange(n_paths, leave=False, desc='eval objective'):
            path = arch_to_paths[arches[0]]['probs'][ix]

            gti, uncertain, dsm = gt_info_from_path(path)

            probs_m1 = memo_read_arr(arch_to_paths['unet2']['probs'][ix])
            probs1_m1 = memo_read_arr(arch_to_paths['unet2']['probs1'][ix])

            probs_m2 = memo_read_arr(arch_to_paths['dense_unet']['probs'][ix])
            probs1_m2 = memo_read_arr(arch_to_paths['dense_unet']['probs1'][ix])

            probs = (alpha * probs_m1 + (1 - alpha) * probs_m2)
            probs1 = (alpha * probs1_m1 + (1 - alpha) * probs1_m2)

            INNER_BUILDING_ID = 1
            BUILDING_ID = 1

            seed_prob = probs[:, :, INNER_BUILDING_ID]
            mask_prob = probs1[:, :, BUILDING_ID]

            pred = seeded_instance_label_from_probs(seed_prob, mask_prob,
                                                    **params)

            scores = instance_fscore(gti, uncertain, dsm, pred)

            fscore = scores[0]
            fscores.append(fscore)
        mean_fscore = np.mean(fscores)
        return mean_fscore
    return preload, seeded_objective


def optimize_postproc_params(arch_to_paths, arches, train_data_path):
    def bo_best(self):
        return {'max_val': self.Y.max(),
                'max_params': dict(zip(self.keys, self.X[self.Y.argmax()]))}

    preload, seeded_objective = _make_scorable_objective(arch_to_paths, arches,
                                                         train_data_path)
    preload()  # read datas into memory

    seeded_bounds = {
        'mask_thresh': (.4, .9),
        'seed_thresh': (.4, .9),
        'min_seed_size': (0, 100),
        'min_size': (0, 100),
        'alpha': (0.0, 1.0),
    }

    seeded_bo = BayesianOptimization(seeded_objective, seeded_bounds)
    cand_params = [
        {'mask_thresh': 0.9000, 'min_seed_size': 100.0000, 'min_size': 100.0000, 'seed_thresh': 0.4000},
        {'mask_thresh': 0.8367, 'seed_thresh': 0.4549, 'min_seed_size': 97, 'min_size': 33},  # 'max_val': 0.8708
        {'mask_thresh': 0.8367, 'min_seed_size': 97.0000, 'min_size': 33.0000, 'seed_thresh': 0.4549},  # max_val': 0.8991
        {'mask_thresh': 0.7664, 'min_seed_size': 48.5327, 'min_size': 61.8757, 'seed_thresh': 0.4090},  # 'max_val': 0.9091}
        {'mask_thresh': 0.6666, 'min_seed_size': 81.5941, 'min_size': 13.2919, 'seed_thresh': 0.4241},  # full dataset 'max_val': 0.9142}
        # {'mask_thresh': 0.8, 'seed_thresh': 0.5, 'min_seed_size': 20, 'min_size': 0},
        # {'mask_thresh': 0.5, 'seed_thresh': 0.8, 'min_seed_size': 20, 'min_size': 0},
        # {'mask_thresh': 0.8338, 'min_seed_size': 25.7651, 'min_size': 38.6179, 'seed_thresh': 0.6573},
        # {'mask_thresh': 0.6225, 'min_seed_size': 93.2705, 'min_size': 5, 'seed_thresh': 0.4401},
        # {'mask_thresh': 0.7870, 'min_seed_size': 85.1641, 'min_size': 64.0634, 'seed_thresh': 0.4320},
    ]
    for p in cand_params:
        p['alpha'] = .88
    n_init = 2 if DEBUG else 40

    seeded_bo.explore(pd.DataFrame(cand_params).to_dict(orient='list'))

    # Basically just using this package for random search.
    # The BO doesnt seem to help much
    seeded_bo.plog.print_header(initialization=True)
    seeded_bo.init(n_init)
    print('seeded ' + ub.repr2(bo_best(seeded_bo), nl=0, precision=4))

    gp_params = {"alpha": 1e-5, "n_restarts_optimizer": 2}

    n_iter = 2 if DEBUG else 10
    for kappa in [10, 5, 1]:
        seeded_bo.maximize(n_iter=n_iter, acq='ucb', kappa=kappa, **gp_params)

    best_res = bo_best(seeded_bo)
    print('seeded ' + ub.repr2(best_res, nl=0, precision=4))

    max_params = best_res['max_params']
    max_value = best_res['max_val']

    # search for a good alpha
    # TODO: improve bayes_opt package to handle this
    for alpha in tqdm.tqdm(np.linspace(0, 1, 50), desc='opt alpha'):
        params = max_params.copy()
        params['alpha'] = alpha
        val = seeded_objective(**params)
        if val > max_value:
            max_value = val
            max_params = params
    return max_value, max_params


class UrbanPredictHarness(object):
    def __init__(pharn, dataset, xpu):
        pharn.dataset = dataset
        pharn.xpu = xpu
        pharn.model = None
        pharn.test_dump_dpath = None

    def load_normalize_center(pharn, train_dpath):
        info_dpath = join(train_dpath, 'train_info.json')
        info = util.read_json(info_dpath)
        # TODO: better deserialization
        transform_list = []
        for tup in info['hack_centers']:
            classname = tup[0]
            state = tup[1]
            cls = getattr(transforms, classname, None)
            transform_list.append(cls(**state))
        centering = transforms.ZipTransforms(transform_list)
        return centering

    def load_snapshot(pharn, load_path):
        print('Loading snapshot onto {}'.format(pharn.xpu))
        snapshot = torch.load(load_path, map_location=pharn.xpu.map_location())

        if 'model_kw' not in snapshot:
            # FIXME: we should be able to get information from the snapshot
            print('warning snapshot not saved with modelkw')
            n_classes = pharn.dataset.n_classes
            n_channels = pharn.dataset.n_channels

        # Infer which model this belongs to
        # FIXME: The model must be constructed with the EXACT same kwargs This
        # will be easier when onnx supports model serialization.
        if snapshot['model_class_name'] == 'UNet':
            pharn.model = models.UNet(in_channels=n_channels,
                                      n_classes=n_classes,
                                      nonlinearity='leaky_relu')
        elif snapshot['model_class_name'] == 'UNet2':
            pharn.model = unet2.UNet2(
                in_channels=n_channels, n_classes=n_classes, n_alt_classes=3,
                nonlinearity='leaky_relu'
            )
        elif snapshot['model_class_name'] == 'DenseUNet':
            pharn.model = unet3.DenseUNet(
                in_channels=n_channels, n_classes=n_classes, n_alt_classes=3,
            )
        else:
            raise NotImplementedError(snapshot['model_class_name'])

        pharn.model = pharn.xpu.to_xpu(pharn.model)
        pharn.model.load_state_dict(snapshot['model_state_dict'])

    def run(pharn):
        print('Preparing to predict {} on {}'.format(
            pharn.model.__class__.__name__, pharn.xpu))
        pharn.model.train(False)

        # Hack in the restitching here to not have to deal with expensive IO
        def _extract_part_grid(paths):
            # hack to use filenames to extract upper left locations of tiles in
            # the larger image.
            rc_locs = [[int(x) for x in basename(p).split('.')[0].split('_')[-2:]]
                       for p in paths]
            return rc_locs

        def stitch_tiles_avew(rc_locs, tiles):
            """
            Recombine parts back into an entire image

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

            # assume all shapes are the same
            h, w = shapes[0]
            weight = np.ones((h, w) )
            # Weight borders less than center
            # should really use receptive fields for this calculation
            # but this should be fine.
            weight[:h // 4]  = .25
            weight[-h // 4:] = .25
            weight[:w // 4]  = .25
            weight[-w // 4:] = .25
            weight3c = weight
            if n_channels > 1:
                weight3c = weight[:, :, None]

            # Assume we are not in log-space here, so the weighted average
            # formula does not need any exponentiation.
            for bbox, tile in zip(bboxes, tiles):
                r1, c1, r2, c2 = bbox
                sums[r1:r2, c1:c2] += (tile * weight3c)
                nums[r1:r2, c1:c2] += weight

            if len(sums.shape) == 2:
                stiched = sums / nums
            else:
                stiched = sums / nums[:, :, None]
            return stiched

        groupids = [basename(p).split('_part')[0]
                    for p in pharn.dataset.inputs.dump_im_names]
        grouped_indices = ub.group_items(range(len(groupids)), groupids)

        output_dpath = join(pharn.test_dump_dpath, 'stitched')
        ub.ensuredir(output_dpath)

        for key, groupxs in tqdm.tqdm(grouped_indices.items(), desc='predict group proba'):

            grouped_probs = ub.odict()
            grouped_probs[''] = []
            grouped_probs['1'] = []

            for ix in tqdm.tqdm(groupxs, desc='pred'):

                if pharn.dataset.with_gt:
                    inputs_ = pharn.dataset[ix][0]
                else:
                    inputs_ = pharn.dataset[ix]
                inputs_ = inputs_[None, ...]

                if not isinstance(inputs_, (list, tuple)):
                    inputs_ = [inputs_]

                inputs_ = pharn.xpu.to_xpu_var(*inputs_)
                outputs = pharn.model.forward(inputs_)

                for ox in range(len(outputs)):
                    suffix = '' if ox == 0 else str(ox)

                    output_tensor = outputs[ox]
                    log_prob_tensor = torch.nn.functional.log_softmax(output_tensor, dim=1)[0]
                    prob_tensor = torch.exp(log_prob_tensor)
                    probs = np.ascontiguousarray(prob_tensor.data.cpu().numpy().transpose(1, 2, 0))

                    grouped_probs[suffix].append(probs)

            rc_locs = _extract_part_grid(ub.take(pharn.dataset.inputs.dump_im_names, groupxs))
            for suffix, tiles in grouped_probs.items():
                # from clab.profiler import profile_onthefly
                # profile_onthefly(stitch_tiles_ave)(rc_locs, tiles, weighted=True)
                stitched = stitch_tiles_avew(rc_locs, tiles)

                dpath = ub.ensuredir(join(output_dpath, 'probs' + suffix))
                fpath = join(dpath, key + '.h5')
                util.write_h5arr(fpath, stitched)


def stitched_predictions(dataset, arches, xpu, arch_to_train_dpath, workdir,
                         _epochs, tag):

    dataset.inputs.input_id
    print('dataset.inputs.input_id = {!r}'.format(dataset.inputs.input_id))

    # Predict probabilities for each model in the ensemble
    arch_to_paths = {}
    for arch in arches:
        train_dpath = arch_to_train_dpath[arch]
        epoch = _epochs[arch]
        load_path = fit_harn2.get_snapshot(train_dpath, epoch=epoch)

        pharn = UrbanPredictHarness(dataset, xpu)
        dataset.center_inputs = pharn.load_normalize_center(train_dpath)

        pharn.test_dump_dpath = ub.ensuredir((workdir, tag,
                                              dataset.inputs.input_id, arch,
                                              'epoch{}'.format(epoch)))

        stitched_dpath = join(pharn.test_dump_dpath, 'stitched')

        cfgstr = util.hash_data([
            # depend on both the inputs and the exact model specs
            dataset.inputs.input_id,
            util.hash_file(load_path)
        ])

        # predict the whole scene
        cacher = ub.Cacher('prediction_stamp', cfgstr=cfgstr, dpath=stitched_dpath)
        if cacher.tryload() is None:
            # Only execute this if we haven't done so
            pharn.load_snapshot(load_path)
            pharn.run()
            cacher.save(True)

        paths = {
            'probs': glob.glob(join(stitched_dpath, 'probs', '*.h5')),
            'probs1': glob.glob(join(stitched_dpath, 'probs1', '*.h5')),
        }
        arch_to_paths[arch] = paths
    return arch_to_paths


def write_submission_file(arch_to_paths, params, output_file, arches,
                          ensemble_weights=None):
    """
    TODO: take in multiple prediction paths for an enemble

    # Ensemble these two predictions together
    params = {}
    params = {'alpha': 0.8800, 'mask_thresh': 0.7870, 'min_seed_size': 85.1641, 'min_size': 64.0634, 'seed_thresh': 0.4320}

    alpha = params.get('alpha')

    stitched_dpath1 = '/home/local/KHQ/jon.crall/data/work/urban_mapper2/eval/input_18600-vvhcfmyo/solver_52200-fqljkqlk_unet2_ybypbjtw_smvuzfkv_a=1,c=RGB,n_ch=6,n_cl=4/_epoch_00000000/stitched'

    stitched_dpath2 = '/home/local/KHQ/jon.crall/data/work/urban_mapper4/eval/input_18600-vkmqiooh/solver_25800-phpjjsqu_dense_unet_mmavmuou_zeosddyf_a=1,c=RGB,n_ch=6,n_cl=4/_epoch_00000026/stitched'

    dump_dpath = ub.ensuredir('/home/local/KHQ/jon.crall/data/work/urban_mapper_ensemble')
    stitched_dpaths = [stitched_dpath1, stitched_dpath2]
    ensemble_weights = [alpha, 1 - alpha]
    """

    # prob_paths = pharn._restitch_type('probs', blend='avew', force=False)
    # prob1_paths = pharn._restitch_type('probs1', blend='avew', force=False)
    ensemble_paths = list(ub.take(arch_to_paths, arches))

    if ensemble_weights is None:
        assert len(ensemble_paths[0]) == 1
        ensemble_weights = [1]

    def seeded_predictions(**params):
        # Convert to submission output format
        n_scenes = len(ensemble_paths[0]['probs'])

        for ix in tqdm.tqdm(list(range(n_scenes)), desc='classifying'):
            probs = 0
            probs1 = 0
            for paths, w in zip(ensemble_paths, ensemble_weights):
                path = paths['probs'][ix]
                probs = probs +  w * util.read_arr(paths['probs'][ix])
                probs1 = probs1 + w * util.read_arr(paths['probs1'][ix])

            INNER_BUILDING_ID = 1
            BUILDING_ID = 1

            seed_prob = probs[:, :, INNER_BUILDING_ID]
            mask_prob = probs1[:, :, BUILDING_ID]

            pred = seeded_instance_label_from_probs(seed_prob, mask_prob,
                                                    **params)

            tile_id = splitext(basename(path))[0]
            yield tile_id, pred

    lines = []
    for tile_id, pred in seeded_predictions(**params):
        (width, height), runlen = util.run_length_encoding(pred)
        lines.append(tile_id)
        lines.append('{},{}'.format(width, height))
        lines.append(','.join(list(map(str, runlen))))

    text = '\n'.join(lines)
    print('output_file = {!r}'.format(output_file))
    fpath = output_file + '.txt'
    ub.writeto(fpath, text)


def load_training_datasets(train_data_path, workdir):
    """
    Loads a dataset with groundtruth and splits it into train / validation
    """
    task = UrbanMapper3D(root=train_data_path, workdir=workdir, boundary=True)

    fullres = task.load_fullres_inputs('.')
    fullres = task.create_boundary_groundtruth(fullres)
    del fullres.paths['gti']

    rng = np.random.RandomState(0)
    idxs = np.arange(len(fullres))
    rng.shuffle(idxs)

    vali_frac = .15
    n_vali = int(len(idxs) * vali_frac)

    train_idx = idxs[0:-n_vali]
    vali_idx = idxs[-n_vali:]

    if DEBUG:
        train_idx = train_idx[0:5]
        vali_idx = vali_idx[0:5]

    train_fullres_inputs = fullres.take(train_idx)
    vali_fullres_inputs = fullres.take(vali_idx)
    # take doesnt take the dump_im_names
    train_fullres_inputs.dump_im_names = list(ub.take(fullres.dump_im_names, train_idx))
    vali_fullres_inputs.dump_im_names = list(ub.take(fullres.dump_im_names, vali_idx))

    prep = preprocess.Preprocessor(ub.ensuredir((task.workdir, 'data_train1')))
    prep.part_config['overlap'] = 0 if DEBUG else .75
    prep.ignore_label = task.ignore_label
    train_part_inputs = prep.make_parts(train_fullres_inputs, scale=1, clear=0)

    prep = preprocess.Preprocessor(ub.ensuredir((task.workdir, 'data_vali1')))
    prep.part_config['overlap'] = 0 if DEBUG else .75
    prep.ignore_label = task.ignore_label
    vali_part_inputs2 = prep.make_parts(vali_fullres_inputs, scale=1, clear=0)

    if DEBUG:
        vali_part_inputs = vali_part_inputs2
    else:
        # Make two versions of vali, one with 75% overlap for stiched
        # prediction and another with 0 overlap, for loss validation
        prep = preprocess.Preprocessor(ub.ensuredir((task.workdir, 'data_vali2')))
        prep.part_config['overlap'] = 0
        prep.ignore_label = task.ignore_label
        vali_part_inputs = prep.make_parts(vali_fullres_inputs, scale=1, clear=0)

    train_dataset = UrbanDataset(train_part_inputs, task)
    vali_dataset = UrbanDataset(vali_part_inputs, task)
    vali_dataset2 = UrbanDataset(vali_part_inputs2, task)

    # Shrink epochs by a factor of 16 for more frequent progress
    train_dataset.epoch_shrink = 16

    print('* len(train_dataset) = {}'.format(len(train_dataset)))
    print('* len(vali_dataset) = {}'.format(len(vali_dataset)))
    print('* len(vali_dataset2) = {}'.format(len(vali_dataset2)))
    datasets = {
        'train': train_dataset,
        'vali': vali_dataset,
    }

    datasets['train'].center_inputs = datasets['train']._make_normalizer()
    datasets['vali'].center_inputs = datasets['train'].center_inputs
    vali_dataset2.center_inputs = datasets['train'].center_inputs

    datasets['train'].augment = True
    return datasets, vali_dataset2


def load_testing_dataset(test_data_path, workdir):
    """
    Loads a dataset without groundtruth
    """
    task = UrbanMapper3D(root=test_data_path, workdir=workdir, boundary=True)
    test_fullres = task.load_fullres_inputs('.')

    if DEBUG:
        test_fullres = test_fullres.take([0, 1, 2, 3, 4, 5])

    prep = preprocess.Preprocessor(ub.ensuredir((task.workdir, 'data_test')))
    prep.part_config['overlap'] = 0 if DEBUG else .75
    prep.ignore_label = task.ignore_label
    test_part_inputs = prep.make_parts(test_fullres, scale=1, clear=0)

    test_dataset = UrbanDataset(test_part_inputs, task)

    test_dataset.inputs.make_dumpsafe_names()
    test_dataset.with_gt = False
    return test_dataset


def fit_networks(datasets, xpu):
    print('datasets = {}'.format(datasets))
    n_classes = datasets['train'].n_classes
    n_channels = datasets['train'].n_channels
    class_weights = datasets['train'].class_weights()
    ignore_label = datasets['train'].ignore_label

    print('n_classes = {!r}'.format(n_classes))
    print('n_channels = {!r}'.format(n_channels))

    arches = [
        'unet2',
        'dense_unet',
    ]

    arch_to_train_dpath = {}
    arch_to_best_epochs = {}

    for arch in arches:

        hyper = hyperparams.HyperParams(
            criterion=(criterions.CrossEntropyLoss2D, {
                'ignore_label': ignore_label,
                # TODO: weight should be a FloatTensor
                'weight': class_weights,
            }),
            optimizer=(torch.optim.SGD, {
                # 'weight_decay': .0006,
                'weight_decay': .0005,
                'momentum': .9,
                'nesterov': True,
            }),
            scheduler=('Exponential', {
                'gamma': 0.99,
                'base_lr': 0.001,
                'stepsize': 2,
            }),
            other={
                'n_classes': n_classes,
                'n_channels': n_channels,
                'augment': datasets['train'].augment,
                'colorspace': datasets['train'].colorspace,
            }
        )

        train_dpath = ub.ensuredir((datasets['train'].task.workdir, 'train', arch))

        train_info =  {
            'arch': arch,
            'train_id': datasets['train'].input_id,
            'train_hyper_id': hyper.hyper_id(),
            'colorspace': datasets['train'].colorspace,
            # Hack in centering information
            'hack_centers': [
                (t.__class__.__name__, t.__getstate__())
                for t in datasets['train'].center_inputs.transforms
            ]
        }
        util.write_json(join(train_dpath, 'train_info.json'), train_info)

        arch_to_train_dpath[arch] = train_dpath

        if arch == 'unet2':
            batch_size = 14
            model = unet2.UNet2(n_alt_classes=3, in_channels=n_channels,
                                n_classes=n_classes, nonlinearity='leaky_relu')
        elif arch == 'dense_unet':
            batch_size = 6
            model = unet3.DenseUNet(n_alt_classes=3, in_channels=n_channels,
                                    n_classes=n_classes)

        dry = 0
        harn = fit_harn2.FitHarness(
            model=model, hyper=hyper, datasets=datasets, xpu=xpu,
            train_dpath=train_dpath, dry=dry,
            batch_size=batch_size,
        )
        harn.criterion2 = criterions.CrossEntropyLoss2D(
            weight=torch.FloatTensor([0.1, 1.0, 0.0]),
            ignore_label=2
        )
        if DEBUG:
            harn.config['max_iter'] = 2
        else:
            # Note on aretha we can do 140 epochs in 7 days, so
            # be careful with how long we take to train.
            # With a reduction of 16, we can take a few more epochs
            # Unet2 take ~10 minutes to get through one
            harn.config['max_iter'] = 432  # 3 days max
        harn.early_stop.patience = 10

        def compute_loss(harn, outputs, labels):

            output1, output2 = outputs
            label1, label2 = labels

            # Compute the loss
            loss1 = harn.criterion(output1, label1)
            loss2 = harn.criterion2(output2, label2)
            loss = (.45 * loss1 + .55 * loss2)
            return loss

        harn.compute_loss = compute_loss

        def custom_metrics(harn, output, label):
            ignore_label = datasets['train'].ignore_label
            labels = datasets['train'].task.labels

            metrics_dict = metrics._sseg_metrics(output[1], label[1],
                                                 labels=labels,
                                                 ignore_label=ignore_label)
            return metrics_dict

        harn.add_metric_hook(custom_metrics)

        harn.run()
        arch_to_best_epochs[arch] = harn.early_stop.best_epochs()

    # Select model and hyperparams
    print('arch_to_train_dpath = {}'.format(ub.repr2(arch_to_train_dpath, nl=1)))
    print('arch_to_best_epochs = {}'.format(ub.repr2(arch_to_best_epochs, nl=1)))
    return arch_to_train_dpath, arch_to_best_epochs


def train(train_data_path):
    """
    Fits multiple networks and learns postprocessing parameters.
    Writes a single file that contains paths to the best model snapshots and
    postprocessing parameters to the GLOBAL work directory.
    """
    workdir = script_workdir()
    datasets, vali_dataset2 = load_training_datasets(train_data_path, workdir)
    print('datasets = {!r}'.format(datasets))

    xpu = xpu_device.XPU.from_argv()

    arch_to_train_dpath, arch_to_best_epochs = fit_networks(datasets, xpu)

    arches = ['unet2', 'dense_unet']

    vali_dataset2.inputs.make_dumpsafe_names()
    vali_dataset2.with_gt = False

    max_value = None
    max_params = None
    max_epochs = None

    # Find the model that performs the best on the real objective
    cand_epochs = [arch_to_best_epochs[a][0:1] for a in arches]
    for epoch_combo in it.product(*cand_epochs):
        _epochs = dict(zip(arches, epoch_combo))

        # Predict probabilities for each model in the ensemble
        arch_to_paths = stitched_predictions(vali_dataset2, arches, xpu,
                                             arch_to_train_dpath, workdir,
                                             _epochs, 'vali')

        # Find the right hyper-params
        value, params = optimize_postproc_params(arch_to_paths, arches, train_data_path)

        if max_value is None or max_value < value:
            max_value = value
            max_params = params
            max_epochs = _epochs
            print('max_epochs = {!r}'.format(max_epochs))
            print('max_value = {!r}'.format(max_value))
            print('max_params = {!r}'.format(max_params))

    solution = {
        'max_params': max_params,
        'max_epochs': max_epochs,
        'arch_to_train_dpath': arch_to_train_dpath,
        'arches': arches,
    }
    soln_fpath = join(workdir, 'trained_soln.pkl')
    print('write trained model/param info to = {!r}'.format(soln_fpath))
    with open(soln_fpath, 'wb') as file:
        file.write(pickle.dumps(solution))


def test(train_data_path, test_data_path, output_file):
    """

    train_data_path
    test_data_path
    output_file
    """
    use_ots = ub.argflag('--use_ots')
    vali_check = ub.argflag('--vali_check')

    workdir = script_workdir()

    if vali_check:
        # Check the the test script works on the validation dataset
        _, vali_dataset2 = load_training_datasets(train_data_path, workdir)
        vali_dataset2.inputs.make_dumpsafe_names()
        vali_dataset2.with_gt = False
        test_dataset = vali_dataset2
    else:
        # Do the validation on the real test data
        test_dataset = load_testing_dataset(test_data_path, workdir)

    # Check if we have a locally trained model
    soln_fpath = join(workdir, 'trained_soln.pkl')
    if not exists(soln_fpath):
        # use the pretrained off the shelf model if we dont have a trained
        # model
        use_ots = True

    if use_ots:
        # use pretrained
        soln_fpath = ub.truepath('~/docker/final_model/trained_soln.pkl')
        assert exists(soln_fpath)

    with open(soln_fpath, 'rb') as file:
        solution = pickle.load(file)

    # load paths relative to the soln file
    arch_to_train_dpath = ub.map_vals(
        lambda p: join(dirname(soln_fpath), p),
        solution['arch_to_train_dpath']
    )
    max_epochs = solution['max_epochs']
    max_params = solution['max_params']
    arches = solution['arches']

    xpu = xpu_device.XPU.from_argv()

    arch_to_paths = stitched_predictions(test_dataset, arches, xpu,
                                         arch_to_train_dpath, workdir,
                                         max_epochs, 'eval')

    if vali_check:
        preload, objective =  _make_scorable_objective(arch_to_paths, arches,
                                                       train_data_path)
        score = objective(**max_params)
        print('checked score = {!r}'.format(score))

    alpha = max_params.pop('alpha')
    ensemble_weights = [alpha, 1 - alpha]
    params = max_params

    write_submission_file(arch_to_paths, params, output_file, arches,
                          ensemble_weights)


def script_workdir():
    if DEBUG:
        workdir = ub.ensuredir(ub.truepath('~/data/work_phase2_debug'))
    else:
        workdir = ub.ensuredir(ub.truepath('~/data/work_phase2'))
    return workdir


def main():
    train_data_path = ub.truepath('~/remote/aretha/data/UrbanMapper3D/training')
    test_data_path = ub.truepath('~/remote/aretha/data/UrbanMapper3D/testing')
    output_file = 'prediction'

    train_data_path = ub.argval('--train_data_path', default=train_data_path)
    test_data_path = ub.argval('--test_data_path', default=test_data_path)
    output_file = ub.argval('--output_file', default=output_file)

    if sys.argv[1] == 'train':
        train(train_data_path)

    if sys.argv[1] == 'test':
        test(train_data_path, test_data_path, output_file)


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m clab.live.final train

        python -m clab.live.final test --use_ots --vali_check

        # After setup, run docker via
        nvidia-docker run -v ~/data:/data -it urban3d

        # On Docker
        python3 -m clab.live.final train --train_data_path=/data/UrbanMapper3D/training
        python3 -m clab.live.final test --train_data_path=/data/UrbanMapper3D/training --test_data_path=/data/UrbanMapper3D/testing --output_file=outfile
    """
    main()
