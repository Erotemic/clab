# -*- coding: utf-8 -*-
"""
Plan for Phase II:

    TRAINING PHASE:

        Optimize Network Weights:
            for each model:
                * Read dataset
                * split into train / validation
                * stop once validation loss does not improve after 5-20 epochs
                * record top 3 epochs with lowest loss

        Choose Operating Points / Postprocess Params / Ensemble Weights:

            # Output static probability maps for the final layer
            for each model:
                * for each of the top 3 epochs with the lowest loss
                    * evaluated stiched probabilities on validation dataset
                    * record path to these stiched predictions; tag path with model-id and epoch.

            For each combination of model/epoch-id tags:
                * load predictions and groundtruth into memory
                * run seeded / random search / Baysian hyper param search using
                  the contest criteria as the objective function.
                    * note: the parameters also include the ensemble weights
                * choose the best hyperparm config.
                * run a grid search over ensemble weights
                * choose the best.
            * choose the best over all of these runs.
            * Record these the model weights / ensemble weights / hyperparams / ...

        Test:
            * apply the preprocessor
            * do stiched prediction for all models in the ensemble
            * weight the predicted probabilities
            * transform combined probabilities into predictions using the
              chosen hyperparams
            * return the result

"""
from __future__ import absolute_import, division, print_function, unicode_literals
import torch
import glob
import numpy as np  # NOQA
import ubelt as ub
import os  # NOQA
from os.path import join, splitext, basename  # NOQA
from clab import util
from clab import xpu_device
from clab import models
from clab.util import imutil
from clab.live.urban_metrics import instance_fscore
from clab.fit_harness import get_snapshot


def urban_mapper_eval_dataset(boundary=True, arch=None):
    from clab.live.urban_train import get_task, SSegInputsWrapper
    from clab import preprocess
    task = get_task('urban_mapper_3d', boundary=boundary, arch=arch)
    eval_fullres = task.load_fullres_inputs('testing')
    datadir = ub.ensuredir((task.workdir, 'eval_data'))
    prep = preprocess.Preprocessor(datadir)
    prep.part_config['overlap'] = .75
    eval_part1_scale = prep.make_parts(eval_fullres, scale=1, clear=0)
    # from clab.profiler import profile_onthefly
    # profile_onthefly(prep.make_parts)(eval_fullres, scale=1, clear=0)

    eval_dataset = SSegInputsWrapper(eval_part1_scale, task, colorspace='RGB')
    eval_dataset.with_gt = False
    eval_dataset.inputs.make_dumpsafe_names()
    eval_dataset.fullres = eval_fullres
    eval_dataset.tag = 'eval'

    return eval_dataset


def eval_contest_testset():
    """
    hacked together script to get the testing data and run prediction for submission

    train_dpath = ub.truepath('~/remote/aretha/data/work/urban_mapper2/arch/unet2/train/input_4214-guwsobde/solver_4214-guwsobde_unet2_mmavmuou_tqynysqo_a=1,c=RGB,n_ch=5,n_cl=4')

    # Submission URL
    # https://community.topcoder.com/longcontest/

    CommandLine:
        python -m clab.live.urban_mapper eval_contest_testset --arch=unet2 --combine

    Script:
        >>> eval_contest_testset()

    """
    MODE = 'DENSE'
    MODE = 'UNET6CH'
    if MODE == 'DENSE':
        arch = 'dense_unet'
        train_dpath = ub.truepath(
            '~/remote/aretha/data/work/urban_mapper4/arch/dense_unet/train/input_25800-phpjjsqu/'
            'solver_25800-phpjjsqu_dense_unet_mmavmuou_zeosddyf_a=1,c=RGB,n_ch=6,n_cl=4')
        epoch = 26
        if epoch == 26:
            params = {'mask_thresh': 0.7870, 'min_seed_size': 85.1641, 'min_size': 64.0634, 'seed_thresh': 0.4320}  # .902
            pass
        use_aux_diff = True
        boundary = True
    elif MODE == 'UNET6CH':
        arch = 'unet2'
        train_dpath = ub.truepath(
            '~/remote/aretha/data/work/urban_mapper2/arch/unet2/train/input_25800-hemanvft/'
            'solver_25800-hemanvft_unet2_mmavmuou_stuyuerd_a=1,c=RGB,n_ch=6,n_cl=4')
        # epoch = 34
        # epoch = None
        boundary = True
        use_aux_diff = True
        # params = {'seed_thresh': 0.6573, 'mask_thresh': 0.8338, 'min_seed_size': 25, 'min_size': 38,}
        # params = {'mask_thresh': 0.8367, 'seed_thresh': 0.4549, 'min_seed_size': 97, 'min_size': 33}
        # params = {'mask_thresh': 0.7664, 'seed_thresh': 0.4090, 'min_seed_size': 48, 'min_size': 61}
        # if epoch == 34:
        #     params = {'mask_thresh': 0.8427, 'seed_thresh': 0.4942, 'min_seed_size': 56, 'min_size': 82}  # 0.9091

        # if epoch == 36:
        #     # TODO: FIND CORRECT PARAMS FOR THIS EPOCH
        #     params = {'mask_thresh': 0.8427, 'seed_thresh': 0.4942, 'min_seed_size': 56, 'min_size': 82}

        train_dpath = '/home/local/KHQ/jon.crall/data/work/urban_mapper2/arch/unet2/train/input_52200-fqljkqlk/solver_52200-fqljkqlk_unet2_ybypbjtw_smvuzfkv_a=1,c=RGB,n_ch=6,n_cl=4'
        epoch = 0
        if epoch == 0:
            params = {'mask_thresh': 0.6666, 'min_seed_size': 81, 'min_size': 13, 'seed_thresh': 0.4241}  # bo_best on more data
            params = {'mask_thresh': 0.7870, 'min_seed_size': 85, 'min_size': 64, 'seed_thresh': 0.4320}  # 0.9169 (vali seen in training)

        epoch = 5

        epoch = 9
        if epoch == 9:
            # just guessing
            params = {'mask_thresh': 0.6666, 'min_seed_size': 81, 'min_size': 13, 'seed_thresh': 0.4241}

        train_dpath = '/home/local/KHQ/jon.crall/data/work/urban_mapper2/arch/unet2/train/input_39000-xsldtcgn/solver_39000-xsldtcgn_unet2_ybypbjtw_smvuzfkv_a=1,c=RGB,n_ch=6,n_cl=4'
        epoch = 10
        if epoch == 10:
            params = {'mask_thresh': 0.9000, 'min_seed_size': 100.0000, 'min_size': 100.0000, 'seed_thresh': 0.4000}
    else:
        raise KeyError(MODE)

    load_path = get_snapshot(train_dpath, epoch=epoch)

    eval_dataset = urban_mapper_eval_dataset(boundary=boundary, arch=arch)
    eval_dataset.use_aux_diff = use_aux_diff
    eval_dataset.with_gt = False
    eval_dataset.inputs.make_dumpsafe_names()
    eval_dataset.tag = 'eval'

    pharn = PredictHarness(eval_dataset)
    eval_dataset.center_inputs = pharn.load_normalize_center(train_dpath)
    pharn.hack_dump_path(load_path)

    stitched_dpath = join(pharn.test_dump_dpath, 'stitched')

    prob_paths = glob.glob(join(stitched_dpath, 'probs', '*.h5'))
    prob1_paths = glob.glob(join(stitched_dpath, 'probs1', '*.h5'))
    if len(prob_paths) == 0:
        pharn.load_snapshot(load_path)
        pharn.run()

    stitched_dpaths = [stitched_dpath]
    dump_dpath = pharn.test_dump_dpath
    task = eval_dataset.task
    make_submission_file(task, stitched_dpaths, params, dump_dpath)


def make_submission_file(task, stitched_dpaths, params, dump_dpath, ensemble_weights=None):
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
    ensemble_paths = []
    for dpath in stitched_dpaths:
        paths = {}
        paths['prob'] = glob.glob(join(dpath, 'probs', '*.h5'))
        paths['prob1'] = glob.glob(join(dpath, 'probs1', '*.h5'))
        ensemble_paths.append(paths)

    if ensemble_weights is None:
        assert len(stitched_dpaths) == 1
        ensemble_weights = [1]

    def seeded_predictions(**params):
        # Convert to submission output format
        import tqdm

        params.pop('alpha')

        n_scenes = len(ensemble_paths[0]['prob'])

        for ix in tqdm.tqdm(list(range(n_scenes)), desc='classifying'):
            probs = 0
            probs1 = 0
            for paths, w in zip(ensemble_paths, ensemble_weights):
                path = paths['prob'][ix]
                probs = probs +  w * util.read_arr(paths['prob'][ix])
                probs1 = probs1 + w * util.read_arr(paths['prob1'][ix])

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
        (width, height), runlen = imutil.run_length_encoding(pred)
        lines.append(tile_id)
        lines.append('{},{}'.format(width, height))
        lines.append(','.join(list(map(str, runlen))))

    text = '\n'.join(lines)
    post_idstr = 'seeded_' + util.compact_idstr(params)
    mode = 'prob'
    suffix = '_'.join(dump_dpath.split('/')[-2:]) + '_' + mode + '_' + post_idstr
    fpath = join(dump_dpath, 'urban_mapper_test_pred_' + suffix + '.txt')
    print('fpath = {!r}'.format(fpath))
    ub.writeto(fpath, text)
    print(ub.codeblock(
        '''
        # Execute on remote computer
        cd ~/Dropbox/TopCoder
        rsync aretha:{fpath} .

        # submit here: https://community.topcoder.com/longcontest/
        '''
    ).format(fpath=fpath))

    # Submission URL
    # https://community.topcoder.com/longcontest/
    # https://community.topcoder.com/longcontest/?module=Submit&compid=57607&rd=17007&cd=15282
    """
    Leaderboards:
        https://community.topcoder.com/longcontest/?module=ViewStandings&rd=17007
    """


def eval_internal_testset():
    """
    Working with the testing set (don't submit with this)

    CommandLine:
        python -m clab.live.urban_mapper eval_internal_testset --arch=unet2 \
            --epoch=386 \
            --train-dpath ~/remote/aretha/data/work/urban_mapper2/arch/unet2/train/input_4214-guwsobde/solver_4214-guwsobde_unet2_mmavmuou_tqynysqo_a=1,c=RGB,n_ch=5,n_cl=4

    Ignore:
        import ubelt as ub
        epoch = 100
        train_dpath = ub.truepath('~/remote/aretha/data/work/urban_mapper2/arch/unet2/train/input_4214-guwsobde/solver_4214-guwsobde_unet2_mmavmuou_tqynysqo_a=1,c=RGB,n_ch=5,n_cl=4')

    Script:
        >>> eval_internal_testset()
    """
    MODE = 'DENSE'
    MODE = 'UNET6CH'

    if MODE == 'DENSE':
        arch = 'dense_unet'
        train_dpath = ub.truepath(
            '~/remote/aretha/data/work/urban_mapper4/arch/dense_unet/train/input_25800-phpjjsqu/'
            'solver_25800-phpjjsqu_dense_unet_mmavmuou_zeosddyf_a=1,c=RGB,n_ch=6,n_cl=4')
        epoch = None
        use_aux_diff = True
        epoch = 26

        train_dpath = ub.truepath(
            '~/data/work/urban_mapper4/arch/dense_unet/train/input_52200-qnwuaiqm/'
            'solver_52200-qnwuaiqm_dense_unet_dbszrgef_qzavgexx_a=1,c=RGB,n_ch=6,n_cl=4'
        )
        epoch = None

    elif MODE == 'UNET6CH':
        arch = 'unet2'
        train_dpath = ub.truepath(
            '~/remote/aretha/data/work/urban_mapper2/arch/unet2/train/input_25800-hemanvft/'
            'solver_25800-hemanvft_unet2_mmavmuou_stuyuerd_a=1,c=RGB,n_ch=6,n_cl=4')
        # epoch = 15
        epoch = None
        epoch = 34
        use_aux_diff = True
        epoch = 0
        epoch = 5

        train_dpath = '/home/local/KHQ/jon.crall/data/work/urban_mapper2/arch/unet2/train/input_39000-xsldtcgn/solver_39000-xsldtcgn_unet2_ybypbjtw_smvuzfkv_a=1,c=RGB,n_ch=6,n_cl=4'
        halfcombo = True
        epoch = 10
    else:
        raise KeyError(MODE)

    from clab.live.urban_train import load_task_dataset
    datasets = load_task_dataset('urban_mapper_3d', combine=False, arch=arch,
                                 halfcombo=halfcombo)
    test_dataset = datasets['test']
    test_dataset.use_aux_diff = use_aux_diff
    test_dataset.with_gt = False
    test_dataset.inputs.make_dumpsafe_names()
    test_dataset.tag = 'test'

    load_path = get_snapshot(train_dpath, epoch=epoch)
    print('load_path = {!r}'.format(load_path))

    pharn = PredictHarness(test_dataset)
    test_dataset.center_inputs = pharn.load_normalize_center(train_dpath)
    pharn.hack_dump_path(load_path)
    task = test_dataset.task

    paths = {}
    paths['probs'] = glob.glob(join(pharn.test_dump_dpath, 'stitched', 'probs', '*.h5'))
    paths['probs1'] = glob.glob(join(pharn.test_dump_dpath, 'stitched', 'probs1', '*.h5'))

    if len(paths['probs']) == 0:
        # gpu part
        pharn.load_snapshot(load_path)
        pharn.run()

    paths = {}
    paths['probs'] = glob.glob(join(pharn.test_dump_dpath, 'stitched', 'probs', '*.h5'))
    paths['probs1'] = glob.glob(join(pharn.test_dump_dpath, 'stitched', 'probs1', '*.h5'))
    if False:
        pharn._blend_full_probs(task, 'probs', npy_fpaths=paths['probs'])
        pharn._blend_full_probs(task, 'probs1', npy_fpaths=paths['probs1'])

    # draw_failures()
    seeded_bo = hypersearch_probs(task, paths)
    print('seeded ' + ub.repr2(bo_best(seeded_bo), nl=0, precision=4))
    print(arch)


def bo_best(self):
    return {'max_val': self.Y.max(),
            'max_params': dict(zip(self.keys, self.X[self.Y.argmax()]))}


def check_ensemble():
    model1 = '/home/local/KHQ/jon.crall/data/work/urban_mapper2/test/input_26400-sotwptrx/solver_52200-fqljkqlk_unet2_ybypbjtw_smvuzfkv_a=1,c=RGB,n_ch=6,n_cl=4/_epoch_00000000/stitched'

    # model2 = '/home/local/KHQ/jon.crall/data/work/urban_mapper4/test/input_26400-fgetszbh/solver_25800-phpjjsqu_dense_unet_mmavmuou_zeosddyf_a=1,c=RGB,n_ch=6,n_cl=4/_epoch_00000026/stitched'

    model2 = '/home/local/KHQ/jon.crall/data/work/urban_mapper4/test/input_26400-fgetszbh/solver_52200-qnwuaiqm_dense_unet_dbszrgef_qzavgexx_a=1,c=RGB,n_ch=6,n_cl=4/_epoch_00000028/stitched'

    paths_m1 = {}
    paths_m1['probs'] = glob.glob(join(model1, 'probs', '*.h5'))
    paths_m1['probs1'] = glob.glob(join(model1, 'probs1', '*.h5'))

    paths_m2 = {}
    paths_m2['probs'] = glob.glob(join(model2, 'probs', '*.h5'))
    paths_m2['probs1'] = glob.glob(join(model2, 'probs1', '*.h5'))

    @ub.memoize
    def gt_info_from_path(pred_fpath):
        gtl_fname = ub.augpath(basename(pred_fpath), suffix='_GTL', ext='.tif')
        gti_fname = ub.augpath(basename(pred_fpath), suffix='_GTI', ext='.tif')
        dsm_fname = ub.augpath(basename(pred_fpath), suffix='_DSM', ext='.tif')
        bgr_fname = ub.augpath(basename(pred_fpath), suffix='_RGB', ext='.tif')
        gtl_fpath = join(ub.truepath('~/remote/aretha/data/UrbanMapper3D/training/'), gtl_fname)
        gti_fpath = join(ub.truepath('~/remote/aretha/data/UrbanMapper3D/training/'), gti_fname)
        dsm_fpath = join(ub.truepath('~/remote/aretha/data/UrbanMapper3D/training/'), dsm_fname)
        bgr_fpath = join(ub.truepath('~/remote/aretha/data/UrbanMapper3D/training/'), bgr_fname)

        gti = util.imread(gti_fpath)
        gtl = util.imread(gtl_fpath)
        dsm = util.imread(dsm_fpath)
        bgr = util.imread(bgr_fpath)
        uncertain = (gtl == 65)
        return gti, uncertain, dsm, bgr

    @ub.memoize
    def memo_read_arr(fpath):
        return util.read_arr(fpath)

    def preload():
        import tqdm
        n_paths = len(paths_m2['probs'])
        for ix in tqdm.trange(n_paths, leave=True, desc='preload'):
            gti, uncertain, dsm, bgr = gt_info_from_path(paths_m1['probs'][ix])

            memo_read_arr(paths_m1['probs'][ix])
            memo_read_arr(paths_m1['probs1'][ix])

            memo_read_arr(paths_m2['probs'][ix])
            memo_read_arr(paths_m2['probs1'][ix])

    preload()  # read datas into memory

    def seeded_objective(**params):
        # CONVERT PROBABILITIES TO INSTANCE PREDICTIONS
        import tqdm

        alpha = params.pop('alpha', .88)

        fscores = []
        # params = {'mask_thresh': 0.7664, 'min_seed_size': 48.5327, 'min_size': 61.8757, 'seed_thresh': 0.4090}
        n_paths = len(paths_m2['probs'])
        for ix in tqdm.trange(n_paths, leave=False, desc='eval objective'):

            gti, uncertain, dsm, bgr = gt_info_from_path(paths_m1['probs'][ix])

            probs_m1 = memo_read_arr(paths_m1['probs'][ix])
            probs1_m1 = memo_read_arr(paths_m1['probs1'][ix])

            probs_m2 = memo_read_arr(paths_m2['probs'][ix])
            probs1_m2 = memo_read_arr(paths_m2['probs1'][ix])

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

    seeded_bounds = {
        'mask_thresh': (.4, .9),
        'seed_thresh': (.4, .9),
        'min_seed_size': (0, 100),
        'min_size': (0, 100),
        'alpha': (.80, 1.0),
    }

    from bayes_opt import BayesianOptimization
    seeded_bo = BayesianOptimization(seeded_objective, seeded_bounds)
    import pandas as pd
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
    n_init = 20

    seeded_bo.explore(pd.DataFrame(cand_params).to_dict(orient='list'))

    # Basically just using this package for random search.
    # The BO doesnt seem to help much
    seeded_bo.plog.print_header(initialization=True)
    seeded_bo.init(n_init)
    print('seeded ' + ub.repr2(bo_best(seeded_bo), nl=0, precision=4))

    gp_params = {"alpha": 1e-5, "n_restarts_optimizer": 2}

    n_iter = n_init // 4
    for kappa in [10, 5, 1]:
        seeded_bo.maximize(n_iter=n_iter, acq='ucb', kappa=kappa, **gp_params)

    print('seeded ' + ub.repr2(bo_best(seeded_bo), nl=0, precision=4))
    return seeded_bo


def hypersearch_probs(task, paths):
    prob_paths  = paths['probs']
    prob1_paths = paths['probs1']

    @ub.memoize
    def gt_info_from_path(pred_fpath):
        gtl_fname = ub.augpath(basename(pred_fpath), suffix='_GTL', ext='.tif')
        gti_fname = ub.augpath(basename(pred_fpath), suffix='_GTI', ext='.tif')
        dsm_fname = ub.augpath(basename(pred_fpath), suffix='_DSM', ext='.tif')
        bgr_fname = ub.augpath(basename(pred_fpath), suffix='_RGB', ext='.tif')
        gtl_fpath = join(ub.truepath('~/remote/aretha/data/UrbanMapper3D/training/'), gtl_fname)
        gti_fpath = join(ub.truepath('~/remote/aretha/data/UrbanMapper3D/training/'), gti_fname)
        dsm_fpath = join(ub.truepath('~/remote/aretha/data/UrbanMapper3D/training/'), dsm_fname)
        bgr_fpath = join(ub.truepath('~/remote/aretha/data/UrbanMapper3D/training/'), bgr_fname)

        gti = util.imread(gti_fpath)
        gtl = util.imread(gtl_fpath)
        dsm = util.imread(dsm_fpath)
        bgr = util.imread(bgr_fpath)
        uncertain = (gtl == 65)
        return gti, uncertain, dsm, bgr

    # https://github.com/fmfn/BayesianOptimization
    # https://github.com/fmfn/BayesianOptimization/blob/master/examples/usage.py
    # https://github.com/fmfn/BayesianOptimization/blob/master/examples/exploitation%20vs%20exploration.ipynb
    sub0 = prob_paths
    sub1 = prob1_paths

    # subx = [0, 1, 2, 3, 4, 5]
    # subx = [2, 4, 5, 9, 10, 14, 17, 18, 20, 30, 33, 39, 61, 71, 72, 73, 75, 81, 84]
    # subx = [0, 1]
    # sub0 = list(ub.take(prob_paths, subx))
    # sub1 = list(ub.take(prob1_paths, subx))
    # pip install bayesian-optimization
    from bayes_opt import BayesianOptimization

    @ub.memoize
    def memo_read_arr(fpath):
        return util.read_arr(fpath)

    def preload():
        import tqdm
        n_paths = len(paths['probs'])
        for ix in tqdm.trange(n_paths, leave=True, desc='preload'):
            gti, uncertain, dsm, bgr = gt_info_from_path(paths['probs'][ix])

            memo_read_arr(paths['probs'][ix])
            memo_read_arr(paths['probs1'][ix])

    def seeded_objective(**params):
        # CONVERT PROBABILITIES TO INSTANCE PREDICTIONS
        fscores = []
        import tqdm
        for path, path1 in tqdm.tqdm(list(zip(sub0, sub1)), desc='calc objective', leave=False):
            gti, uncertain, dsm, bgr = gt_info_from_path(path)

            probs = memo_read_arr(path)
            seed_prob = probs[:, :, task.classname_to_id['inner_building']]

            probs1 = memo_read_arr(path1)
            # from clab.live.filters import crf_posterior
            # Doesnt help
            # if use_crf:
            #     probs1 = crf_posterior(bgr[:, :, ::-1], np.log(probs1).transpose(2, 0, 1)).transpose(1, 2, 0)

            mask_prob = probs1[:, :, 1]

            pred = seeded_instance_label_from_probs(seed_prob, mask_prob,
                                                    **params)

            scores = instance_fscore(gti, uncertain, dsm, pred)
            fscore = scores[0]
            fscores.append(fscore)
        mean_fscore = np.mean(fscores)
        return mean_fscore
    # params = {'mask_thresh': 0.7664, 'min_seed_size': 48.5327, 'min_size': 61.8757, 'seed_thresh': 0.4090}

    seeded_bounds = {
        'mask_thresh': (.4, .9),
        'seed_thresh': (.4, .9),
        'min_seed_size': (0, 100),
        'min_size': (0, 100),
    }
    n_init = 10
    seeded_bo = BayesianOptimization(seeded_objective, seeded_bounds)
    import pandas as pd
    cand_params = [
        {'mask_thresh': 0.9000, 'min_seed_size': 100.0000, 'min_size': 100.0000, 'seed_thresh': 0.4000},
        {'mask_thresh': 0.8367, 'seed_thresh': 0.4549, 'min_seed_size': 97, 'min_size': 33},  # 'max_val': 0.8708
        {'mask_thresh': 0.8367, 'min_seed_size': 97.0000, 'min_size': 33.0000, 'seed_thresh': 0.4549},  # max_val': 0.8991
        {'mask_thresh': 0.7664, 'min_seed_size': 48.5327, 'min_size': 61.8757, 'seed_thresh': 0.4090},  # 'max_val': 0.9091}
        {'mask_thresh': 0.6666, 'min_seed_size': 81.5941, 'min_size': 13.2919, 'seed_thresh': 0.4241},  # full dataset 'max_val': 0.9142}

        {'mask_thresh': 0.8, 'seed_thresh': 0.5, 'min_seed_size': 20, 'min_size': 0},
        {'mask_thresh': 0.5, 'seed_thresh': 0.8, 'min_seed_size': 20, 'min_size': 0},
        {'mask_thresh': 0.8338, 'min_seed_size': 25.7651, 'min_size': 38.6179, 'seed_thresh': 0.6573},
        {'mask_thresh': 0.6225, 'min_seed_size': 93.2705, 'min_size': 5, 'seed_thresh': 0.4401},
        {'mask_thresh': 0.7870, 'min_seed_size': 85.1641, 'min_size': 64.0634, 'seed_thresh': 0.4320},
    ]

    # for params in cand_params:
    #     print('---')
    #     print(seeded_objective(**params, use_crf=False))
    #     print(seeded_objective(**params, use_crf=True))

    seeded_bo.explore(pd.DataFrame(cand_params).to_dict(orient='list'))

    # Basically just using this package for random search.
    # The BO doesnt seem to help much (due to int constraints?)
    seeded_bo.plog.print_header(initialization=True)
    seeded_bo.init(n_init)
    print('seeded ' + ub.repr2(bo_best(seeded_bo), nl=0, precision=4))

    gp_params = {"alpha": 1e-5, "n_restarts_optimizer": 2}

    n_iter = n_init // 2
    for kappa in [10, 5, 1]:
        seeded_bo.maximize(n_iter=n_iter, acq='ucb', kappa=kappa, **gp_params)

    print('seeded ' + ub.repr2(bo_best(seeded_bo), nl=0, precision=4))
    return seeded_bo


def draw_failures(task, paths):
    import cv2
    prob_paths  = paths['probs']
    prob1_paths = paths['probs1']

    params = {
        'mask_thresh': 0.8338, 'min_seed_size': 25.7651,
        'min_size': 38.6179, 'seed_thresh': 0.6573
    }

    @ub.memoize
    def gt_info_from_path(pred_fpath):
        gtl_fname = ub.augpath(basename(pred_fpath), suffix='_GTL', ext='.tif')
        gti_fname = ub.augpath(basename(pred_fpath), suffix='_GTI', ext='.tif')
        dsm_fname = ub.augpath(basename(pred_fpath), suffix='_DSM', ext='.tif')
        bgr_fname = ub.augpath(basename(pred_fpath), suffix='_RGB', ext='.tif')
        gtl_fpath = join(ub.truepath('~/remote/aretha/data/UrbanMapper3D/training/'), gtl_fname)
        gti_fpath = join(ub.truepath('~/remote/aretha/data/UrbanMapper3D/training/'), gti_fname)
        dsm_fpath = join(ub.truepath('~/remote/aretha/data/UrbanMapper3D/training/'), dsm_fname)
        bgr_fpath = join(ub.truepath('~/remote/aretha/data/UrbanMapper3D/training/'), bgr_fname)

        gti = util.imread(gti_fpath)
        gtl = util.imread(gtl_fpath)
        dsm = util.imread(dsm_fpath)
        bgr = util.imread(bgr_fpath)
        uncertain = (gtl == 65)
        return gti, uncertain, dsm, bgr

    seed_thresh, mask_thresh, min_seed_size, min_size = ub.take(
        params, 'seed_thresh, mask_thresh, min_seed_size, min_size'.split(', '))
    for ix, (path, path1) in enumerate(ub.ProgIter(list(zip(prob_paths, prob1_paths)))):
        gti, uncertain, dsm, bgr = gt_info_from_path(path)

        probs = util.read_arr(path)
        seed_probs = probs[:, :, task.classname_to_id['inner_building']]
        seed = (seed_probs > seed_thresh).astype(np.uint8)

        probs1 = util.read_arr(path1)
        mask_probs = probs1[:, :, 1]
        mask = (mask_probs > mask_thresh).astype(np.uint8)

        pred = seeded_instance_label(seed, mask,
                                     min_seed_size=min_seed_size,
                                     min_size=min_size)

        scores, infod = instance_fscore(gti, uncertain, dsm, pred, info=True)

        fn_labels = infod['fn']

        # visualize failure cases
        from clab.metrics import CumMovingAve
        ave_scores = CumMovingAve()
        if True:
            from clab.tasks import urban_mapper_3d
            from clab.tasks.urban_mapper_3d import instance_contours, draw_contours

            gtl = (uncertain * 65)
            # tp_assign = infod['tp']
            fp_labels = infod['fp']
            fn_labels = infod['fn']

            fn_contours = list(ub.flatten(ub.take(instance_contours(gti), fn_labels)))
            fp_contours = list(ub.flatten(ub.take(instance_contours(pred), fp_labels)))

            color_probs = util.make_heatmask(mask_probs)
            color_probs[:, :, 3] *= .3
            blend_probs = util.overlay_colorized(color_probs, bgr, keepcolors=False)

            # Draw False Positives and False Negatives with a big thickness
            DEEP_SKY_BLUE_BGR = (255, 191, 0)
            MAGENTA_BGR = (255, 0, 255)
            RED_BGR = (0, 0, 255)
            GREEN_BGR = (0, 255, 0)

            draw_img = blend_probs
            draw_img = draw_contours(draw_img, fp_contours, thickness=6, alpha=.5, color=MAGENTA_BGR)
            draw_img = draw_contours(draw_img, fn_contours, thickness=6, alpha=.5, color=RED_BGR)

            # Overlay GT and Pred contours
            draw_img = urban_mapper_3d.draw_instance_contours(
                draw_img, gti, gtl=gtl, color=GREEN_BGR, thickness=2, alpha=.5)

            draw_img = urban_mapper_3d.draw_instance_contours(
                draw_img, pred, color=DEEP_SKY_BLUE_BGR, thickness=2, alpha=.5)

            ave_scores.update(dict(zip(['fscore', 'precision', 'recall'], scores)))

            text = ub.codeblock(
                '''
                F-score:   {:.4f}
                Precision: {:.4f}
                Recall:    {:.4f}
                '''
            ).format(*scores)

            draw_img = imutil.putMultiLineText(draw_img, text, org=(10, 70),
                                               fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                               fontScale=1.5,
                                               color=GREEN_BGR, thickness=3,
                                               lineType=cv2.LINE_AA)

            out_fpath = ub.augpath(path.replace('/probs/', '/failures/'), ext='.png')
            from os.path import dirname
            ub.ensuredir(dirname(out_fpath))
            imutil.imwrite(out_fpath, draw_img)
            print(ave_scores.average())

    print(ave_scores.average())
    # mean_fscore = np.mean(fscores)
    # print('mean_fscore = {!r}'.format(mean_fscore))


class PredictHarness(object):
    def __init__(pharn, dataset):
        pharn.dataset = dataset
        pharn.xpu = xpu_device.XPU.from_argv()
        pharn.model = None
        pharn.test_dump_dpath = None

    def load_normalize_center(pharn, train_dpath):
        info_dpath = join(train_dpath, 'train_info.json')
        info = util.read_json(info_dpath)
        # TODO: better deserialization
        from clab import transforms
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
        elif snapshot['model_class_name'] == 'SegNet':
            pharn.model = models.SegNet(in_channels=n_channels, n_classes=n_classes)
        elif snapshot['model_class_name'] == 'UNet2':
            from clab.live import unet2
            pharn.model = unet2.UNet2(
                in_channels=n_channels, n_classes=n_classes, n_alt_classes=3,
                nonlinearity='leaky_relu'
            )
        elif snapshot['model_class_name'] == 'DenseUNet':
            from clab.live import unet3
            pharn.model = unet3.DenseUNet(
                in_channels=n_channels, n_classes=n_classes, n_alt_classes=3,
            )
        else:
            raise NotImplementedError(snapshot['model_class_name'])

        pharn.model = pharn.xpu.move(pharn.model)
        pharn.model.load_state_dict(snapshot['model_state_dict'])

    def hack_dump_path(pharn, load_path):
        # HACK
        eval_dpath = ub.ensuredir((pharn.dataset.task.workdir, pharn.dataset.tag, 'input_' + pharn.dataset.input_id))
        subdir = list(ub.take(os.path.splitext(load_path)[0].split('/'), [-3, -1]))
        # base output dump path on the training id string
        pharn.test_dump_dpath = ub.ensuredir((eval_dpath, '/'.join(subdir)))
        print('pharn.test_dump_dpath = {!r}'.format(pharn.test_dump_dpath))

    def _blend_full_probs(pharn, task, mode='probs1', npy_fpaths=None):
        """
        Ignore:
            mode = 'probs1'
            pharn._restitch_type('probs1', blend='avew')

            from clab.profiler import profile_onthefly
            profile_onthefly(pharn._blend_full_probs)(task, mode='probs', npy_fpaths=npy_fpaths)
            profile_onthefly(foo)(npy_fpaths)
        """
        if npy_fpaths is None:
            dpath = join(pharn.test_dump_dpath, 'restiched', mode)
            npy_fpaths = glob.glob(join(dpath, '*.npy'))

        out_dpath = ub.ensuredir((pharn.test_dump_dpath, 'restiched', 'blend_' + mode))

        for fpath in ub.ProgIter(npy_fpaths, label='viz full probs'):

            out_fpath = join(out_dpath, basename(fpath))
            gtl_fname = basename(out_fpath).replace('.npy', '_GTL.tif')
            gti_fname = basename(out_fpath).replace('.npy', '_GTI.tif')
            bgr_fname = basename(out_fpath).replace('.npy', '_RGB.tif')
            gtl_fpath = join(ub.truepath('~/remote/aretha/data/UrbanMapper3D/training/'), gtl_fname)
            gti_fpath = join(ub.truepath('~/remote/aretha/data/UrbanMapper3D/training/'), gti_fname)
            bgr_fpath = join(ub.truepath('~/remote/aretha/data/UrbanMapper3D/training/'), bgr_fname)

            gtl = util.imread(gtl_fpath)
            gti = util.imread(gti_fpath)
            bgr = util.imread(bgr_fpath)
            probs = util.read_arr(fpath)

            # Dump each channel
            from clab.tasks import urban_mapper_3d
            # from clab import profiler
            # _ = profiler.profile_onthefly(urban_mapper_3d.draw_instance_contours)(blend_probs, gti, gtl)

            for c in reversed(range(probs.shape[2])):
                if mode.endswith('1'):
                    # hack
                    name = ['non-building', 'building', 'uncertain'][c]
                else:
                    name = task.classnames[c]

                if name in task.ignore_classnames:
                    continue
                c_dpath = ub.ensuredir(join(out_dpath, 'c{}_{}'.format(c, name)))
                c_fname = ub.augpath(basename(fpath), ext='.png')
                c_fpath = join(c_dpath, c_fname)

                color_probs = util.make_heatmask(probs[:, :, c])[:, :, 0:3]
                blend_probs = util.overlay_colorized(color_probs, bgr, alpha=.3)

                draw_img = urban_mapper_3d.draw_instance_contours(
                    blend_probs, gti, gtl, thickness=2, alpha=.4)

                util.imwrite(c_fpath, draw_img)

    # from clab.profiler import profile_onthefly
    # @profile_onthefly
    def run(pharn):
        print('Preparing to predict {} on {}'.format(pharn.model.__class__.__name__, pharn.xpu))
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

        # map(len, grouped_indices)
        # ub.chunks
        #     loader = torch.utils.data.DataLoader(
        #         pharn.dataset, shuffle=False,
        #         pin_memory=True,
        #         num_workers=0,
        #         batch_size=1,
        #     )

        output_dpath = join(pharn.test_dump_dpath, 'stitched')
        ub.ensuredir(output_dpath)

        # prog = ub.ProgIter(length=len(grouped_indices), label='predict group proba', verbose=3)

        import tqdm
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

                inputs_ = pharn.xpu.variables(*inputs_)
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


def draw_gt_contours2(img, gt, thickness=4, alpha=1):
    import cv2

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


def draw_gt_contours(img, gti, thickness=4):
    import cv2

    rc_locs = np.where(gti > 0)
    grouped_cc_xys = util.group_items(
        np.ascontiguousarray(np.vstack(rc_locs[::-1]).T),
        gti[rc_locs], axis=0
    )
    gt_hulls = ub.map_vals(cv2.convexHull, grouped_cc_xys)
    BGR_GREEN = (0, 255, 0)
    img = util.ensure_float01(img)
    draw_img = np.ascontiguousarray((255 * img[:, :, 0:3]).astype(np.uint8))
    draw_img = cv2.drawContours(
        image=draw_img, contours=list(gt_hulls.values()), contourIdx=-1,
        color=BGR_GREEN, thickness=thickness)
    return draw_img


def draw_with_gt(task, pred, gti, bgr):
    blend_pred = task.instance_colorize(pred, bgr)
    draw_img = draw_gt_contours(blend_pred, gti)
    return draw_img


def unet2_instance_restitch(restitched_pred0, restitched_pred1, task):
    from os.path import dirname
    out_fpaths = []
    # restitched_pred0 = eval_dataset.fullres.align(restitched_pred1)
    for pred1_fpath in ub.ProgIter(restitched_pred1):
        # CUSTOM INSTANCE RESTITCHING
        pred_seg0 = util.imread(pred1_fpath.replace('/pred1/', '/pred/'))
        pred_seg1 = util.imread(pred1_fpath)

        seed = (pred_seg0 == task.classname_to_id['inner_building']).astype(np.uint8)
        mask = (pred_seg1 == 1)
        pred = seeded_instance_label(seed, mask, min_seed_size=75)
        out_fpath = pred1_fpath.replace('/pred1/', '/instance_pred/')

        ub.ensuredir(dirname(out_fpath))
        util.imwrite(out_fpath, (pred > 0).astype(np.uint8))
        out_fpaths.append(out_fpath)
    return out_fpaths


def test_instance_restitch(out_fpaths, task):
    from os.path import dirname, exists
    import cv2
    all_scores = []
    for out_fpath in ub.ProgIter(out_fpaths, freq=1, adjust=False):
        gtl_fname = basename(out_fpath).replace('.png', '_GTL.tif')
        gti_fname = basename(out_fpath).replace('.png', '_GTI.tif')
        dsm_fname = basename(out_fpath).replace('.png', '_DSM.tif')
        bgr_fname = basename(out_fpath).replace('.png', '_RGB.tif')
        gtl_fpath = join(ub.truepath('~/remote/aretha/data/UrbanMapper3D/training/'), gtl_fname)
        gti_fpath = join(ub.truepath('~/remote/aretha/data/UrbanMapper3D/training/'), gti_fname)
        dsm_fpath = join(ub.truepath('~/remote/aretha/data/UrbanMapper3D/training/'), dsm_fname)
        bgr_fpath = join(ub.truepath('~/remote/aretha/data/UrbanMapper3D/training/'), bgr_fname)

        blend_fpath = out_fpath.replace('/instance_pred', '/blend_instance_pred')
        ub.ensuredir(dirname(blend_fpath))

        if exists(gti_fpath):
            bgr = util.imread(bgr_fpath)

            pred = util.imread(out_fpath)

            pred_ccs = cv2.connectedComponents(pred, connectivity=4)[1]
            blend_pred = task.instance_colorize(pred_ccs, bgr)
            gti = util.imread(gti_fpath)
            gtl = util.imread(gtl_fpath)
            dsm = util.imread(dsm_fpath)
            uncertain = (gtl == 65)

            scores, assign = instance_fscore(gti, uncertain, dsm, pred_ccs, info=True)
            all_scores.append(scores)

            blend_pred = draw_gt_contours(blend_pred, gti)
            util.imwrite(blend_fpath, blend_pred)
    return out_fpaths


def mask_instance_label(pred, k=15, n_iters=1, dist_thresh=5,
                        min_size=0, watershed=False):
    import cv2
    mask = pred
    # noise removal
    if k > 1:
        kernel = np.ones((k, k), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel,
                                iterations=n_iters)

    if watershed:
        from clab.live import filters
        mask = filters.watershed_filter(mask, dist_thresh=dist_thresh)

    mask = mask.astype(np.uint8)
    n_ccs, pred_ccs = cv2.connectedComponents(mask, connectivity=4)

    if min_size > 0:
        # Remove small predictions
        for inner_id, inner_rcs in util.cc_locs(pred_ccs).items():
            if len(inner_rcs) < min_size:
                pred_ccs[tuple(inner_rcs.T)] = 0

    return pred_ccs


def seeded_instance_label_from_probs(seed_prob, mask_prob, seed_thresh=.4,
                                     mask_thresh=.6, inner_k=0, outer_k=0,
                                     post_k=0, min_seed_size=0, min_size=0):
    """
    Convert outputs from the sseg network to an instance prediction
    TODO: deep watershed
    """

    seed = (seed_prob > seed_thresh).astype(np.uint8)
    mask = (mask_prob > mask_thresh).astype(np.uint8)

    return seeded_instance_label(seed, mask, inner_k=inner_k, outer_k=outer_k,
                                 post_k=post_k, min_seed_size=min_seed_size,
                                 min_size=min_size)


def seeded_instance_label(seed, mask, inner_k=0, outer_k=0, post_k=0,
                          min_seed_size=0, min_size=0):
    import cv2

    mask = mask.astype(np.uint8)
    seed = seed.astype(np.uint8)

    if inner_k > 0:
        kernel = np.ones((inner_k, inner_k), np.uint8)
        seed = cv2.morphologyEx(seed, cv2.MORPH_OPEN, kernel,
                                 iterations=1)

    if min_seed_size > 0:
        # Remove very small seeds
        for inner_id, inner_rcs in util.cc_locs(seed).items():
            if len(inner_rcs) < min_seed_size:
                seed[tuple(inner_rcs.T)] = 0

    seed_ccs = cv2.connectedComponents(seed, connectivity=4)[1]

    # Remove seeds not surrounded by a mask
    seed[(seed & ~mask)] = 0

    if outer_k > 1:
        mask = cv2.morphologyEx(mask, cv2.MORPH_ERODE,
                                np.ones((outer_k, outer_k), np.uint8),
                                iterations=1)
        # Ensure we dont clobber a seed
        mask[seed.astype(np.bool)] = 1

    dmask1 = cv2.dilate(mask, np.ones((3, 3)))
    dmask2 = cv2.dilate(dmask1, np.ones((3, 3)))

    # Build a topological wall between mask components
    twall = dmask1 - mask

    # Pixels beyond the wall region are sure background
    sure_bg = 1 - dmask2

    # prepare watershed seeds
    # Label sure background as 1
    wseed = sure_bg.astype(np.int)
    # Add the seeds starting at 2
    seed_mask = seed_ccs > 0
    seed_labels = seed_ccs[seed_mask]
    wseed[seed_mask] = seed_labels + 1
    # The unsure region is now labeled as zero

    topology = np.dstack([twall * 255] * 3)
    markers = np.ascontiguousarray(wseed.astype(np.int32).copy())
    markers = cv2.watershed(topology, markers)
    # Remove background and border labels
    markers[markers <= 1] = 0

    instance_mask = (markers > 0).astype(np.uint8)

    if post_k > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_ERODE,
                                np.ones((post_k, post_k), np.uint8),
                                iterations=1)

    pred_ccs = cv2.connectedComponents(instance_mask, connectivity=4)[1]

    if min_size > 0:
        # Remove small predictions
        for inner_id, inner_rcs in util.cc_locs(pred_ccs).items():
            if len(inner_rcs) < min_size:
                pred_ccs[tuple(inner_rcs.T)] = 0
    return pred_ccs


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m clab.live.urban_mapper
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
