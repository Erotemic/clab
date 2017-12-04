

def opt_crf():
    from clab.torch.urban_pred import get_snapshot, urban_mapper_eval_dataset, PredictHarness  # NOQA
    from clab.torch.sseg_train import task_datasets, get_task, SSegInputsWrapper  # NOQA
    from clab import util
    import ubelt as ub

    # train_dpath = ub.truepath(
    #     '~/remote/aretha/data/work/urban_mapper/arch/unet/train/input_4214-yxalqwdk/solver_4214-yxalqwdk_unet_vgg_nttxoagf_a=1,n_ch=5,n_cl=3')
    # load_path = get_snapshot(train_dpath, epoch=202)

    datasets = task_datasets(get_task('urban_mapper_3d'))
    test_dataset = datasets['test']
    test_dataset.with_gt = False
    test_dataset.inputs.make_dumpsafe_names()
    test_dataset.center_inputs = test_dataset._original_urban_mapper_normalizer()
    test_dataset.tag = 'test'

    prob_folder = ub.truepath(
        '~/remote/aretha/data/work/urban_mapper/test/input_4224-rwyxarza/solver_4214-yxalqwdk_unet_vgg_nttxoagf_a=1,n_ch=5,n_cl=3/_epoch_00000202/log_probs')
    import glob

    subset = slice(300, 310)
    prob_paths = test_dataset.inputs.align(glob.glob(prob_folder + '/*.npz'))[subset]
    gt_paths = test_dataset.inputs.gt_paths[subset]
    im_paths = test_dataset.inputs.im_paths[subset]

    import numpy as np

    imgs = [util.imread(p) for p in ub.ProgIter(im_paths)]
    probs = [np.load(p)['arr_0'] for p in ub.ProgIter(prob_paths)]
    gts = [util.imread(p) for p in ub.ProgIter(gt_paths)]

    from .torch import filters
    # true = gts[4]

    import optml
    class_median_weights = test_dataset.class_weights()
    class_weights = class_median_weights / class_median_weights.sum()

    class CRFModel(optml.models.Model):
        __model_module__ = 'sklearn'  # hack
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def get_params(self, deep=False):
            return self.kwargs

        def fit(self, X, y=None):
            pass

        def predict(self, X):
            return [filters.crf_posterior(imgs[i], probs[i], **self.kwargs).argmax(axis=0) for i in ub.ProgIter(X, label='predicting')]

    def clf_score(y_true, y_pred):
        from .metrics import confusion_matrix, jaccard_score_from_confusion  # NOQA

        cfsn = np.zeros((3, 3))
        for i, pred in zip(y_true, y_pred):
            true = gts[i]
            cfsn += confusion_matrix(true.ravel(), pred.ravel(), [0, 1, 2])

        ious = jaccard_score_from_confusion(cfsn)
        weighted_miou = ((ious * class_weights)[0:2]).sum()
        return weighted_miou

    from optml.bayesian_optimizer import BayesianOptimizer

    model = CRFModel()

    params = [
        # w1 = kwargs.get('w1', 4)
        # w2 = kwargs.get('w2', 3)
        # sigma_alpha = kwargs.get('sigma_alpha', 100)
        # sigma_beta  = kwargs.get('sigma_beta', 3)
        # sigma_gamma = kwargs.get('sigma_gamma', 3)
        # n_iters = kwargs.get('n_iters', 10)

        optml.Parameter(name='w1',          param_type='integer', lower=1, upper=100),
        optml.Parameter(name='sigma_alpha', param_type='integer', lower=1, upper=150),
        optml.Parameter(name='sigma_beta',  param_type='integer', lower=1, upper=150),

        optml.Parameter(name='w2',          param_type='integer', lower=3, upper=3),
        optml.Parameter(name='sigma_gamma', param_type='integer', lower=3, upper=3),

        optml.Parameter(name='n_iters', param_type='integer', lower=10, upper=10),
        # optml.Parameter(name='param3', param_type='categorical', possible_values=['val1','val2','val3'])
    ]

    optimizer = BayesianOptimizer(model=model,
                                  hyperparams=params,
                                  eval_func=clf_score)
    # optimizer.model = model

    X_train = np.arange(len(prob_paths))  # dummy
    y_train = np.arange(len(X_train))

    # Add some good values to help out initial model
    seed_params = [
        # Best known so far
        {'n_iters': 10, 'sigma_alpha': 100, 'sigma_beta': 3, 'w1': 4, 'w2': 3, 'sigma_gamma': 3},

        {'n_iters': 10, 'sigma_alpha': 100, 'sigma_beta': 4, 'w1': 4, 'w2': 3, 'sigma_gamma': 3},
        {'n_iters': 10, 'sigma_alpha': 100, 'sigma_beta': 2, 'w1': 4, 'w2': 3, 'sigma_gamma': 3},

        {'n_iters': 10, 'sigma_alpha': 100, 'sigma_beta': 3, 'w1': 5, 'w2': 3, 'sigma_gamma': 3},
        {'n_iters': 10, 'sigma_alpha': 100, 'sigma_beta': 3, 'w1': 3, 'w2': 3, 'sigma_gamma': 3},

        {'n_iters': 10, 'sigma_alpha': 105, 'sigma_beta': 3, 'w1': 4, 'w2': 3, 'sigma_gamma': 3},
        {'n_iters': 10, 'sigma_alpha':  95, 'sigma_beta': 3, 'w1': 4, 'w2': 3, 'sigma_gamma': 3},

        {'n_iters': 10, 'sigma_alpha': 101, 'sigma_beta': 3, 'w1': 4, 'w2': 3, 'sigma_gamma': 3},
        {'n_iters': 10, 'sigma_alpha':  99, 'sigma_beta': 3, 'w1': 4, 'w2': 3, 'sigma_gamma': 3},

        {'n_iters': 10, 'sigma_alpha':  61, 'sigma_beta': 11, 'w1': 4, 'w2': 3, 'sigma_gamma': 3},

        {'n_iters': 10, 'sigma_alpha': 139, 'sigma_beta':  1, 'w1': 50, 'w2': 3, 'sigma_gamma': 3},
        {'n_iters': 10, 'sigma_alpha': 139, 'sigma_beta':  3, 'w1': 50, 'w2': 3, 'sigma_gamma': 3},
        {'n_iters': 10, 'sigma_alpha': 139, 'sigma_beta':  3, 'w1':  4, 'w2': 3, 'sigma_gamma': 3},
    ]

    seed_params = [
        # Best known so far
        {'n_iters':  5, 'sigma_alpha': 100, 'sigma_beta': 3, 'w1': 4, 'w2': 3, 'sigma_gamma': 3},
        {'n_iters': 20, 'sigma_alpha': 100, 'sigma_beta': 3, 'w1': 4, 'w2': 3, 'sigma_gamma': 3},
        # {'n_iters': 10, 'sigma_alpha': 100, 'sigma_beta': 3, 'w1': 4, 'w2': 1, 'sigma_gamma': 1},
        # {'n_iters': 10, 'sigma_alpha': 100, 'sigma_beta': 3, 'w1': 4, 'w2': 2, 'sigma_gamma': 2},
        # {'n_iters': 10, 'sigma_alpha': 100, 'sigma_beta': 3, 'w1': 4, 'w2': 4, 'sigma_gamma': 4},
    ]
    for seed in seed_params:
        print('seed = {}'.format(ub.repr2(seed, nl=0, precision=2)))
        print(optimizer._try_params(seed, X_train, y_train, X_train, y_train))

    bayes_best_params, bayes_best_model = optimizer.fit(X_train=X_train,
                                                        y_train=y_train,
                                                        n_iters=10,
                                                        verbose=True)

    names = [p.name for p in optimizer.hyperparams]
    names = ['w1', 'sigma_alpha', 'sigma_beta']
    xs = np.array([list(ub.take(params, names)) for score, params in optimizer.hyperparam_history])
    ys = np.array([score for score, params in optimizer.hyperparam_history])

    xs.T[0]
    import plottool as pt
    pt.qtensure()
    pt.plt.clf()
    for i in range(len(names)):
        pt.plt.plot(xs.T[i], ys, 'o', label=names[i])
    pt.plt.legend()


def opt_postprocess_boundary():
    import optml
    from clab import util
    from os.path import join, splitext, basename  # NOQA
    import glob
    import ubelt as ub
    import itertools as it
    import numpy as np
    from clab.live.urban_mapper import instance_fscore

    path = ub.truepath(
        '~/remote/aretha/data/work/urban_mapper2/test/input_4224-exkudlzu/'
        'solver_4214-guwsobde_unet_mmavmuou_eqnoygqy_a=1,c=RGB,n_ch=5,n_cl=4/'
        '_epoch_00000154/restiched/pred')
    mode_paths = sorted(glob.glob(path + '/*.png'))

    results = ub.odict()

    """
    #  40/40... rate=0.86 Hz, eta=0:00:00, total=0:00:46, wall=16:29 EST
    # ------------------------------------
    # best_score = '[0.60784, 0.84042, 0.47841]'
    # key        = '((only_inner, False), (inner_k, 0), (post_k, 0), (outer_k, 0), (min_seed_size, 50))'
    # key        = '((only_inner, False), (inner_k, 0), (post_k, 0), (outer_k, 0), (min_seed_size, 50))'
    # res        = '[0.60784, 0.84042, 0.47841]'
    #  40/40... rate=0.90 Hz, eta=0:00:00, total=0:00:44, wall=16:29 EST
    # ------------------------------------
    # best_score = '[0.60784, 0.84042, 0.47841]'
    # key        = '((only_inner, False), (inner_k, 0), (post_k, 0), (outer_k, 0), (min_seed_size, 50))'
    # key        = '((only_inner, False), (inner_k, 0), (post_k, 0), (outer_k, 0), (min_seed_size, 93))'
    # res        = '[0.60543, 0.86952, 0.46611]'
    #  40/40... rate=0.89 Hz, eta=0:00:00, total=0:00:44, wall=16:30 EST
    # ------------------------------------
    # best_score = '[0.60784, 0.84042, 0.47841]'
    # key        = '((only_inner, False), (inner_k, 0), (post_k, 0), (outer_k, 0), (min_seed_size, 50))'
    # key        = '((only_inner, False), (inner_k, 0), (post_k, 0), (outer_k, 0), (min_seed_size, 100))'
    # res        = '[0.60316, 0.87241, 0.46250]'
    #  40/40... rate=0.93 Hz, eta=0:00:00, total=0:00:42, wall=16:31 EST
    # ------------------------------------
    # best_score = '[0.60784, 0.84042, 0.47841]'
    # key        = '((only_inner, False), (inner_k, 0), (post_k, 0), (outer_k, 0), (min_seed_size, 50))'
    # key        = '((only_inner, True), (inner_k, 0), (post_k, 0), (outer_k, 0), (min_seed_size, 50))'
    # res        = '[0.57623, 0.77876, 0.45955]'
    #  40/40... rate=0.98 Hz, eta=0:00:00, total=0:00:40, wall=16:32 EST
    # ------------------------------------
    # best_score = '[0.60784, 0.84042, 0.47841]'
    # key        = '((only_inner, False), (inner_k, 0), (post_k, 0), (outer_k, 0), (min_seed_size, 50))'
    # key        = '((only_inner, True), (inner_k, 0), (post_k, 0), (outer_k, 0), (min_seed_size, 93))'
    # res        = '[0.58619, 0.83016, 0.45472]'
    #  40/40... rate=1.01 Hz, eta=0:00:00, total=0:00:39, wall=16:32 EST
    # ------------------------------------
    # best_score = '[0.60784, 0.84042, 0.47841]'
    # key        = '((only_inner, False), (inner_k, 0), (post_k, 0), (outer_k, 0), (min_seed_size, 50))'
    # key        = '((only_inner, True), (inner_k, 0), (post_k, 0), (outer_k, 0), (min_seed_size, 100))'
    # res        = '[0.58660, 0.83831, 0.45268]'
    """

    param_space = [
        # optml.Parameter(name='s', param_type='boolean'),
        # optml.Parameter(name='w', param_type='boolean'),
        optml.Parameter(name='only_inner', param_type='categorical', possible_values=[False, True]),
        optml.Parameter(name='inner_k', param_type='categorical', possible_values=[0]),
        optml.Parameter(name='post_k', param_type='categorical', possible_values=[0, 3, 5, 7, 9, 12, 15][0:1]),
        optml.Parameter(name='outer_k', param_type='categorical', possible_values=[0, 3, 5, 7, 9, 12, 15][0:1]),
        optml.Parameter(name='min_seed_size', param_type='categorical',
                        possible_values=[50, 93, 100]),
        # optml.Parameter(name='d', param_type='integer', lower=1, upper=5),
        # optml.Parameter(name='n', param_type='integer', lower=1, upper=1),
    ]

    from clab.tasks.urban_mapper_3d import UrbanMapper3D
    task = UrbanMapper3D('', '', boundary=True)

    def itergrid():
        names = [p.name for p in param_space]
        for values in it.product(*map(iter, param_space)):
            yield ub.odict(zip(names, values))

    def instance_label2(pred_seg, only_inner=False, inner_k=0, outer_k=0,
                        post_k=0, min_seed_size=0):
        import cv2

        inner = (pred_seg == task.classname_to_id['inner_building']).astype(np.uint8)
        # outer = (pred_seg == task.classname_to_id['outer_building']).astype(np.uint8)

        # outer_k = 15
        # outer = cv2.morphologyEx(outer, cv2.MORPH_ERODE,
        #                          np.ones((outer_k, outer_k), np.uint8),
        #                          iterations=1)

        if inner_k > 0:
            kernel = np.ones((inner_k, inner_k), np.uint8)
            inner = cv2.morphologyEx(inner, cv2.MORPH_OPEN, kernel,
                                     iterations=1)

        def cc_locs(mask):
            ccs = cv2.connectedComponents(mask.astype(np.uint8), connectivity=4)[1]
            rc_locs = np.where(mask > 0)
            rc_ids = ccs[rc_locs]
            rc_arrs = np.ascontiguousarray(np.vstack(rc_locs).T)
            cc_to_loc = util.group_items(rc_arrs, rc_ids, axis=0)
            return cc_to_loc

        if min_seed_size > 0:
            # Remove very small seeds
            for inner_id, inner_rcs in cc_locs(inner).items():
                if len(inner_rcs) < min_seed_size:
                    inner[tuple(inner_rcs.T)] = 0

        seeds = cv2.connectedComponents(inner, connectivity=4)[1]

        if only_inner:
            return seeds

        mask = (pred_seg > 0).astype(np.uint8)
        if outer_k > 1:
            mask = cv2.morphologyEx(mask, cv2.MORPH_ERODE,
                                    np.ones((outer_k, outer_k), np.uint8),
                                    iterations=1)
            # Ensure we dont clobber a seed
            mask[inner] = 1

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
        seed_mask = seeds > 0
        seed_labels = seeds[seed_mask]
        wseed[seed_mask] = seed_labels + 1
        # The unsure region is now labeled as zero

        """
        from clab.torch.urban_mapper import draw_gt_contours
        wseed_color = task.instance_colorize(wseed)

        twall_alpha = util.ensure_alpha_channel(twall * 255, alpha=1)
        twall_alpha[twall == 0, 3] = 0
        twall_alpha[twall == 1, 3]

        color_seed_wall = util.imutil.overlay_alpha_images(twall_alpha, wseed_color)
        pt.imshow(color_seed_wall)

        draw_img = draw_gt_contours(color_seed_wall, gti)
        pt.imshow(draw_img)
        """

        topology = np.dstack([twall * 255] * 3)
        markers = np.ascontiguousarray(wseed.astype(np.int32).copy())
        markers = cv2.watershed(topology, markers)
        # Remove background and border labels
        markers[markers <= 1] = 0
        """
        color_markers = task.instance_colorize(markers)

        pt.imshow(draw_gt_contours(color_markers, gti))

        color_seed_wall_ = util.ensure_alpha_channel(color_seed_wall[:, :, 0:3], alpha=.6)
        overlay_markers = util.imutil.overlay_alpha_images(color_seed_wall_, color_markers)
        pt.imshow(overlay_markers)
        pt.imshow(color_markers)

        draw_img = draw_gt_contours(overlay_markers, gti)
        pt.imshow(draw_img)
        """
        instance_mask = (markers > 0).astype(np.uint8)

        if post_k > 0:
            mask = cv2.morphologyEx(mask, cv2.MORPH_ERODE,
                                    np.ones((post_k, post_k), np.uint8),
                                    iterations=1)

        pred_ccs = cv2.connectedComponents(instance_mask, connectivity=4)[1]
        return pred_ccs

    best_key = None
    best_score = [-np.inf]

    for params in itergrid():
        key = tuple(params.items())
        if key not in results:
            scores = []
            for pred_fpath in ub.ProgIter(mode_paths[10:50]):
                gtl_fname = basename(pred_fpath).replace('.png', '_GTL.tif')
                gti_fname = basename(pred_fpath).replace('.png', '_GTI.tif')
                dsm_fname = basename(pred_fpath).replace('.png', '_DSM.tif')
                # bgr_fname = basename(pred_fpath).replace('.png', '_RGB.tif')
                gtl_fpath = join(ub.truepath('~/remote/aretha/data/UrbanMapper3D/training/'), gtl_fname)
                gti_fpath = join(ub.truepath('~/remote/aretha/data/UrbanMapper3D/training/'), gti_fname)
                dsm_fpath = join(ub.truepath('~/remote/aretha/data/UrbanMapper3D/training/'), dsm_fname)
                # bgr_fpath = join(ub.truepath('~/remote/aretha/data/UrbanMapper3D/training/'), bgr_fname)

                pred_seg = util.imread(pred_fpath)
                gti = util.imread(gti_fpath)
                gtl = util.imread(gtl_fpath)
                dsm = util.imread(dsm_fpath)
                # bgr = util.imread(bgr_fpath)

                pred = instance_label2(pred_seg, **params)

                uncertain = (gtl == 65)

                score = instance_fscore(gti, uncertain, dsm, pred)
                scores.append(score)

            res = np.array(scores).mean(axis=0)
            if res[0] > best_score[0]:
                best_score = res
                best_key = key

            print('------------------------------------')
            print('best_score = {!r}'.format(ub.repr2(list(best_score), precision=5, nl=0)))
            print('key        = {!r}'.format(ub.repr2(best_key, si=True, nl=0)))
            print('key        = {!r}'.format(ub.repr2(key, si=True, nl=0)))
            print('res        = {!r}'.format(ub.repr2(list(res), precision=5, nl=0)))
            results[key] = res

    import pandas as pd
    rdf = pd.DataFrame(list(results.values()),
                       index=results.keys(),
                       columns=['f1', 'precision', 'recall'])
    rdf.sort_values('f1').index[-1]

def junk():
    pass
    import optml
    from clab import util
    from os.path import join, splitext, basename  # NOQA
    import glob
    import ubelt as ub
    import itertools as it
    import numpy as np
    from clab.live.urban_mapper import instance_fscore

    path = ub.truepath(
        '~/remote/aretha/data/work/urban_mapper2/test/input_4224-exkudlzu/'
        'solver_4214-guwsobde_unet_mmavmuou_eqnoygqy_a=1,c=RGB,n_ch=5,n_cl=4/'
        '_epoch_00000154/restiched/pred')
    mode_paths = sorted(glob.glob(path + '/*.png'))

    results = ub.odict()

    """
    #  40/40... rate=0.86 Hz, eta=0:00:00, total=0:00:46, wall=16:29 EST
    # ------------------------------------
    # best_score = '[0.60784, 0.84042, 0.47841]'
    # key        = '((only_inner, False), (inner_k, 0), (post_k, 0), (outer_k, 0), (min_seed_size, 50))'
    # key        = '((only_inner, False), (inner_k, 0), (post_k, 0), (outer_k, 0), (min_seed_size, 50))'
    # res        = '[0.60784, 0.84042, 0.47841]'
    #  40/40... rate=0.90 Hz, eta=0:00:00, total=0:00:44, wall=16:29 EST
    # ------------------------------------
    # best_score = '[0.60784, 0.84042, 0.47841]'
    # key        = '((only_inner, False), (inner_k, 0), (post_k, 0), (outer_k, 0), (min_seed_size, 50))'
    # key        = '((only_inner, False), (inner_k, 0), (post_k, 0), (outer_k, 0), (min_seed_size, 93))'
    # res        = '[0.60543, 0.86952, 0.46611]'
    #  40/40... rate=0.89 Hz, eta=0:00:00, total=0:00:44, wall=16:30 EST
    # ------------------------------------
    # best_score = '[0.60784, 0.84042, 0.47841]'
    # key        = '((only_inner, False), (inner_k, 0), (post_k, 0), (outer_k, 0), (min_seed_size, 50))'
    # key        = '((only_inner, False), (inner_k, 0), (post_k, 0), (outer_k, 0), (min_seed_size, 100))'
    # res        = '[0.60316, 0.87241, 0.46250]'
    #  40/40... rate=0.93 Hz, eta=0:00:00, total=0:00:42, wall=16:31 EST
    # ------------------------------------
    # best_score = '[0.60784, 0.84042, 0.47841]'
    # key        = '((only_inner, False), (inner_k, 0), (post_k, 0), (outer_k, 0), (min_seed_size, 50))'
    # key        = '((only_inner, True), (inner_k, 0), (post_k, 0), (outer_k, 0), (min_seed_size, 50))'
    # res        = '[0.57623, 0.77876, 0.45955]'
    #  40/40... rate=0.98 Hz, eta=0:00:00, total=0:00:40, wall=16:32 EST
    # ------------------------------------
    # best_score = '[0.60784, 0.84042, 0.47841]'
    # key        = '((only_inner, False), (inner_k, 0), (post_k, 0), (outer_k, 0), (min_seed_size, 50))'
    # key        = '((only_inner, True), (inner_k, 0), (post_k, 0), (outer_k, 0), (min_seed_size, 93))'
    # res        = '[0.58619, 0.83016, 0.45472]'
    #  40/40... rate=1.01 Hz, eta=0:00:00, total=0:00:39, wall=16:32 EST
    # ------------------------------------
    # best_score = '[0.60784, 0.84042, 0.47841]'
    # key        = '((only_inner, False), (inner_k, 0), (post_k, 0), (outer_k, 0), (min_seed_size, 50))'
    # key        = '((only_inner, True), (inner_k, 0), (post_k, 0), (outer_k, 0), (min_seed_size, 100))'
    # res        = '[0.58660, 0.83831, 0.45268]'
    """

    param_space = [
        # optml.Parameter(name='s', param_type='boolean'),
        # optml.Parameter(name='w', param_type='boolean'),
        optml.Parameter(name='only_inner', param_type='categorical', possible_values=[False, True]),
        optml.Parameter(name='inner_k', param_type='categorical', possible_values=[0]),
        optml.Parameter(name='post_k', param_type='categorical', possible_values=[0, 3, 5, 7, 9, 12, 15][0:1]),
        optml.Parameter(name='outer_k', param_type='categorical', possible_values=[0, 3, 5, 7, 9, 12, 15][0:1]),
        optml.Parameter(name='min_seed_size', param_type='categorical',
                        possible_values=[50, 93, 100]),
        # optml.Parameter(name='d', param_type='integer', lower=1, upper=5),
        # optml.Parameter(name='n', param_type='integer', lower=1, upper=1),
    ]

    from clab.tasks.urban_mapper_3d import UrbanMapper3D
    task = UrbanMapper3D('', '', boundary=True)

    def itergrid():
        names = [p.name for p in param_space]
        for values in it.product(*map(iter, param_space)):
            yield ub.odict(zip(names, values))

    def instance_label2(pred_seg, only_inner=False, inner_k=0, outer_k=0,
                        post_k=0, min_seed_size=0):
        import cv2

        inner = (pred_seg == task.classname_to_id['inner_building']).astype(np.uint8)
        # outer = (pred_seg == task.classname_to_id['outer_building']).astype(np.uint8)

        # outer_k = 15
        # outer = cv2.morphologyEx(outer, cv2.MORPH_ERODE,
        #                          np.ones((outer_k, outer_k), np.uint8),
        #                          iterations=1)

        if inner_k > 0:
            kernel = np.ones((inner_k, inner_k), np.uint8)
            inner = cv2.morphologyEx(inner, cv2.MORPH_OPEN, kernel,
                                     iterations=1)

        def cc_locs(mask):
            ccs = cv2.connectedComponents(mask.astype(np.uint8), connectivity=4)[1]
            rc_locs = np.where(mask > 0)
            rc_ids = ccs[rc_locs]
            rc_arrs = np.ascontiguousarray(np.vstack(rc_locs).T)
            cc_to_loc = util.group_items(rc_arrs, rc_ids, axis=0)
            return cc_to_loc

        if min_seed_size > 0:
            # Remove very small seeds
            for inner_id, inner_rcs in cc_locs(inner).items():
                if len(inner_rcs) < min_seed_size:
                    inner[tuple(inner_rcs.T)] = 0

        seeds = cv2.connectedComponents(inner, connectivity=4)[1]

        if only_inner:
            return seeds

        mask = (pred_seg > 0).astype(np.uint8)
        if outer_k > 1:
            mask = cv2.morphologyEx(mask, cv2.MORPH_ERODE,
                                    np.ones((outer_k, outer_k), np.uint8),
                                    iterations=1)
            # Ensure we dont clobber a seed
            mask[inner] = 1

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
        seed_mask = seeds > 0
        seed_labels = seeds[seed_mask]
        wseed[seed_mask] = seed_labels + 1
        # The unsure region is now labeled as zero

        """
        from clab.torch.urban_mapper import draw_gt_contours
        wseed_color = task.instance_colorize(wseed)

        twall_alpha = util.ensure_alpha_channel(twall * 255, alpha=1)
        twall_alpha[twall == 0, 3] = 0
        twall_alpha[twall == 1, 3]

        color_seed_wall = util.imutil.overlay_alpha_images(twall_alpha, wseed_color)
        pt.imshow(color_seed_wall)

        draw_img = draw_gt_contours(color_seed_wall, gti)
        pt.imshow(draw_img)
        """

        topology = np.dstack([twall * 255] * 3)
        markers = np.ascontiguousarray(wseed.astype(np.int32).copy())
        markers = cv2.watershed(topology, markers)
        # Remove background and border labels
        markers[markers <= 1] = 0
        """
        color_markers = task.instance_colorize(markers)

        pt.imshow(draw_gt_contours(color_markers, gti))

        color_seed_wall_ = util.ensure_alpha_channel(color_seed_wall[:, :, 0:3], alpha=.6)
        overlay_markers = util.imutil.overlay_alpha_images(color_seed_wall_, color_markers)
        pt.imshow(overlay_markers)
        pt.imshow(color_markers)

        draw_img = draw_gt_contours(overlay_markers, gti)
        pt.imshow(draw_img)
        """
        instance_mask = (markers > 0).astype(np.uint8)

        if post_k > 0:
            mask = cv2.morphologyEx(mask, cv2.MORPH_ERODE,
                                    np.ones((post_k, post_k), np.uint8),
                                    iterations=1)

        pred_ccs = cv2.connectedComponents(instance_mask, connectivity=4)[1]
        return pred_ccs

    best_key = None
    best_score = [-np.inf]

    for params in itergrid():
        key = tuple(params.items())
        if key not in results:
            scores = []
            for pred_fpath in ub.ProgIter(mode_paths[10:50]):
                gtl_fname = basename(pred_fpath).replace('.png', '_GTL.tif')
                gti_fname = basename(pred_fpath).replace('.png', '_GTI.tif')
                dsm_fname = basename(pred_fpath).replace('.png', '_DSM.tif')
                # bgr_fname = basename(pred_fpath).replace('.png', '_RGB.tif')
                gtl_fpath = join(ub.truepath('~/remote/aretha/data/UrbanMapper3D/training/'), gtl_fname)
                gti_fpath = join(ub.truepath('~/remote/aretha/data/UrbanMapper3D/training/'), gti_fname)
                dsm_fpath = join(ub.truepath('~/remote/aretha/data/UrbanMapper3D/training/'), dsm_fname)
                # bgr_fpath = join(ub.truepath('~/remote/aretha/data/UrbanMapper3D/training/'), bgr_fname)

                pred_seg = util.imread(pred_fpath)
                gti = util.imread(gti_fpath)
                gtl = util.imread(gtl_fpath)
                dsm = util.imread(dsm_fpath)
                # bgr = util.imread(bgr_fpath)

                pred = instance_label2(pred_seg, **params)

                uncertain = (gtl == 65)

                score = instance_fscore(gti, uncertain, dsm, pred)
                scores.append(score)

            res = np.array(scores).mean(axis=0)
            if res[0] > best_score[0]:
                best_score = res
                best_key = key

            print('------------------------------------')
            print('best_score = {!r}'.format(ub.repr2(list(best_score), precision=5, nl=0)))
            print('key        = {!r}'.format(ub.repr2(best_key, si=True, nl=0)))
            print('key        = {!r}'.format(ub.repr2(key, si=True, nl=0)))
            print('res        = {!r}'.format(ub.repr2(list(res), precision=5, nl=0)))
            results[key] = res

    import pandas as pd
    rdf = pd.DataFrame(list(results.values()),
                       index=results.keys(),
                       columns=['f1', 'precision', 'recall'])
    rdf.sort_values('f1').index[-1]

        import optml
        from optml.bayesian_optimizer import BayesianOptimizer
        hyperparams = [
            optml.Parameter(name='mask_thresh', param_type='continuous', lower=.15, upper=.85),
            optml.Parameter(name='seed_thresh', param_type='continuous', lower=.15, upper=.85),
            optml.Parameter(name='min_seed_size', param_type='integer', lower=0, upper=100),
        ]
        self = BayesianOptimizer(model=None, hyperparams=hyperparams, eval_func=None)

        for path, path1 in zip(prob_paths, prob1_paths):
            pass

        def func(**new_hyperparams):
            probs = np.load(path)['arr_0']
            probs1 = np.load(path1)['arr_0']
            seed_probs = probs[:, :, task.classname_to_id['inner_building']]
            mask_probs = probs1[:, :, 1]

            seed_thresh, mask_thresh, min_seed_size = ub.take(
                new_hyperparams, ['seed_thresh', 'mask_thresh', 'min_seed_size'])
            seed = (seed_probs > mask_thresh).astype(np.uint8)
            mask = (mask_probs > seed_thresh).astype(np.uint8)
            pred = seeded_instance_label(seed, mask, min_seed_size=min_seed_size)
            scores = instance_fscore(gti, uncertain, dsm, pred)
            fscore = scores[0]
            return fscore

        history = self.simple(func, n_iters=10, verbose=True)
        best = max(history)
        print('best = {!r}'.format(best))

        # {'mask_thresh': 0.45318221555100013, 'seed_thresh': 0.69172340500683327, 'min_seed_size': 41}
        func(**{'mask_thresh': 0.5, 'seed_thresh': 0.7, 'min_seed_size': 20})


def hypersearch_probs():
    prob_paths  = paths['probs']
    prob1_paths = paths['probs1']

    # https://github.com/fmfn/BayesianOptimization
    # https://github.com/fmfn/BayesianOptimization/blob/master/examples/usage.py
    # https://github.com/fmfn/BayesianOptimization/blob/master/examples/exploitation%20vs%20exploration.ipynb
    # subx = [0, 1, 2, 3, 4, 5]
    subx = [2, 4, 5, 9, 10, 14, 17, 18, 20, 30, 33, 39, 61, 71, 72, 73, 75, 81, 84]
    from bayes_opt import BayesianOptimization

    def best(self):
        return {'max_val': self.Y.max(),
                'max_params': dict(zip(self.keys,
                                       self.X[self.Y.argmax()]))}

    def seeded_objective(**params):
        seed_thresh, mask_thresh, min_seed_size, min_size = ub.take(
            params, 'seed_thresh, mask_thresh, min_seed_size, min_size'.split(', '))
        fscores = []
        for path, path1 in zip(ub.take(prob_paths, subx), ub.take(prob1_paths, subx)):
            gti, uncertain, dsm, bgr = gt_info_from_path(path)

            probs = np.load(path)['arr_0']
            seed_probs = probs[:, :, task.classname_to_id['inner_building']]
            seed = (seed_probs > seed_thresh).astype(np.uint8)

            probs1 = np.load(path1)['arr_0']
            mask_probs = probs1[:, :, 1]
            mask = (mask_probs > mask_thresh).astype(np.uint8)

            pred = seeded_instance_label(seed, mask,
                                         min_seed_size=min_seed_size,
                                         min_size=min_size)
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
    }
    n_init = 50
    seeded_bo = BayesianOptimization(seeded_objective, seeded_bounds)
    seeded_bo.explore(pd.DataFrame([
        {'mask_thresh': 0.9000, 'min_seed_size': 100.0000, 'min_size': 100.0000, 'seed_thresh': 0.4000},
        {'mask_thresh': 0.8, 'seed_thresh': 0.5, 'min_seed_size': 20, 'min_size': 0},
        {'mask_thresh': 0.5, 'seed_thresh': 0.8, 'min_seed_size': 20, 'min_size': 0},
        {'mask_thresh': 0.8338, 'min_seed_size': 25.7651, 'min_size': 38.6179, 'seed_thresh': 0.6573},
        {'mask_thresh': 0.6225, 'min_seed_size': 93.2705, 'min_size': 5, 'seed_thresh': 0.4401},
        {'mask_thresh': 0.7870, 'min_seed_size': 85.1641, 'min_size': 64.0634, 'seed_thresh': 0.4320},
        {'mask_thresh': 0.8367, 'seed_thresh': 0.4549, 'min_seed_size': 97, 'min_size': 33},  # 'max_val': 0.8708
        {'mask_thresh': 0.7664, 'min_seed_size': 48.5327, 'min_size': 61.8757, 'seed_thresh': 0.4090},  # 'max_val': 0.9091}
        {'mask_thresh': 0.8367, 'min_seed_size': 97.0000, 'min_size': 33.0000, 'seed_thresh': 0.4549},  # max_val': 0.8991
    ]).to_dict(orient='list'))
    seeded_bo.plog.print_header(initialization=True)
    seeded_bo.init(n_init)
    print(ub.repr2(best(seeded_bo), nl=0, precision=4))

    print('seeded ' + ub.repr2(best(seeded_bo), nl=0, precision=4))
    print('inner ' + ub.repr2(best(inner_bo), nl=0, precision=4))
    print('outer ' + ub.repr2(best(outer_bo), nl=0, precision=4))

    # {'max_params': {'thresh': 0.8000, 'min_size': 0.0000}, 'max_val': 0.6445}
    gp_params = {"alpha": 1e-5, "n_restarts_optimizer": 2}

    n_iter = n_init // 2
    for kappa in [10, 5, 1]:
        seeded_bo.maximize(n_iter=n_iter, acq='ucb', kappa=kappa, **gp_params)
        inner_bo.maximize(n_iter=n_iter, acq='ucb', kappa=kappa, **gp_params)
        outer_bo.maximize(n_iter=n_iter, acq='ucb', kappa=kappa, **gp_params)

    print('seeded ' + ub.repr2(best(seeded_bo), nl=0, precision=4))
    print('inner ' + ub.repr2(best(inner_bo), nl=0, precision=4))
    print('outer ' + ub.repr2(best(outer_bo), nl=0, precision=4))
    print(arch)

    # bo.maximize(n_iter=3, acq='poi', xi=1e-4)
    # bo.maximize(n_iter=3, acq='poi', xi=1e-1)

    # bo.maximize(n_iter=3, acq='ucb', kappa=1)
    # bo.maximize(n_iter=3, acq='ucb', kappa=5)

    # bo.maximize(n_iter=3, acq='ei', xi=1e-4)
    # bo.maximize(n_iter=3, acq='ei', xi=1e-1)

    # import functools
    # bounds2 = {
    #     'mask_thresh': (.3, .95),
    #     'thresh': (.3, .95),
    # }
    # func2 = functools.partial(seeded_objective, min_seed_size=20)
    # bo2 = BayesianOptimization(func2, bounds2)
    # bo2.explore(pd.DataFrame([
    #     {'mask_thresh': 0.88201857575666986, 'seed_thresh': 0.51167580304776372},
    #     {'mask_thresh': 0.7,  'seed_thresh': 0.2},
    #     {'mask_thresh': 0.69,  'seed_thresh': 0.45},
    #     {'mask_thresh': 0.5,  'seed_thresh': 0.5},
    #     {'mask_thresh': 0.89, 'seed_thresh': 0.52},
    # ]).to_dict(orient='list'))
    # bo2.init(5)
    # bo2.maximize(n_iter=3, acq='ucb', **gp_params)
    # bo2.res['max']['max_params']

    # # Seeded version
    # for path, path1 in zip(prob_paths, prob1_paths):
    #     pass

    #     probs = np.load(path)['arr_0']
    #     probs1 = np.load(path1)['arr_0']
    #     seed_probs = probs[:, :, task.classname_to_id['inner_building']]
    #     mask_probs = probs1[:, :, 1]

    #     gti, uncertain, dsm, bgr = gt_info_from_path(path1)

    #     # convert probs into a prediction and score it
    #     x = {}
    #     mask_thresh = .9
    #     seed_thresh = .2
    #     min_seed_size = 50

    #     seed = (seed_probs > mask_thresh).astype(np.uint8)
    #     mask = (mask_probs > seed_thresh).astype(np.uint8)
    #     pred = seeded_instance_label(seed, mask, min_seed_size=min_seed_size)

    #     scores = instance_fscore(gti, uncertain, dsm, pred)

    #     print('thresh, scores = {}, {}'.format(ub.repr2(thresh, precision=3), ub.repr2(scores, nl=0, precision=3)))
    #     x[thresh] = scores

    #     print(pd.DataFrame.from_dict(x, orient='index')[0].argmax())
    #     print(pd.DataFrame.from_dict(x, orient='index')[0].max())

