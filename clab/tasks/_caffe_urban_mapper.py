# old pieces of the urban mapper task need for caffe


def script_evaluate_challenge_test_images(task):
    """
    TODO: REMOVE: use pytorch code instead

    Script:
        >>> import clab
        >>> from clab.tasks.urban_mapper_3d import *
        >>> task = UrbanMapper3D(root='~/remote/aretha/data/UrbanMapper3D', workdir='~/data/work/urban_mapper')
    """
    # SEE sseg train
    eval_fullres = task.load_fullres_inputs('testing')

    datadir = ub.ensuredir((task.workdir, 'eval_data'))
    prep = preprocess.Preprocessor(datadir)
    eval_part1_scale = prep.make_parts(eval_fullres, scale=1, clear=0)

    test_weights_fpath = expanduser(
        '~/data/work/urban_mapper/harness/xval/split_00/arch/segnet_proper/train/input_46354-gytwbrsy/solver_46354-gytwbrsy_segnet_proper_None_uxspkpj/testable/test_weights_46354-gytwbrsy_segnet_proper_None_uxspkpj_00040000.caffemodel')

    from clab import harness
    harn = harness.Harness(workdir=task.exptdir('eval'),
                           arch='segnet_proper', task=task)
    harn.set_inputs(train=None, test=eval_part1_scale)
    harn.prepare_test_model()
    harn.test_weights_fpath = test_weights_fpath
    harn.test_dump_dpath = ub.ensuredir((harn.workdir, 'dump'))
    harn.predict(have_true=False)

    # import glob
    # pred_paths = sorted(glob.glob(expanduser('~/data/work/urban_mapper/data/im/part-scale1-360_480-titxvmed/*.tif')))
    # restitched_pred_dpath = ub.ensuredir((harn.test_dump_dpath, 'restiched', 'pred'))
    # restich_and_dump(restitched_pred_dpath, pred_paths)

    restitched_pred_dpath = ub.ensuredir((
        harn.test_dump_dpath, 'restiched', 'pred'))
    pred_paths = [join(harn.test_dump_dpath, 'pred', n).replace('.tif', '.png')
                  for n in harn.test.dump_im_names]
    assert all([exists(p) for p in pred_paths])
    # dpath, part_paths = restitched_pred_dpath, pred_paths
    restitched_pred = task.restitch(
        restitched_pred_dpath, pred_paths, blend='vote')

    restitched_color_pred_dpath = ub.ensuredir((
        harn.test_dump_dpath, 'restiched', 'color_pred'))

    lines = []
    for fpath in ub.ProgIter(restitched_pred, label='rle'):
        pred = imutil.imread(fpath)

        cc_labels = task.instance_label(pred)
        # n_ccs, cc_labels = cv2.connectedComponents(pred)

        fname = splitext(basename(fpath))[0]
        height, width = pred.shape[0:2]
        runlen = imutil.run_length_encoding(cc_labels)

        lines.append(fname)
        lines.append('{},{}'.format(width, height))
        lines.append(','.join(list(map(str, runlen))))

    # Submission URL
    # https://community.topcoder.com/longcontest/?module=Submit&compid=57607&rd=17007&cd=15282

    text = '\n'.join(lines)
    ub.writeto('urban_mapper_test_pred.txt', text)

    for fpath in restitched_pred:
        pred = imutil.imread(fpath)
        color_pred = task.colorize(pred)
        imutil.imwrite(join(restitched_color_pred_dpath,
                            basename(fpath)), color_pred)

    restitched_pred_blend_dpath = ub.ensuredir((
        harn.test_dump_dpath, 'restiched', 'pred_blend'))
    blend_pred_paths = [
        join(harn.test_dump_dpath, 'blend_pred', n).replace('.tif', '.png')
        for n in harn.test.dump_im_names
    ]
    assert all([exists(p) for p in blend_pred_paths])
    # dpath, part_paths = restitched_pred_blend_dpath, blend_pred_paths
    restitched_blend_pred = task.restitch(
        restitched_pred_blend_dpath, blend_pred_paths, blend=None)

def run_xval_evaluation(task, arch='segnet_proper', pretrained=None,
                        hyperparams={}, test_keys=None, fit=True):
    """
    Writes test/train data files containing the image paths that will be
    used for testing and training as well as the appropriate solvers and
    prediction models.

    TODO: remove in favor of pytorch code

    Args:
        fit (bool): if False, we will only evaluate models that have been
            trained so far (useful for getting results from existing while a
            model is not done training)

    CommandLine:
        python -m clab.tasks.urban_mapper_3d UrbanMapper3D.run_xval_evaluation --fit=True
        python -m clab.tasks.urban_mapper_3d UrbanMapper3D.run_xval_evaluation --fit=False

    Script:
        >>> import clab
        >>> from clab.tasks.urban_mapper_3d import *
        >>> task = UrbanMapper3D(root='~/remote/aretha/data/UrbanMapper3D', workdir='~/data/work/urban_mapper')
        >>> print(task.classnames)
        >>> task.preprocess()
        >>> import utool as ut
        >>> kw = ut.exec_argparse_funckw(task.run_xval_evaluation, globals())
        >>> print('kw = {!r}'.format(kw))
        >>> task.run_xval_evaluation(**kw)
    """
    # import os
    # os.environ['GLOG_minloglevel'] = '3'

    from clab import harness
    # from clab import models

    xval_base = abspath(task.exptdir('xval'))

    xval_results = []
    for idx, xval_split in enumerate(task.xval_splits(test_keys=test_keys)):
        print(ub.color_text('XVAL iteration {}'.format(idx), 'blue'))

        (train, test) = xval_split
        xval_dpath = ub.ensuredir((xval_base, 'split_{:0=2}'.format(idx)))

        harn = harness.Harness(workdir=xval_dpath, arch=arch, task=task)
        harn.task = task
        harn.set_inputs(train, test)

        harn.init_pretrained_fpath = pretrained
        harn.init_pretrained_fpath = None
        harn.params.update(hyperparams)

        harn.test.prepare_images()
        harn.train.prepare_images()

        harn.gpu_num = gpu_util.find_unused_gpu(min_memory=6000)
        print('harn.gpu_num = {!r}'.format(harn.gpu_num))
        if harn.gpu_num is not None:
            avail_mb = gpu_util.gpu_info()[harn.gpu_num]['mem_avail']
            # Estimate how much we can fit in memory
            # TODO: estimate this from the model arch instead.
            # (90 is a mgic num corresponding to segnet_proper)
            harn.train_batch_size = int(
                (avail_mb * 90) // np.prod(task.input_shape))
            harn.train_batch_size = int(harn.train_batch_size)
            if harn.train_batch_size == 0:
                raise MemoryError(
                    'not enough GPU memory to train the model')
        else:
            # not sure what the best to do on CPU is. Probably nothing.
            harn.train_batch_size = 4

        harn.prepare_solver()

        # Check if we can resume a previous training state
        print(ub.color_text('Checking for previous snapshot states', 'blue'))
        previous_states = harn.snapshot_states()
        print('Found {} previous_states'.format(len(previous_states)))

        if fit:
            if len(previous_states) == 0:
                print(ub.color_text('Starting a fresh training session', 'blue'))
                harn.fit()
            else:
                from clab.backend import iface_caffe as iface
                solver_info = iface.parse_solver_info(harn.solver_fpath)
                prev_state = previous_states[-1]
                prev_iter = iface.snapshot_iterno(prev_state)
                if prev_iter < solver_info['max_iter']:
                    # continue from the previous iteration
                    print(ub.color_text(
                        'Resume training from iter {}'.format(prev_iter), 'blue'))
                    harn.fit(prevstate_fpath=prev_state)
                else:
                    print(ub.color_text(
                        'Already finished training this model', 'blue'))

        for _ in harn.deploy_trained_for_testing():
            # hack to evaulate while deploying
            harn.evaulate_all()
        xval_results.append(list(harn._test_results_fpaths()))
    return xval_results

