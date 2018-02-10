

    # Recombined predictions on chips into predictions on the original inputs
    paths = {}
    for mode in ['pred', 'pred1']:
        restitched_paths = pharn._restitch_type(mode, blend='vote')
        if 1:
            pharn._restitch_type('blend_' + mode, blend=None)
        paths[mode] = restitched_paths

    import pandas as pd
    task = test_dataset.task

    restitched_pred0 = pharn._restitch_type('pred', blend='vote')
    restitched_pred1 = pharn._restitch_type('pred1', blend='vote')

    out_fpaths = unet2_instance_restitch(restitched_pred0, restitched_pred1, task)
    test_instance_restitch(out_fpaths, task)

    # Evaluate the binary predictions by themselves
    mode = 'pred1'
    restitched_paths = paths[mode]
    scores1 = []
    for pred_fpath in ub.ProgIter(restitched_paths, label='scoring 1'):
        gti, uncertain, dsm, bgr = gt_info_from_path(pred_fpath)
        pred_seg = util.imread(pred_fpath)
        pred = task.instance_label(pred_seg, dist_thresh=5, k=12, watershed=True)
        scores1.append(instance_fscore(gti, uncertain, dsm, pred))

    scores_df1 = pd.DataFrame(scores1, columns=['fscore', 'precision', 'recall'])
    print('binary fscore {}'.format(scores_df1['fscore'].mean()))

    # ----------------------------------------
    # Combine the binary and inner predictions.

    mode = 'pred'
    restitched_paths = paths[mode]
    scores0 = []
    for pred_fpath in ub.ProgIter(restitched_paths, label='scoring seeds'):
        gti, uncertain, dsm, bgr = gt_info_from_path(pred_fpath)

        pred_seg0 = util.imread(pred_fpath)
        pred_seg1 = util.imread(pred_fpath.replace('/pred/', '/pred1/'))

        seed = (pred_seg0 == task.classname_to_id['inner_building']).astype(np.uint8)
        mask = (pred_seg1 == 1)
        pred = seeded_instance_label(seed, mask, min_seed_size=50)

        scores0.append(instance_fscore(gti, uncertain, dsm, pred))

    scores_df0 = pd.DataFrame(scores0, columns=['fscore', 'precision', 'recall'])
    print('binary scores\n{}'.format(scores_df0.mean(axis=0)))

    # -------- OLD ---------
    # hack
    if 0:
        for mode in ['blend_pred', 'blend_pred_crf']:
            restitched_paths = pharn._restitch_type(mode, blend=None)

        paths = {}
        for mode in ['pred', 'pred_crf']:
            restitched_paths = pharn._restitch_type(mode, blend='vote')
            for big_pred_fpath in ub.ProgIter(restitched_paths, label='open ' + mode):
                big_pred = imutil.imread(big_pred_fpath)

                k = 7
                n_iters = 1
                new_fpath = big_pred_fpath.replace('/' + mode + '/', '/' + mode + '_open{}x{}/'.format(k, n_iters))
                new_blend_fpath = big_pred_fpath.replace('/' + mode + '/', '/blend_' + mode + '_open{}x{}/'.format(k, n_iters))
                ub.ensuredir(os.path.dirname(new_fpath))
                ub.ensuredir(os.path.dirname(new_blend_fpath))

                pred2 = (test_dataset.task.instance_label(big_pred, k=k,
                                                          n_iters=n_iters,
                                                          watershed=False) > 0
                         ).astype(np.int8)
                imutil.imwrite(new_fpath, pred2)

                big_im_fname = basename(big_pred_fpath).replace('.png', '_RGB.tif')
                big_orig_fpath = join('/home/local/KHQ/jon.crall/remote/aretha/data/UrbanMapper3D/training/', big_im_fname)
                big_orig = imutil.imread(big_orig_fpath)

                big_blend_instance_pred = test_dataset.task.colorize(pred2, big_orig)
                imutil.imwrite(new_blend_fpath, big_blend_instance_pred)

        mode = 'pred_crf'
        mode = 'pred'
        restitched_paths = pharn._restitch_type(mode, blend='vote')

        big_pred_fpath = restitched_paths[17]
        orig_fname = basename(big_pred_fpath).replace('.png', '_RGB.tif')
        big_orig_fpath = join('/home/local/KHQ/jon.crall/remote/aretha/data/UrbanMapper3D/training/', orig_fname)
        # big_orig_fpath = '/home/local/KHQ/jon.crall/remote/aretha/data/UrbanMapper3D/training/TAM_Tile_017_RGB.tif'

        big_pred = imutil.imread(big_pred_fpath)
        big_orig = imutil.imread(big_orig_fpath)

        k = 3
        kernel = np.ones((k, k), np.uint8)
        import cv2
        opening = cv2.morphologyEx(big_pred, cv2.MORPH_OPEN, kernel, iterations=2)
        n_ccs, cc_labels = cv2.connectedComponents(opening.astype(np.uint8), connectivity=4)

        # cc_labels = task.instance_label(big_pred)

        big_blend_instance_pred = test_dataset.task.instance_colorize(cc_labels, big_orig)
        # big_blend_instance_pred = task.colorize(cc_labels > 0, big_orig)
        restitched_pred_dpath = ub.ensuredir((pharn.test_dump_dpath, 'restiched', 'blend_instance_' + mode))
        fname = basename(big_pred_fpath)
        imutil.imwrite(join(restitched_pred_dpath, fname), big_blend_instance_pred)

    if 1:
        import pandas as pd  # NOQA
        from clab.metrics import confusion_matrix, jaccard_score_from_confusion  # NOQA
        from clab import filters  # NOQA

        paths = {}
        for mode in ['pred', 'pred_crf']:
            restitched_paths = pharn._restitch_type(mode, blend='vote')
            paths[mode] = restitched_paths

        scores = {}
        import cv2
        for mode in ['pred', 'pred_crf']:
            print('mode = {!r}'.format(mode))
            restitched_paths = paths[mode]

            for n_iters in range(1, 2):
                for k in range(5, 10, 2):
                    for watershed in [False, True]:
                        cfsn2 = np.zeros((3, 3))
                        for big_pred_fpath in restitched_paths:
                            big_pred = imutil.imread(big_pred_fpath)

                            big_gt_fname = basename(big_pred_fpath).replace('.png', '_GTL.tif')
                            big_gt_fpath = join('/home/local/KHQ/jon.crall/remote/aretha/data/UrbanMapper3D/training/', big_gt_fname)
                            big_gt = imutil.imread(big_gt_fpath)
                            big_gt[big_gt == 2] = 0
                            big_gt[big_gt == 6] = 1
                            big_gt[big_gt == 65] = 2

                            pred2 = (test_dataset.task.instance_label(
                                big_pred, k=k, n_iters=n_iters,
                                watershed=watershed) > 0).astype(np.int8)

                            # # cfsn1 += confusion_matrix(big_gt.ravel(), big_pred.ravel(), labels=[0, 1, 2])
                            # if k > 1:
                            #     kernel = np.ones((k, k), np.uint8)
                            #     opening = cv2.morphologyEx(big_pred, cv2.MORPH_OPEN, kernel, iterations=n_iters)
                            #     # opening = filters.watershed_filter(opening)
                            #     # n_ccs, cc_labels = cv2.connectedComponents(opening.astype(np.uint8), connectivity=4)
                            #     # pred2 = (cc_labels > 0).astype(np.int)
                            #     pred2 = opening
                            # else:
                            #     pred2 = big_pred

                            cfsn2 += confusion_matrix(big_gt.ravel(), pred2.ravel(), labels=[0, 1, 2])

                        miou = jaccard_score_from_confusion(cfsn2)[0:2].mean()
                        scores[(mode, k, n_iters, watershed)] = miou
                        print('mode={}, k={:3d}, n_iters={}, w={} miou = {!r}'.format(mode, k, n_iters, int(watershed), miou))

        print(pd.Series(scores).sort_values())


        # if False:
        #     loader = torch.utils.data.DataLoader(
        #         pharn.dataset, shuffle=False,
        #         pin_memory=True,
        #         num_workers=0,
        #         batch_size=1,
        #     )
        #     prog = ub.ProgIter(length=len(loader), label='predict proba')
        #     for ix, loaded in enumerate(prog(loader)):
        #         fname = pharn.dataset.inputs.dump_im_names[ix]
        #         fname = os.path.splitext(fname)[0] + '.png'

        #         if pharn.dataset.with_gt:
        #             inputs_ = loaded[0]
        #         else:
        #             inputs_ = loaded

        #         if not isinstance(inputs_, (list, tuple)):
        #             inputs_ = [inputs_]

        #         inputs_ = pharn.xpu.variabless(*inputs_)
        #         outputs = pharn.model.forward(inputs_)

        #         if not isinstance(outputs, (list, tuple)):
        #             outputs = [outputs]

        #         for ox in range(len(outputs)):
        #             suffix = '' if ox == 0 else str(ox)

        #             output_tensor = outputs[ox]
        #             log_prob_tensor = torch.nn.functional.log_softmax(output_tensor, dim=1)[0]
        #             prob_tensor = torch.exp(log_prob_tensor)
        #             probs = prob_tensor.data.cpu().numpy().transpose(1, 2, 0)

        #             output_dict = {
        #                 'probs' + suffix: probs,
        #             }

        #             for key, data in output_dict.items():
        #                 dpath = join(pharn.test_dump_dpath, key)
        #                 ub.ensuredir(dpath)
        #                 fpath = join(dpath, fname)
        #                 if key == 'probs' + suffix:
        #                     fpath = ub.augpath(fpath, ext='.h5')
        #                     # fpath = ub.augpath(fpath, ext='.npy')
        #                     util.write_arr(fpath, data)
        #                     # util.write_arr(fpath, data)
        #                 else:
        #                     imutil.imwrite(fpath, data)


# def erode_ccs(ccs):
#     pass



# # mode = 'pred_crf'
# def two_channel_version():
#     task = eval_dataset.task
#     restitched_pred0 = pharn._restitch_type('pred', blend='vote')
#     restitched_pred1 = pharn._restitch_type('pred1', blend='vote')
#     pharn._restitch_type('blend_pred', blend=None)
#     pharn._restitch_type('blend_pred1', blend=None)

#     out_fpaths = unet2_instance_restitch(restitched_pred0, restitched_pred1, task)

#     lines = []
#     for fpath in sorted(out_fpaths):
#         pred = imutil.imread(fpath)
#         import cv2
#         cc_labels = cv2.connectedComponents(pred, connectivity=4)[1]

#         fname = splitext(basename(fpath))[0]
#         (width, height), runlen = imutil.run_length_encoding(cc_labels)

#         lines.append(fname)
#         lines.append('{},{}'.format(width, height))
#         lines.append(','.join(list(map(str, runlen))))

#     text = '\n'.join(lines)
#     post_idstr = 'dualout'
#     mode = 'pred'
#     suffix = '_'.join(pharn.test_dump_dpath.split('/')[-2:]) + '_' + mode + '_' + post_idstr
#     fpath = join(pharn.test_dump_dpath, 'urban_mapper_test_pred_' + suffix + '.txt')
#     print('fpath = {!r}'.format(fpath))
#     ub.writeto(fpath, text)
#     print(ub.codeblock(
#         '''
#         # Execute on remote computer
#         cd ~/Dropbox/TopCoder
#         rsync aretha:{fpath} .
#         '''
#     ).format(fpath=fpath))

# def one_channel_version():
#     mode = 'pred'
#     restitched_pred = pharn._restitch_type(mode, blend='vote')
#     if True:
#         pharn._restitch_type('blend_' + mode, blend=None)
#     restitched_pred = eval_dataset.fullres.align(restitched_pred)

#     # Convert to submission output format
#     post_kw = dict(k=15, n_iters=1, dist_thresh=5, watershed=True)
#     # post_kw = dict(k=0, watershed=False)
#     post_idstr = compact_idstr(post_kw)

#     lines = []
#     for ix, fpath in enumerate(ub.ProgIter(restitched_pred, label='rle')):
#         pred = imutil.imread(fpath)
#         cc_labels = eval_dataset.task.instance_label(pred, **post_kw)

#         fname = splitext(basename(fpath))[0]
#         (width, height), runlen = imutil.run_length_encoding(cc_labels)

#         lines.append(fname)
#         lines.append('{},{}'.format(width, height))
#         lines.append(','.join(list(map(str, runlen))))

#     text = '\n'.join(lines)
#     suffix = '_'.join(pharn.test_dump_dpath.split('/')[-2:]) + '_' + mode + '_' + post_idstr
#     fpath = join(pharn.test_dump_dpath, 'urban_mapper_test_pred_' + suffix + '.txt')
#     ub.writeto(fpath, text)

# if '/unet2/' in train_dpath:
#     two_channel_version()
# else:
#     one_channel_version()


def color_probs(pharn, task):
    """
    Ignore:
        pharn._restitch_type('blend_probs/c0_non-building', blend=None)
        pharn._restitch_type('blend_probs/c1_inner-building', blend=None)
        pharn._restitch_type('blend_probs/c2_outer-building', blend=None)

        pharn._restitch_type('blend_probs1/c0_non-building', blend=None)
        pharn._restitch_type('blend_probs1/c1_building', blend=None)
    """

    mode = 'probs1'
    mode = 'probs'

    dpath = join(pharn.test_dump_dpath, mode)

    out_dpath = join(pharn.test_dump_dpath, 'blend_' + mode)

    npy_fpaths = glob.glob(join(dpath, '*.npy'))

    bgr_paths = pharn.dataset.inputs.paths['im']
    gtl_paths = pharn.dataset.inputs.paths['gt']
    npy_fpaths = pharn.dataset.inputs.align(npy_fpaths)

    for ix, fpath in enumerate(ub.ProgIter(npy_fpaths)):
        gt = util.imread(gtl_paths[ix])
        bgr = util.imread(bgr_paths[ix])

        probs = util.read_arr(fpath)

        # Dump each channel
        for c in reversed(range(probs.shape[0])):
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

            draw_img = draw_gt_contours2(blend_probs, gt, thickness=2, alpha=.5)
            util.imwrite(c_fpath, draw_img)
    pharn._restitch_type('blend_probs1/c0_non-building', blend='avew')
    pharn._restitch_type('blend_probs1/c1_building', blend='avew')


    def _restitch_type(pharn, mode, blend='vote', force=True):
        """
        hacky code to restitch parts into a whole segmentation based on chip filenames

        mode = 'probs'
        blend = 'avew'
        force = 1
        """
        if mode.startswith('probs'):
            part_paths = sorted(glob.glob(pharn.test_dump_dpath + '/{}/*.npy'.format(mode)))
        else:
            part_paths = sorted(glob.glob(pharn.test_dump_dpath + '/{}/*.png'.format(mode)))

        output_dpath = ub.ensuredir((pharn.test_dump_dpath, 'restiched', mode))
        if not force:
            restitched_paths = sorted(glob.glob(output_dpath + '/*.npy'.format(mode)))
            if len(restitched_paths) > 0:
                return restitched_paths
        restitched_paths = pharn.dataset.task.restitch(output_dpath, part_paths,
                                                       blend=blend)
        return restitched_paths
