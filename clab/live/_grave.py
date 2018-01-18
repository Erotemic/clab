    # train_dpath = ub.truepath(
    #     '~/remote/aretha/data/work/urban_mapper/arch/unet/train/input_4214-yxalqwdk/solver_4214-yxalqwdk_unet_vgg_nttxoagf_a=1,n_ch=5,n_cl=3')
    # load_path = get_snapshot(train_dpath, epoch=202)
    # if False:
    #     train_dpath = ub.truepath(
    #         '~/remote/aretha/data/work/urban_mapper/arch/unet/train/'
    #         'input_8438-haplmmpq/solver_8438-haplmmpq_unet_None_kvterjeu_a=1,c=RGB,n_ch=5,n_cl=3')
    #     load_path = get_snapshot(train_dpath, epoch=258)

    #     eval_dataset = urban_mapper_eval_dataset()
    #     eval_dataset.center_inputs = eval_dataset._original_urban_mapper_normalizer()
    # if False:
    #     train_dpath = ub.truepath(
    #         '~/data/work/urban_mapper2/arch/unet/train/input_4214-guwsobde/'
    #         'solver_4214-guwsobde_unet_mmavmuou_eqnoygqy_a=1,c=RGB,n_ch=5,n_cl=4/')
    #     load_path = get_snapshot(train_dpath)

    #     eval_dataset = urban_mapper_eval_dataset()
    #     eval_dataset.center_inputs = eval_dataset._original_urban_mapper_normalizer()

    # if False:
    #     eval_dataset = urban_mapper_eval_dataset()
    #     from clab.live.urban_train import load_task_dataset
    #     datasets = load_task_dataset('urban_mapper_3d')
    #     eval_dataset.center_inputs = datasets['train']._make_normalizer()
    #     train_dpath = ub.truepath('~/remote/aretha/data/work/urban_mapper2/arch/unet2/train/input_4214-guwsobde/solver_4214-guwsobde_unet2_mmavmuou_tqynysqo_a=1,c=RGB,n_ch=5,n_cl=4')
    #     load_path = get_snapshot(train_dpath, epoch=100)

    # if True:
    #     eval_dataset = urban_mapper_eval_dataset()
    #     from clab.live.urban_train import load_task_dataset
    #     datasets = load_task_dataset('urban_mapper_3d', combine=True)
    #     eval_dataset.center_inputs = datasets['train']._make_normalizer()
    #     train_dpath = ub.truepath('~/remote/aretha/data/work/urban_mapper2/arch/unet2/train/input_8438-xqwzrwfj/solver_8438-xqwzrwfj_unet2_edmtxaov_gksatgso_a=1,c=RGB,n_ch=5,n_cl=4')
    #     # TODO: just read the normalization from the train_dpath instead of
    #     # hacking it together from the train dataset
    #     load_path = get_snapshot(train_dpath, epoch=75)



    # from os.path import dirname, split
    # for ix, fpath in enumerate(ub.ProgIter(restitched_pred[0:10], label='blend instance')):
    #     base_dpath, mode = split(dirname(fpath))
    #     output_dpath = ub.ensuredir(join(base_dpath, 'blend_instance_' + mode))
    #     output_fpath = join(output_dpath, basename(fpath))

    #     pred = imutil.imread(fpath)
    #     cc_labels = eval_dataset.task.instance_label(pred, k=7, n_iters=1,
    #                                                  watershed=True)
    #     big_orig_fpath = eval_dataset.fullres.im_paths[ix]
    #     big_orig = imutil.imread(big_orig_fpath)
    #     big_blend_instance_pred = eval_dataset.task.instance_colorize(cc_labels, big_orig)
    #     imutil.imwrite(output_fpath, big_blend_instance_pred)


    # if False:
    #     train_dpath = ub.truepath(
    #         '~/remote/aretha/data/work/urban_mapper/arch/unet/train/input_4214-yxalqwdk/solver_4214-yxalqwdk_unet_vgg_nttxoagf_a=1,n_ch=5,n_cl=3')
    # load_path = get_snapshot(train_dpath, epoch=202)

    # train_dpath = ub.argval('--train-dpath', default=None)
    # train_dpath = ub.truepath(
    #     '~/data/work/urban_mapper2/arch/unet/train/input_4214-guwsobde/'
    #     'solver_4214-guwsobde_unet_mmavmuou_eqnoygqy_a=1,c=RGB,n_ch=5,n_cl=4/')
    # epoch = ub.argval('--epoch', default=None)

    # if False:
    #     test_dataset.center_inputs = test_dataset._original_urban_mapper_normalizer()
    # else:
    #     datasets['test'].center_inputs = datasets['train']._make_normalizer()


    # def run()
                # if False:
                #     from clab import filters
                #     posterior = filters.crf_posterior(bgr, log_probs)
                #     pred_crf = posterior.argmax(axis=0)
                #     blend_pred_crf = pharn.dataset.task.colorize(pred_crf, bgr)
                #     # color_pred = task.colorize(pred)
                #     output_dict.update({
                #         'blend_pred_crf' + suffix: blend_pred_crf,
                #         'pred_crf' + suffix: pred_crf,
                #     })

                # if pharn.dataset.with_gt:
                #     bgr = imutil.imread(pharn.dataset.inputs.im_paths[ix])
                #     true = imutil.imread(pharn.dataset.inputs.gt_paths[ix])
                #     blend_true = pharn.dataset.task.colorize(true, bgr, alpha=.5)
                #     # color_true = task.colorize(true, alpha=.5)
                #     output_dict['true' + suffix] = true
                #     output_dict['blend_true' + suffix] = blend_true
                #     # output_dict['color_true'] = color_true


                # .astype(np.float32)
                # Just reload rgb data without inverting the transform
                # bgr = imutil.imread(pharn.dataset.inputs.im_paths[ix])

                # output = prob_tensor.data.cpu().numpy()[0]

                # pred = log_probs.argmax(axis=0)

                # pred = argmax.data.cpu().numpy()[0]
                # blend_pred = pharn.dataset.task.colorize(pred, bgr)
                    # 'blend_pred' + suffix: blend_pred,
                    # 'color_pred': color_pred,
                    # 'pred' + suffix: pred,
