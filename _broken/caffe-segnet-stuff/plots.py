import json
import pandas as pd
import ubelt as ub
import glob
from os.path import dirname, basename, join, exists
import numpy as np

from pysseg import getLogger
logger = getLogger(__name__)
print = logger.info


def _set_mpl_rcparams():
    import matplotlib as mpl
    mpl.style.use('ggplot')
    mpl.rcParams['ytick.labelsize'] = 18
    mpl.rcParams['xtick.labelsize'] = 18
    mpl.rcParams['axes.labelsize'] = 18
    mpl.rcParams['axes.titlesize'] = 18
    mpl.rcParams['legend.fontsize'] = 16
    mpl.rcParams['legend.loc'] = 'lower right'


def _show_color_key(task):
    """
    Example:
        >>> from pysseg.tasks import *
        >>> task = DivaV1(clean=0)
        >>> for scene in ub.ProgIter(task.scene_ids, label='make full scene', verbose=3):
        >>>     task.make_full_scene(scene)
    """
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt
    patches = [
        patches.Patch(color=np.array(bgr[::-1]) / 255, label=classname)
        for classname, bgr in task.class_colors.items()
    ]
    plt.legend(handles=patches)
    plt.show()


def get_iters_vs_miou(harn):
    from pysseg.util import jsonutil
    import pysseg.backend.iface_caffe as iface
    harn.prepare_test_model()
    test_weight_dpaths = harn.find_test_weights_dpaths()
    # for test_weights_dpath in test_weight_dpaths:
    #     harn.test_weights_dpath = test_weights_dpath
    #     harn.test_weights_fpath = ub.readfrom(join(test_weights_dpath, 'test_weights.caffemodel.lnk'))
    #     # if not exists(join(harn.test_weights_dpath, 'pred')):
    #     results_fpath = join(harn.test_weights_dpath, 'results.json')
    #     if exists(results_fpath):
    #         results = json.load(results_fpath)

    iter_results = {}
    for test_weights_dpath in test_weight_dpaths:
        results_fpath = join(test_weights_dpath, 'results.json')
        if exists(results_fpath):
            iterno = iface.snapshot_iterno(test_weights_dpath)
            results = json.load(open(results_fpath, 'r'))
            ious = eval(results['ious'])
            iter_results[iterno] = ious

    iter_df = pd.DataFrame(iter_results)
    iter_df.columns.name = 'iterno'
    iter_df.index.name = 'class'

    fpath = join(harn.test_dpath, 'iter_ious.json')
    jsonutil.write_json(fpath, iter_df)

    iter_df = iter_df.drop([57], axis=1)
    iter_df.drop(harn.task.ignore_classnames).mean(axis=0)

    if False:
        """
        ffmpeg -y -f image2 -i ~/aretha/store/data/work/camvid/arch/segnet_proper/test/input_nqmmrhd/weights_abvroyo_segnet_proper_None_xwfmwfo_00040000/blend_pred/%*.png -crf 25  -vcodec libx264  -vf "setpts=4*PTS" camvid-results.avi

        ffmpeg -y -f image2 -i out_haul83/%*.png -vcodec mpeg4 -vf "setpts=10*PTS" haul83-results.avi

        """
        # move to computer with plottool
        iter_df = pd.read_json('/home/joncrall/aretha/store/data/work/camvid/arch/segnet_proper/test/input_nqmmrhd/iter_ious.json')

        import plottool as pt
        pt.qtensure()

        from pysseg.tasks import CamVid
        task = CamVid()

        iter_miou = iter_df.drop(task.ignore_classnames).mean(axis=0)
        iter_miou = iter_miou.sort_index()

        _set_mpl_rcparams()

        fig = pt.figure(fnum=1, pnum=(1, 1, 1))
        ax = pt.gca()
        iter_miou.plot(ax=ax)
        ax.set_xlabel('train iters')
        ax.set_ylabel('test mIoU')
        ax.set_title('Reproduced CamVid Results (init using VGG)')
        ub.ensuredir('result_plots')
        from pysseg.draw import render_figure_to_image
        import cv2
        cv2.imwrite('result_plots/miou.png', render_figure_to_image(fig, dpi=100, transparent=True))

        fig = pt.figure(fnum=2, pnum=(1, 1, 1))
        ax = pt.gca()
        iter_iou = iter_df.drop(task.ignore_classnames).T.sort_index()

        # sort by results
        iter_iou = iter_iou[iter_iou.iloc[-1].sort_values().index[::-1]]

        colors = [tuple(np.array(v[::-1]) / 255) + (1,) for v in
                  ub.take(task.class_colors, iter_iou.columns)]

        iter_iou.plot(ax=ax, colors=colors, lw=4)
        ax.set_xlabel('train iters')

        ax.set_ylabel('test IoU')
        ax.set_title('Reproduced CamVid Results (init using VGG)')
        ub.ensuredir('result_plots')
        from pysseg.draw import render_figure_to_image
        cv2.imwrite('result_plots/perclass_iou.png', render_figure_to_image(fig, dpi=100, transparent=True))


def draw_confusion(df, fpath, logscale=False):
    # Draw the confusion matrix
    from pysseg.util import utildraw
    import pandas as pd
    import cv2
    if isinstance(df, dict):
        df = pd.DataFrame.from_dict(df)
    import matplotlib.pyplot as plt
    fig = utildraw.figure(fnum=1, pnum=(1, 1, 1), doclf=True)
    ax = plt.gca()
    ax = utildraw.pandas_plot_matrix(df, rot=80, ax=ax, label='confusion', logscale=logscale)
    # ax.set_xlabel('')
    ax.figure.set_size_inches(10, 8)
    utildraw.adjust_subplots(bottom=.1, left=.1, right=.9, top=.7)
    cv2.imwrite(fpath, utildraw.render_figure_to_image(fig, transparent=True))


class SnapshotPlots(object):
    """
    Draws information as a function of snapshot n_iters

    Ignore:
        test_dir = expanduser('~/aretha/data/work/v1-fresher/harness/xval/split_00/arch/segnet_proper/test/input_12-mprldfsg')
        test_dir = expanduser('~/aretha/data/work/v1-fresher/harness/xval/split_00/arch/segnet_proper/test/input_380-tlpwyzvq')
        self = SnapshotPlots(test_dir)
    """

    def __init__(self, test_dir):
        self.test_dir = test_dir

    def _read_snapshot_results(self):
        # Parse the output of the test dir
        results_fpaths = glob.glob(join(self.test_dir, 'weights_*/results.json'))
        result_infos = []
        for fpath in results_fpaths:
            dname = basename(dirname(fpath))
            parts = dname.split('_')
            # Weird parsing because arch may contain an underscore.
            item = {}
            item['train_input_id'] = parts[1]
            item['init_id'], item['hyper_id'], item['n_iters'] = parts[-3:]
            item['arch_id'] = '_'.join(parts[2:-3])
            item['n_iters'] = int(item['n_iters'])
            item['fpath'] = fpath
            result_infos.append(item)
        snap_paths_df = pd.DataFrame(result_infos)

        # Group items by their overall train id
        train_id_keys = ('arch_id', 'hyper_id', 'init_id', 'train_input_id')
        groups = list(snap_paths_df.groupby(train_id_keys))
        assert len(groups) == 1, 'TODO, support train_id comparisons'
        for train_id, group in groups:

            # Short by number of iterations
            group = group.sort_values('n_iters')

            # Read the results data into a dictionary
            group_datas = []
            for n_iters, fpath in zip(group.n_iters, group.fpath):
                item = json.loads(ub.readfrom(fpath))
                item['n_iters'] = n_iters
                item['fpath'] = fpath
                group_datas.append(item)

            yield train_id, group_datas

    def _draw_iters_vs_global_metrics(self):
        import matplotlib.pyplot as plt
        from pysseg.util import utildraw

        _set_mpl_rcparams()
        for train_id, group_datas in self._read_snapshot_results():
            keys = ['n_iters', 'global_acc', 'class_acc', 'global_miou']
            subdata = [ub.dict_subset(d, keys) for d in group_datas]
            results_df = pd.DataFrame(subdata).set_index('n_iters', drop=True)

            fig = utildraw.figure(fnum=1, pnum=(1, 1, 1), doclf=True)
            ax = plt.gca()
            results_df.plot(ax=ax)
            ax.set_title('train_id = ' + str(train_id))

            n_iters = results_df.global_miou.argmax()
            print(results_df.loc[n_iters])

            fig.set_size_inches(10, 8)
            utildraw.savefig2(fig, 'iters_vs_global_metrics.png')

    def _draw_iters_vs_local_ious(self):
        import matplotlib.pyplot as plt
        from pysseg.util import utildraw

        _set_mpl_rcparams()
        for train_id, group_datas in self._read_snapshot_results():
            def dict_union(*args):
                d3 = {}
                for d in args:
                    d3.update(d)
                return d3

            subdata = [dict_union(ub.dict_subset(d, ['n_iters']), d['ious'])
                       for d in group_datas]
            iou_df = pd.DataFrame(subdata).set_index('n_iters', drop=True)

            iou_df = iou_df.drop('Unannotated', axis=1)

            fig = utildraw.figure(fnum=1, pnum=(1, 1, 1), doclf=True)
            ax = plt.gca()
            iou_df.plot(ax=ax)
            ax.set_title('train_id = ' + str(train_id))
            ax.set_ylabel('class iou')

            # n_iters = iou_df.global_miou.argmax()
            # print(iou_df.loc[n_iters])

            fig.set_size_inches(10, 8)
            utildraw.savefig2(fig, 'iters_vs_local_ious.png')

    def _draw_iters_vs_local_accuracy(self):
        import matplotlib.pyplot as plt
        from pysseg.util import utildraw

        _set_mpl_rcparams()
        for train_id, group_datas in self._read_snapshot_results():
            subdata = []
            for d in group_datas:
                cfsn =  pd.DataFrame(d['total_cfsn'])
                n_ii = np.diag(cfsn)
                t_i = cfsn.sum(axis=1)
                perclass_acc = n_ii / t_i
                item = perclass_acc.to_dict()
                item['n_iters'] = d['n_iters']
                subdata.append(item)
            df = pd.DataFrame(subdata).set_index('n_iters', drop=True)
            df = df.drop('Unannotated', axis=1)

            fig = utildraw.figure(fnum=1, pnum=(1, 1, 1), doclf=True)
            ax = plt.gca()
            df.plot(ax=ax)
            ax.set_title('train_id = ' + str(train_id))
            ax.set_ylabel('class accuracy')

            # n_iters = iou_df.global_miou.argmax()
            # print(iou_df.loc[n_iters])

            fig.set_size_inches(10, 8)
            utildraw.savefig2(fig, 'iters_vs_local_class_acc.png')

        # Draw the confusion matrix
        df = cfsn.drop('Unannotated', axis=1).drop('Unannotated', axis=0)
        from pysseg.util import utildraw
        import pandas as pd
        import cv2
        if isinstance(df, dict):
            df = pd.DataFrame.from_dict(df)
        import matplotlib.pyplot as plt
        fig = utildraw.figure(fnum=1, pnum=(1, 1, 1), doclf=True)
        ax = plt.gca()
        ax = utildraw.pandas_plot_matrix(df, rot=80, ax=ax, label='confusion',
                                         showvals=True,
                                         zerodiag=True, logscale=True)
        ax.set_ylabel('real')
        ax.set_xlabel('pred')
        ax.figure.set_size_inches(10, 8)
        utildraw.adjust_subplots(bottom=.1, left=.1, right=.9, top=.7)
        cv2.imwrite('acc_cfsn.png', utildraw.render_figure_to_image(fig, transparent=True))


# def draw_iters_vs_metric(test_dir):
#     snap_paths_df = _parse_snapshot_result_paths()
#         item = ub.dict_subset(result_data, [
#             'global_acc', 'class_acc', 'global_miou'])
#     results_df = pd.DataFrame(datas)
#     results_df = results_df.set_index('n_iters', drop=True)


#     import matplotlib.pyplot as plt
#     from pysseg.util import utildraw
#     fig = utildraw.figure(fnum=1, pnum=(1, 1, 1))
#     ax = plt.gca()
#     results_df.plot(ax=ax)

#     n_iters = results_df.global_miou.argmax()
#     results_df.loc[n_iters]

#     fig.set_size_inches(10, 8)
#     fig.savefig('iters_vs_metrics.png', transparent=True)
