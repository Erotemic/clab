# -*- coding: utf-8 -*-
"""
Scripts that execute some experiment

SeeAlso:
    http://mi.eng.cam.ac.uk/projects/segnet/tutorial.html
    https://github.com/alexgkendall/SegNet-Tutorial/blob/master/Example_Models/segnet_model_zoo.md

Developer:
    ln -s ~/code/baseline-algorithms/Semantic_Seg ~/sseg
    cd ~/sseg/virat
"""
from __future__ import absolute_import, division, print_function
from os.path import expanduser
from pysseg.util import gpu_util


def finetune_diva():
    """
    CommandLine:
        export PYTHONPATH=$HOME/code/fletch/build-py3/install/lib/python3.5/site-packages:$PYTHONPATH
        python -m pysseg.tasks DivaV1.run_xval_evaluation

        ~/sseg/caffe-segnet/build/tools/caffe train -gpu 0 \
                -solver ~/sseg/sseg-data/xval-solvers/split_00/segnet_basic_solver_.prototext \
                -weights /home/local/KHQ/jon.crall/.cache/utool/VGG_ILSVRC_16_layers.caffemodel

        ~/sseg/caffe-segnet/build/tools/caffe train -gpu 0 \
                -solver ~/sseg/SegNet/Models/segnet_solver.prototxt \
                -weights /home/local/KHQ/jon.crall/.cache/utool/VGG_ILSVRC_16_layers.caffemodel

    """
    from pysseg import tasks
    task = tasks.DivaV1()
    task.run_xval_evaluation()
    # harness.Harness()


def reproduce_camvid():
    """
    export PYTHONPATH=$HOME/code/fletch/build-py3/install/lib/python3.5/site-packages:$PYTHONPATH

    python -c "from pysseg.special import reproduce_camvid; reproduce_camvid()"
    """
    # Note these weights do not agree with the actual data
    # (when computing median frequency balancing)
    import utool as ut
    # train_imdir  = expanduser('~/sseg/SegNet/CamVid/train')
    # train_gtdir  = expanduser('~/sseg/SegNet/CamVid/trainannot')
    # THIS IS THE REAL CAMVID
    train_imdir  = expanduser('~/store/segnet-exact/SegNet-Tutorial/CamVid/train')
    train_gtdir  = expanduser('~/store/segnet-exact/SegNet-Tutorial/CamVid/trainannot')
    workdir = expanduser('~/data/work/camvid/')

    from pysseg.harness import Harness
    from pysseg.tasks import CamVid

    camvid = CamVid()

    # arch = 'segnet_proper'
    # train_batch_size = 6
    arch = 'segnet_basic'
    train_batch_size = 16

    harn = Harness(
        train_imdir=train_imdir, train_gtdir=train_gtdir, workdir=workdir,
        arch=arch
    )
    harn.task = camvid
    test_weights_fpath = None
    # test_weights_fpath = join(harn.workdir, 'test_weights.caffemodel')

    from os.path import exists
    if test_weights_fpath and exists(test_weights_fpath):
        # TODO: how should we determine if we re-run or use the previously
        # output test_weights.caffemodel?
        print('already have testing weights')
    else:
        harn.train_batch_size = train_batch_size

        snapshot_states = harn.snapshot_states()
        if snapshot_states:
            # Continue training if we have a previous snapshot
            prevstate_fpath = snapshot_states[-1]
            init_pretrained_fpath = None
        else:
            # Otherwise pre-init with some reasonable weights
            init_pretrained_fpath = ut.grab_file_url('http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel')
            prevstate_fpath = None

        if harn.solver_fpath is None:
            harn.gpu_num = gpu_util.find_unused_gpu()
            harn.prepare_solver()
            harn.fit(init_pretrained_fpath=init_pretrained_fpath,
                     prevstate_fpath=prevstate_fpath)
            # , dry=True)
            # stdbuf -i0 -o0 -e0

        harn.gpu_num = gpu_util.find_unused_gpu()
        test_weights_fpath = harn.prepare_for_testing()

    # harn.test_imdir  = '~/sseg/SegNet/CamVid/test'
    # harn.test_gtdir  = '~/sseg/SegNet/CamVid/testannot'
    harn.test_imdir  = expanduser('~/store/segnet-exact/SegNet-Tutorial/CamVid/test')
    harn.test_gtdir  = expanduser('~/store/segnet-exact/SegNet-Tutorial/CamVid/testannot')
    harn.test_weights_fpath = test_weights_fpath
    # harn = Harness(
    #     test_imdir=test_imdir, test_gtdir=test_gtdir, workdir=workdir,
    #     arch=arch, weights_fpath=test_weights_fpath
    # )
    harn.prepare_test_model()
    harn.gpu_num = gpu_util.find_unused_gpu()

    harn.evaluate()
    # Looks good so far...
    # TODO: implement scoring


def pretrained_camvid_segnet_driving():
    import utool as ut
    from pysseg.harness import Harness
    from pysseg.tasks import CamVid

    workdir = expanduser('~/data/work/camvid/')
    arch = 'segnet_proper'  # not really

    harn = Harness(workdir=workdir, arch=arch)
    harn.task = CamVid()
    harn.test_imdir  = '~/sseg/SegNet/CamVid/test'
    harn.test_gtdir  = '~/sseg/SegNet/CamVid/testannot'
    harn.test_suffix = 'pretrained'
    harn.gpu_num = gpu_util.find_unused_gpu()
    pretrained_weights_fpath = ut.grab_file_url(
        'http://mi.eng.cam.ac.uk/~agk34/resources/SegNet/segnet_weights_driving_webdemo.caffemodel')
    harn.test_weights_fpath = pretrained_weights_fpath
    harn.prepare_test_model()
    # harn.prepare_test_input()
    # Driving demo seems to have a different model...
    # perhaps the labels were trained differently here
    # I THINK THIS ONE WAS TRAINED WITH AN EXTRA CATEGORY
    harn.test_model_fpath = expanduser('~/sseg/SegNet/')
    harn.evaluate()


def pretrained_camvid_segnet_proper():
    """
    CommandLine:
        export PYTHONPATH=$HOME/sseg:$PYTHONPATH
        python -c "import pysseg.special; pysseg.special.pretrained_camvid_segnet_proper()"
    """
    from pysseg.harness import Harness
    from pysseg.tasks import CamVid

    workdir = expanduser('~/data/work/camvid/')
    arch = 'segnet_proper'

    harn = Harness(workdir=workdir, arch=arch)
    harn.task = CamVid()
    harn.test_imdir  = '~/sseg/SegNet/CamVid/test'
    harn.test_gtdir  = '~/sseg/SegNet/CamVid/testannot'
    # Pretrained this inference-ready model myself
    test_weights_fpath = '/data/jon.crall/segnet-exact/SegNet-Tutorial/Models/Inference/test_weights.caffemodel'

    import ubelt as ub
    harn.test_dump_dpath = ub.ensuredir('/data/jon.crall/segnet-exact/viz')

    # import ubelt as ub
    # harn.test_dump_dpath = ub.ensuredir((harn.test_dpath, 'temp'))

    # orig_model_fpath = '/data/jon.crall/segnet-exact/SegNet-Tutorial/Models/segnet_inference.prototxt'
    # import utool as ut
    # test_weights_fpath = ut.grab_file_url(
    #     'http://mi.eng.cam.ac.uk/~agk34/resources/SegNet/segnet_iter_30000_timo.caffemodel')
    harn.test_weights_fpath = test_weights_fpath
    harn.prepare_test_input()
    harn.prepare_test_model()

    ub.cmd(r"sed -i 's/_output11//g' /home/local/KHQ/jon.crall/data/work/camvid/arch/segnet_proper/test/input_nqmmrhd/segnet_proper_predict_model.prototext")

    harn.evaluate()

    """
    diff -u --ignore-space-change \
        /data/jon.crall/segnet-exact/SegNet-Tutorial/Models/segnet_inference.prototxt \
        /home/local/KHQ/jon.crall/data/work/camvid/arch/segnet_proper/test/input_nqmmrhd/segnet_proper_predict_model.prototext

    sed -i 's/_output11//g' /home/local/KHQ/jon.crall/data/work/camvid/arch/segnet_proper/test/input_nqmmrhd/segnet_proper_predict_model.prototext
    """


def eval_multi_iter_pretrained_camvid_segnet_proper():
    """
    CommandLine:
        export PYTHONPATH=/home/local/KHQ/jon.crall/sseg:$PYTHONPATH
        python -c "import pysseg.special; eval_multi_iter_pretrained_camvid_segnet_proper()"
    """
    from pysseg.harness import Harness
    from pysseg.tasks import CamVid

    workdir = expanduser('~/data/work/camvid/')
    arch = 'segnet_proper'

    harn = Harness(workdir=workdir, arch=arch)
    harn.task = CamVid()
    harn.test_imdir  = '~/sseg/SegNet/CamVid/test'
    harn.test_gtdir  = '~/sseg/SegNet/CamVid/testannot'

    harn.train_imdir  = '~/sseg/SegNet/CamVid/train'
    harn.train_gtdir  = '~/sseg/SegNet/CamVid/trainannot'

    if True:
        import pysseg.backend.iface_caffe as iface
        from os.path import join, basename
        # Override the snapshots in solver
        harn._snapshot_dpath = '/data/jon.crall/segnet-exact/SegNet-Tutorial/Models/Training'
        # Weight file surgery (translates old names to new names)
        harn.prepare_solver()
        solver_info = iface.parse_solver_info(harn.solver_fpath)
        new_proto = solver_info['train_model_path']

        old_proto = '/data/jon.crall/segnet-exact/SegNet-Tutorial/Models/segnet_train.prototxt'
        orig_snapshot_weights = harn.snapshot_weights()
        # Unoverride snapshot_dpath
        harn._snapshot_dpath = None
        for weight_path in ub.ProgIter(orig_snapshot_weights):
            weight_fname = basename(weight_path)
            new_weight_path = join(harn.snapshot_dpath, weight_fname)
            iface.convert_weights(old_proto, new_proto, weight_path,
                                  new_weight_path)

    harn.prepare_test_input()
    harn.prepare_train_input()
    harn.prepare_solver()
    harn.prepare_test_model()

    # next(harn.deploy_trained_for_testing())
    for _ in harn.deploy_trained_for_testing():
        # hack to evaulate while deploying
        harn.evaulate_all()

    # Pretrained this inference-ready model myself
    # harn.test_weights_fpath = '/data/jon.crall/segnet-exact/SegNet-Tutorial/Models/Inference/test_weights.caffemodel'

    # import ubelt as ub
    # harn.test_dump_dpath = ub.ensuredir((harn.test_dpath, 'temp'))
    # orig_model_fpath = '/data/jon.crall/segnet-exact/SegNet-Tutorial/Models/segnet_inference.prototxt'
    # import utool as ut
    # test_weights_fpath = ut.grab_file_url(
    #     'http://mi.eng.cam.ac.uk/~agk34/resources/SegNet/segnet_iter_30000_timo.caffemodel')
    # harn.test_weights_fpath = test_weights_fpath
    harn.prepare_test_model()
    harn.prepare_test_input()
    harn.evaluate()


def pretrained_camvid_segnet_basic():
    import utool as ut
    from pysseg.harness import Harness
    from pysseg.tasks import CamVid

    workdir = expanduser('~/data/work/camvid/')
    arch = 'segnet_basic'

    harn = Harness(workdir=workdir, arch=arch)
    harn.task = CamVid()
    harn.test_batch_size = 1
    harn.test_imdir  = '~/sseg/SegNet/CamVid/test'
    harn.test_gtdir  = '~/sseg/SegNet/CamVid/testannot'
    harn.gpu_num = gpu_util.find_unused_gpu()
    pretrained_weights_fpath = ut.grab_file_url(
        'http://mi.eng.cam.ac.uk/~agk34/resources/SegNet/segnet_basic_camvid.caffemodel')
    harn.test_weights_fpath = pretrained_weights_fpath
    harn.prepare_test_model()
    harn.prepare_test_predict_dpath()

    """
    diff -u --ignore-space-change \
        /data/jon.crall/segnet-exact/SegNet-Tutorial/Models/segnet_basic_inference.prototxt \
        /home/local/KHQ/jon.crall/data/work/camvid/models/segnet_basic_predict_hmkcclslvxtlpgwozhiaonfvfmvhigxc.prototext
    """

    # harn.prepare_test_input()
    harn.evaluate()


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m pysseg.special
    """
    import ubelt as ub
    ub.doctest_package()
