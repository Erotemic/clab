from pysseg import util
import numpy as np


class CustomSGD(object):
    def __init__(self, solver, model_info, solver_info):
        # Layer multipliers for weight decay and learning rate
        self.solver = solver
        lr_w_mults = {}
        lr_b_mults = {}
        wd_w_mults = {}
        wd_b_mults = {}
        for layer in model_info['layer']:
            if 'param' in layer:
                param = layer['param']
                w, b = param
                lr_w_mults[layer['name']] = w['lr_mult']
                wd_w_mults[layer['name']] = w['decay_mult']
                lr_b_mults[layer['name']] = b['lr_mult']
                wd_b_mults[layer['name']] = b['decay_mult']

        solver = self.solver
        self.momentum = solver_info['momentum']
        self.base_decay = solver_info['weight_decay']
        self.gamma = solver_info['gamma']
        self.stepsize = solver_info['stepsize']
        self.base_lr = solver_info['base_lr']

        self.momentum_hist = momentum_hist = {}
        for layer in solver.net.params:
            m_w = np.zeros_like(solver.net.params[layer][0].data)
            m_b = np.zeros_like(solver.net.params[layer][1].data)
            momentum_hist[layer] = [m_w, m_b]

    def fit(self, prevstate_fpath):
        from pysseg.backend.find_segnet_caffe import import_segnet_caffe
        from pysseg.backend import iface_caffe as iface
        harn = self.harn
        caffe = import_segnet_caffe(gpu_num=harn.gpu_num)

        harn.prepare_solver()

        solver_info = iface.parse_solver_info(harn.solver_fpath)

        model_fpath = solver_info['train_model_path']
        model_info = iface.parse_model_info(model_fpath)

        self.solver = caffe.SGDSolver(harn.solver_fpath)

        pretrained = harn.init_pretrained_fpath

        if prevstate_fpath is not None:
            print('Restoring State from {}'.format(prevstate_fpath))
            self.solver.restore(prevstate_fpath)
        elif pretrained is not None:
            print('Loading pretrained model weights from {}'.format(pretrained))
            self.solver.net.copy_from(pretrained)

        layers = model_info['layer']
        start_layer = layers[1]['name']

        # Do iterations over batches
        for bx in range(solver_info['max_iter']):
            self.load_batch_data(bx)
            outputs = self.solver.net.forward(start=start_layer)
            import ubelt as ub
            print(ub.repr2(outputs))
            self.solver.net.backwards()
            # need to manually update weights. bleh...
            self.update(bx)
            # diffs = self.solver.net.backward()
            # Here we could monitor the progress by testing occasionally,
            # plotting loss, error, gradients, activations etc.
            # for layer in net.layers:
            #     for blob in layer.blobs:
            #         print('blob = {!r}'.format(blob.diff))

    def load_batch_data(self, bx):
        """
        bx = 0
        """
        offset = bx * self.batch_size
        blob_data = self.solver.net.blobs['data'].data
        blob_label = self.solver.net.blobs['label'].data
        assert blob_data.shape[0] == self.batch_size
        # TODO: shuffle indices
        for jx in range(self.batch_size):
            # push data into the network
            ix = offset + jx
            im_hwc = util.imread(self.input.im_paths[ix])
            gt_hwc = util.atleast_nd(util.imread(self.input.gt_paths[ix]), n=3)
            # TODO: might we do on-the-fly augmentation here?
            im_chw = np.transpose(im_hwc, (2, 0, 1)).astype(np.float32)
            gt_chw = np.transpose(gt_hwc, (2, 0, 1)).astype(np.float32)
            blob_data[jx, :, :, :] = im_chw
            blob_label[jx, :, :, :] = gt_chw

    def update(self, bx):
        """
        We really should be doing this on the GPU
        """

        # for it in range(1, niter+1):
        #     solver.net.forward()  # fprop
        #     solver.net.backward()  # bprop
        #     # *manually update*

        # https://github.com/BVLC/caffe/issues/1855
        # UPDATE STEP
        for layer in self.solver.net.params:
            wb_curr = self.solver.net.params[layer]
            wb_hist = self.momentum_hist[layer]

            w_rate = self.base_lr * self.lr_w_mults[layer]
            b_rate = self.base_lr * self.lr_b_mults[layer]
            w_decay = self.base_decay * self.wd_w_mults[layer]
            b_decay = self.base_decay * self.wd_b_mults[layer]

            wb_hist[0] = (wb_hist[0] * self.momentum) + ((wb_curr[0].diff + w_decay * wb_curr[0].data) * w_rate)
            wb_hist[1] = (wb_hist[1] * self.momentum) + ((wb_curr[1].diff + b_decay * wb_curr[1].data) * b_rate)

            self.solver.net.params[layer][0].data[...] -= self.momentum_hist[layer][0]
            self.solver.net.params[layer][1].data[...] -= self.momentum_hist[layer][1]
            self.solver.net.params[layer][0].diff[...] = 0
            self.solver.net.params[layer][1].diff[...] = 0
        self.base_lr = self.base_lr * np.power(self.gamma, (np.floor(bx / self.stepsize)))


class CaffeTrainer(object):

    def __init__(self, harn):
        self.harn = harn
        self.solver = None
        self.batch_size = harn.train_batch_size
        self.input = harn.train

    def fit(self, prevstate_fpath):
        """
        References:
            pass
            https://github.com/BVLC/caffe/blob/master/src/caffe/proto/caffe.proto#L106-L109

        Ignore:
            >>> from pysseg._train import *
            >>> from os.path import expanduser
            >>> from pysseg import tasks
            >>> import pysseg
            >>> task = tasks.DivaV1(clean=2)
            >>> pretrained = 'segnet_proper_camvid.caffemodel'
            >>> harn = task.harness_from_xval(0, pretrained=pretrained)
            >>> harn.train_batch_size = 3
            >>> solverstate_fpath = prevstate_fpath = expanduser('~/remote/aretha/data/work/v1-fresher/harness/xval/split_00/arch/segnet_proper/train/input_4212-vmidtlwq/solver_4212-vmidtlwq_segnet_proper_agmrdts_fopvszl/snapshots/_iter_2000.solverstate')
            >>> prevstate_fpath = None
            >>> self = CaffeTrainer(harn)
        """
        from pysseg.backend.find_segnet_caffe import import_segnet_caffe
        from pysseg.backend import iface_caffe as iface
        harn = self.harn
        caffe = import_segnet_caffe(gpu_num=harn.gpu_num)

        harn.prepare_solver()

        solver_info = iface.parse_solver_info(harn.solver_fpath)
        snapshot_iters = solver_info['snapshot']

        # Assuming that the solver .prototxt has already been configured including
        # the corresponding training and testing network definitions (as .prototxt).
        self.solver = caffe.SGDSolver(harn.solver_fpath)

        pretrained = harn.init_pretrained_fpath

        if prevstate_fpath is not None:
            print('Restoring State from {}'.format(prevstate_fpath))
            self.solver.restore(prevstate_fpath)
        elif pretrained is not None:
            # https://github.com/BVLC/caffe/issues/3336
            print('Loading pretrained model weights from {}'.format(pretrained))
            self.solver.net.copy_from(pretrained)

        # net = self.solver.net
        # Do iterations over batches
        bx = 0
        while bx < solver_info['max_iter']:
            # Run until we can produce a snapshot
            self.solver.step(snapshot_iters)
            bx += snapshot_iters
            # We should now perform a test run before we continue training
