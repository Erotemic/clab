# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
from os.path import join  # NOQA
import ubelt as ub
import itertools as it
import os
import numpy as np
import torch
import torch.nn
import torchvision

from clab import util
from clab.util import hashutil
from clab import augment
from clab import xpu_device
from clab import models
from clab import hyperparams
from clab import fit_harness
from clab import im_loaders
from clab import criterions
from clab import util  # NOQA
from clab.fit_harness import get_snapshot

# from clab.util import imutil


def positive_sample(pccs, per_cc=None):
    import utool as ut
    rng = ut.ensure_rng(2039141610, 'python')
    yield from util.roundrobin(
        ut.random_combinations(cc, size=2, num=per_cc, rng=rng)
        for cc in pccs
    )


def negative_sample(pccs, per_pair=None):
    import utool as ut
    rng = ut.ensure_rng(2039141610, 'python')
    neg_pcc_pairs = ut.random_combinations(pccs, 2, rng=rng)
    yield from util.roundrobin(
        ut.random_product((cc1, cc2), num=per_pair, rng=rng)
        for cc1, cc2 in neg_pcc_pairs
    )


def pair_sampler(class_to_id, npos=100000, nneg='pos'):
    pccs = sorted(map(tuple, map(sorted, class_to_id.values())))

    pos_pairs = []
    for pair in positive_sample(pccs):
        pos_pairs.append(tuple(sorted(pair)))
        if len(pos_pairs) >= npos:
            break

    if nneg == 'pos':
        nneg = len(pos_pairs)

    neg_pairs = []
    for pair in negative_sample(pccs):
        neg_pairs.append(tuple(sorted(pair)))
        if len(neg_pairs) >= nneg:
            break

    fpaths1, fpaths2 = [], []
    labels = []

    pos_label = 1
    for p1, p2 in pos_pairs:
        fpaths1.append(p1)
        fpaths2.append(p2)
        labels.append(pos_label)

    neg_label = 0
    for p1, p2 in neg_pairs:
        fpaths1.append(p1)
        fpaths2.append(p2)
        labels.append(neg_label)

    return fpaths1, fpaths2, labels


class PairDataset(torch.utils.data.Dataset):
    # DIM = 224
    # DIM = 300
    # DIM = 324
    # DIM = 328
    # DIM = 330
    # DIM = 332
    # DIM = 226
    # DIM = 380
    # DIM = 350
    # DIM = 400
    # DIM = 412
    # DIM = 414
    # DIM = 416  # THIS WORKED BEFORE (WHY NOT AGAIN?)
    # DIM = 418
    # DIM = 420
    # DIM = 422
    def __init__(self, dim):
        transform = 'default'
        self.dim = dim
        self.colorspace = 'RGB'
        if transform  == 'default':
            if not hasattr(torchvision.transforms, 'Resize'):
                torchvision.transforms.Resize = torchvision.transforms.Scale
            transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize((dim, dim)),
                torchvision.transforms.ToTensor(),
                # torchvision.transforms.Normalize([0.5, 0.5, 0.5],
                #                                  [0.225, 0.225, 0.225]),
            ])
        self.transform = transform
        self.loader = im_loaders.pil_loader
        # self.loader = im_loaders.np_loader
        self.augment = False
        # self.center_inputs = None

    def show_sample(self):
        vis_dataloader = torch.utils.data.DataLoader(self, shuffle=True,
                                                     num_workers=8,
                                                     batch_size=8)
        example_batch = next(iter(vis_dataloader))
        concatenated = torch.cat((example_batch[0], example_batch[1]), 0)
        import matplotlib.pyplot as plt
        tensor = torchvision.utils.make_grid(concatenated)
        im = tensor.numpy().transpose(1, 2, 0)
        plt.imshow(im)


def siam_augment(img1, img2, label, rng):
    # PIL augmentation is much much faster (for affine warping)
    # convert from pil to numpy
    # img1 = np.array(img1)
    # img2 = np.array(img2)

    # Augment intensity independently
    img1 = augment.online_intensity_augment(img1, rng)
    img2 = augment.online_intensity_augment(img2, rng)

    if rng.rand() > .5:
        # Do the same perterbation half the time
        img1, img2 = augment.online_affine_perterb([img1, img2], rng)
    else:
        # Do different perterbation half the time
        if label == 1:
            # however, flips must be the same
            kw = {'flip_lr_prob': int(rng.rand() > .5),
                  'flip_ud_prob': int(rng.rand() > .5)}
        else:
            kw = {}
        # For different images do different affine peterbs
        # half the time.
        img1, = augment.online_affine_perterb([img1], rng, **kw)
        img2, = augment.online_affine_perterb([img2], rng, **kw)
    return img1, img2


class LabeledPairDataset(PairDataset):
    """
    Ignore:
        >>> from clab.live.siam_train import *
        >>> train_dataset, vali_dataset, test_dataset = ibeis_datasets('PZ_MTEST')
        >>> ut.qtensure()
        >>> self = train_dataset
        >>> self.augment = True
        >>> self.show_sample()
    """

    def __init__(self, img1_fpaths, img2_fpaths, labels, dim=224):
        super(LabeledPairDataset, self).__init__(dim=dim)
        assert len(img1_fpaths) == len(img2_fpaths)
        assert len(labels) == len(img2_fpaths)
        self.img1_fpaths = list(img1_fpaths)
        self.img2_fpaths = list(img2_fpaths)
        self.labels = list(labels)

        # Hack for input id
        if True:
            depends = [
                self.img1_fpaths,
                self.img2_fpaths,
                self.labels
            ]
            hashid = hashutil.hash_data(depends)[:8]
            self.input_id = '{}-{}'.format(len(self), hashid)

        import utool as ut
        rng = ut.ensure_rng(3432, 'numpy')
        self.rng = rng

    def __len__(self):
        return len(self.img1_fpaths)

    def class_weights(self):
        import pandas as pd
        label_freq = pd.value_counts(self.labels)
        class_weights = label_freq.median() / label_freq
        class_weights = class_weights.sort_index().values
        class_weights = torch.from_numpy(class_weights.astype(np.float32))
        return class_weights

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image1, image2, label)
        """
        fpath1 = self.img1_fpaths[index]
        fpath2 = self.img2_fpaths[index]
        label = self.labels[index]

        img1 = self.loader(fpath1)
        img2 = self.loader(fpath2)

        if self.augment:
            img1, img2 = siam_augment(img1, img2, label, rng=self.rng)

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, label


class RandomBalancedIBEISSample(PairDataset):
    def __init__(self, pblm, pccs, dim=224):
        super(RandomBalancedIBEISSample, self).__init__(dim=dim)
        import utool as ut
        chip_config = {'resize_dim': 'wh', 'dim_size': (self.dim, self.dim)}
        self.pccs = pccs
        all_aids = ut.flatten(pccs)
        all_fpaths = pblm.infr.ibs.depc_annot.get(
            'chips', all_aids, read_extern=False, colnames='img',
            config=chip_config)

        self.aid_to_fpath = dict(zip(all_aids, all_fpaths))

        # self.multitons_pccs = [pcc for pcc in pccs if len(pcc) > 1]
        self.pos_pairs = []

        # SAMPLE ALL POSSIBLE POS COMBINATIONS AND IGNORE INCOMPARABLE
        self.infr = pblm.infr
        # todo each sample should really get a weight depending on num aids in
        # its pcc
        for pcc in pccs:
            if len(pcc) >= 2:
                edges = np.array(list(it.starmap(self.infr.e_, it.combinations(pcc, 2))))
                is_comparable = self.is_comparable(edges)
                pos_edges = edges[is_comparable]
                self.pos_pairs.extend(list(pos_edges))
        rng = ut.ensure_rng(563401, 'numpy')
        self.pyrng = ut.ensure_rng(564043, 'python')
        self.rng = rng

        if True:
            depends = [
                sorted(map(sorted, self.pccs)),
            ]
            hashid = hashutil.hash_data(depends)[:8]
            self.input_id = '{}-{}'.format(len(self), hashid)

    def __len__(self):
        return len(self.pos_pairs) * 2

    def class_weights(self):
        class_weights = torch.FloatTensor([1.0, 1.0])
        return class_weights

    def is_comparable(self, edges):
        from ibeis.algo.graph.state import POSTV, NEGTV, INCMP, UNREV  # NOQA
        infr = self.infr
        def _check(u, v):
            if infr.incomp_graph.has_edge(u, v):
                return False
            elif infr.pos_graph.has_edge(u, v):
                # Only override if the evidence says its positive
                # otherwise guess
                ed = infr.get_edge_data((u, v)).get('evidence_decision', UNREV)
                if ed == POSTV:
                    return True
                else:
                    return np.nan
            elif infr.neg_graph.has_edge(u, v):
                return True
            return np.nan
        flags = np.array([_check(*edge) for edge in edges])
        # hack guess if comparable based on viewpoint
        guess_flags = np.isnan(flags)
        need_edges = edges[guess_flags]
        need_flags = infr.ibeis_guess_if_comparable(need_edges)
        flags[guess_flags] = need_flags
        return np.array(flags, dtype=np.bool)

    def get_aidpair(self, index):
        if index % 2 == 0:
            # Get a positive pair if the index is even
            aid1, aid2 = self.pos_pairs[index // 2]
            label = 1
        else:
            # Get a random negative pair if the index is odd
            pcc1, pcc2 = self.pyrng.sample(self.pccs, k=2)
            while pcc1 is pcc2:
                pcc1, pcc2 = self.pyrng.sample(self.pccs, k=2)
            aid1 = self.pyrng.sample(pcc1, k=1)[0]
            aid2 = self.pyrng.sample(pcc2, k=1)[0]
            label = 0
        return aid1, aid2, label

    def from_edge(self, aid1, aid2):
        fpath1 = self.aid_to_fpath[aid1]
        fpath2 = self.aid_to_fpath[aid2]

        img1 = self.loader(fpath1)
        img2 = self.loader(fpath2)

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2

    def __getitem__(self, index):
        aid1, aid2, label = self.get_aidpair(index)

        fpath1 = self.aid_to_fpath[aid1]
        fpath2 = self.aid_to_fpath[aid2]

        img1 = self.loader(fpath1)
        img2 = self.loader(fpath2)

        if self.augment:
            img1, img2 = siam_augment(img1, img2, label, rng=self.rng)

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        if self.rng.rand() > .5:
            img1, img2 = img2, img1

        return img1, img2, label

        # rng = ut.ensure_rng(44324324, 'python')
        # rng.sample(vali, k=1)
        # rng.choice(np.arange(len(vali)), size=2, replace=False)


def randomized_ibeis_dset(dbname, dim=224):
    """
        >>> from clab.live.siam_train import *
        >>> datasets = randomized_ibeis_dset('PZ_MTEST')
        >>> ut.qtensure()
        >>> self = datasets['train']
        >>> self.augment = True
        >>> self.show_sample()
    """
    # from clab.live.siam_train import *
    # dbname = 'PZ_MTEST'
    import utool as ut
    from ibeis.algo.verif import vsone
    # pblm = vsone.OneVsOneProblem.from_empty('PZ_MTEST')
    pblm = vsone.OneVsOneProblem.from_empty(dbname)

    pccs = list(pblm.infr.positive_components())
    pcc_freq = list(map(len, pccs))
    freq_grouped = ub.group_items(pccs, pcc_freq)

    # Simpler very randomized sample strategy
    train_pccs = []
    vali_pccs = []
    test_pccs = []
    import math

    # vali_frac = .1
    test_frac = .1
    vali_frac = 0

    for i, group in freq_grouped.items():
        group = ut.shuffle(group, rng=432232 + i)
        n_test = 0 if len(group) == 1 else math.ceil(len(group) * test_frac)
        test, learn = group[:n_test], group[n_test:]
        n_vali = 0 if len(group) == 1 else math.ceil(len(learn) * vali_frac)
        vali, train = group[:n_vali], group[-n_vali:]
        train_pccs.extend(train)
        test_pccs.extend(test)
        vali_pccs.extend(vali)

    test_dataset = RandomBalancedIBEISSample(pblm, test_pccs, dim=dim)
    train_dataset = RandomBalancedIBEISSample(pblm, train_pccs, dim=dim)
    vali_dataset = RandomBalancedIBEISSample(pblm, vali_pccs, dim=dim)
    train_dataset.augment = True

    datasets = {
        'train': train_dataset,
        # 'vali': vali_dataset,
        'test': test_dataset,
    }
    return datasets


def ibeis_datasets(dbname='PZ_MTEST', dim=224):
    """
    Example:
        >>> from clab.live.siam_train import *
        >>> from ibeis.algo.verif.vsone import *  # NOQA
        >>> pblm = OneVsOneProblem.from_empty('GZ_Master1')
        >>> ibs = pblm.infr.ibs
        >>> pblm.load_samples()
        >>> samples = pblm.samples
        >>> samples.print_info()
        >>> xval_kw = pblm.xval_kw.asdict()
        >>> skf_list = pblm.samples.stratified_kfold_indices(**xval_kw)
        >>> train_idx, test_idx = skf_list[0]
        >>> aids1, aids2 = pblm.samples.aid_pairs[train_idx].T
        >>> labels = pblm.samples['match_state'].y_enc[train_idx]
        >>> labels = (labels == 1).astype(np.int64)
        >>> chip_config = {'resize_dim': 'wh', 'dim_size': (224, 224)}
        >>> img1_fpaths = ibs.depc_annot.get('chips', aids1, read_extern=False, colnames='img', config=chip_config)
        >>> img2_fpaths = ibs.depc_annot.get('chips', aids2, read_extern=False, colnames='img', config=chip_config)
        >>> self = LabeledPairDataset(img1_fpaths, img2_fpaths, labels)
        >>> img1, img2, label = self[0]
    """
    # wrapper around the RF vsone problem
    from ibeis.algo.verif import vsone
    # pblm = vsone.OneVsOneProblem.from_empty('PZ_MTEST')
    pblm = vsone.OneVsOneProblem.from_empty(dbname)
    ibs = pblm.infr.ibs
    pblm.load_samples()
    samples = pblm.samples
    samples.print_info()
    xval_kw = pblm.xval_kw.asdict()
    skf_list = pblm.samples.stratified_kfold_indices(**xval_kw)

    if False:
        pass

    def load_dataset(subset_idx):
        aids1, aids2 = pblm.samples.aid_pairs[subset_idx].T
        labels = pblm.samples['match_state'].y_enc[subset_idx]

        # train only on positive-vs-negative (ignore incomparable)
        labels = (labels == 1).astype(np.int64)

        chip_config = {'resize_dim': 'wh', 'dim_size': (dim, dim)}
        img1_fpaths = ibs.depc_annot.get('chips', aids1, read_extern=False,
                                         colnames='img', config=chip_config)
        img2_fpaths = ibs.depc_annot.get('chips', aids2, read_extern=False,
                                         colnames='img', config=chip_config)
        img1_fpaths = list(map(str, img1_fpaths))
        img2_fpaths = list(map(str, img2_fpaths))
        labels = list(map(int, labels))
        dataset = LabeledPairDataset(img1_fpaths, img2_fpaths, labels, dim=dim)
        return dataset

    learn_idx, test_idx = skf_list[0]
    train_idx, val_idx = pblm.samples.subsplit_indices(learn_idx, n_splits=10)[0]

    # Split everything in the learning set into training / validation
    train_dataset = load_dataset(train_idx)
    vali_dataset = load_dataset(val_idx)
    test_dataset = load_dataset(test_idx)

    return train_dataset, vali_dataset, test_dataset


# def mnist_datasets():
#     root = ub.ensuredir(os.path.expanduser('~'), 'data', 'mnist')
#     dset = torchvision.datasets.MNIST(root, download=True)
#     pass


def att_faces_datasets(dim=224):
    """
    https://github.com/harveyslash/Facial-Similarity-with-Siamese-Networks-in-Pytorch

        >>> from clab.live.siam_train import *
        >>> train_dataset, vali_dataset, test_dataset = att_faces_datasets()
        train_dataset[0][0].shape

        fpath = train_dataset.img1_fpaths[0]
    """
    def ensure_att_dataset():
        def unzip(zip_fpath, dpath=None, verbose=1):
            """
            Extracts all members of a zipfile.

            Args:
                zip_fpath (str): path of zip file to unzip.
                dpath (str): directory to unzip to. If not specified, it defaults
                    to a folder parallel to the zip file (excluding the extension).
                verbose (int): verbosity level
            """
            import zipfile
            from os.path import splitext
            from ubelt import progiter
            if dpath is None:
                dpath = splitext(zip_fpath)[0]
            with zipfile.ZipFile(zip_fpath, 'r') as zf:
                members = zf.namelist()
                prog = progiter.ProgIter(members, verbose=verbose,
                                         label='unzipping')
                for zipinfo in prog:
                    zf.extract(zipinfo, path=dpath, pwd=None)
            return dpath

        faces_zip_fpath = ub.grabdata('http://www.cl.cam.ac.uk/Research/DTG/attarchive/pub/data/att_faces.zip')
        from os.path import splitext
        dpath = splitext(faces_zip_fpath)[0]
        if not os.path.exists(dpath):
            dpath = unzip(faces_zip_fpath, dpath=dpath)
        return dpath

    # Download the data if you dont have it
    dpath = ensure_att_dataset()

    import torchvision.datasets
    torchvision.datasets.folder.IMG_EXTENSIONS += ['.pgm']
    im_dset = torchvision.datasets.ImageFolder(root=dpath)
    class_to_id = ub.group_items(*zip(*im_dset.imgs))

    import utool as ut
    names = sorted(list(class_to_id.keys()))
    names = ut.shuffle(names, rng=10)
    learn, test = names[:40], names[40:]
    train, vali = learn[:35], learn[35:]
    print('train = {!r}'.format(len(train)))
    print('vali = {!r}'.format(len(vali)))
    print('test = {!r}'.format(len(test)))

    train_dataset = LabeledPairDataset(*pair_sampler(ub.dict_subset(class_to_id, train)), dim=dim)
    vali_dataset = LabeledPairDataset(*pair_sampler(ub.dict_subset(class_to_id, vali)), dim=dim)
    test_dataset = LabeledPairDataset(*pair_sampler(ub.dict_subset(class_to_id, test)), dim=dim)
    print('train_dataset = {!r}'.format(len(train_dataset)))
    print('vali_dataset = {!r}'.format(len(vali_dataset)))
    print('test_dataset = {!r}'.format(len(test_dataset)))
    return train_dataset, vali_dataset, test_dataset


def comparable_vamp():
    """
    Script to get comparison between vamp and siam

    CommandLine:
        python -m clab.live.siam_train comparable_vamp --db GZ_Master1
        python -m clab.live.siam_train comparable_vamp --db PZ_Master1

    Example:
        >>> # SCRIPT
        >>> from clab.live.siam_train import *  # NOQA
        >>> comparable_vamp()
    """
    import parse
    import glob
    from ibeis.algo.verif import vsone
    parse.log.setLevel(30)

    # dbname = ub.argval('--db', default='PZ_MTEST')
    dbname = ub.argval('--db', default='GZ_Master1')
    dim = 512
    datasets = randomized_ibeis_dset(dbname, dim=dim)
    workdir = ub.ensuredir(os.path.expanduser(
        '~/data/work/siam-ibeis2/' + dbname))

    class_names = ['diff', 'same']
    task_name = 'binary_match'

    datasets['test'].pccs
    datasets['train'].pccs

    # pblm = vsone.OneVsOneProblem.from_empty('PZ_MTEST')
    ibs = datasets['train'].infr.ibs
    labeled_aid_pairs = [datasets['train'].get_aidpair(i)
                         for i in range(len(datasets['train']))]
    pblm_train = vsone.OneVsOneProblem.from_labeled_aidpairs(
        ibs, labeled_aid_pairs, class_names=class_names,
        task_name=task_name,
    )

    test_labeled_aid_pairs = [datasets['test'].get_aidpair(i)
                              for i in range(len(datasets['test']))]
    pblm_test = vsone.OneVsOneProblem.from_labeled_aidpairs(
        ibs, test_labeled_aid_pairs, class_names=class_names,
        task_name=task_name,
    )

    # ----------------------------
    # Build a VAMP classifier using the siamese training dataset
    pblm_train.load_features()
    pblm_train.samples.print_info()
    pblm_train.build_feature_subsets()
    pblm_train.samples.print_featinfo()

    pblm_train.learn_deploy_classifiers(task_keys=['binary_match'])
    clf_dpath = ub.ensuredir((workdir, 'clf'))
    classifiers = pblm_train.ensure_deploy_classifiers(dpath=clf_dpath)
    ibs_clf = classifiers['binary_match']
    clf = ibs_clf.clf

    # ----------------------------
    # Evaluate the VAMP classifier on the siamese testing dataset
    pblm_test.load_features()
    pblm_test.samples.print_info()
    pblm_test.build_feature_subsets()
    pblm_test.samples.print_featinfo()
    data_key = pblm_train.default_data_key
    task_key = 'binary_match'
    vamp_res = pblm_test._external_classifier_result(clf, task_key, data_key)
    vamp_report = vamp_res.extended_clf_report()  # NOQA
    print('vamp roc = {}'.format(vamp_res.roc_score()))

    # ----------------------------
    # Evaluate the siamese dataset
    pretrained = 'resnet50'
    branch = getattr(torchvision.models, pretrained)(pretrained=False)
    model = models.SiameseLP(p=2, branch=branch, input_shape=(1, 3, dim, dim))
    xpu = xpu_device.XPU.from_argv()
    print('Preparing to predict {} on {}'.format(model.__class__.__name__,
                                                 xpu))

    xpu.move(model)

    train_dpath = ub.truepath(
        '~/remote/aretha/data/work/siam-ibeis2/GZ_Master1/arch/SiameseLP/train/'
        'input_11934-ldxbwpwz/solver_11934-ldxbwpwz_SiameseLP_resnet50_twuldaap_a=1,n=2/')

    train_dpaths = glob.glob(ub.truepath(
        join(workdir, 'arch/SiameseLP/train/input_*/'
             'solver_*_SiameseLP_resnet50_*_a=1,n=2/')))
    assert len(train_dpaths) == 1
    train_dpath = train_dpaths[0]

    epoch = ub.argval('--epoch', default=None)
    epoch = int(epoch) if epoch is not None else None
    load_path = get_snapshot(train_dpath, epoch=epoch)

    print('Loading snapshot onto {}'.format(xpu))
    snapshot = torch.load(load_path, map_location=xpu.map_location())
    model.load_state_dict(snapshot['model_state_dict'])
    del snapshot

    model.train(False)

    dists = []
    dataset = datasets['test']
    for aid1, aid2 in ub.ProgIter(pblm_test.samples.index, label='predicting'):
        inputs = dataset.from_edge(aid1, aid2)
        # img1, img2 = [torch.autograd.Variable(item.cpu()) for item in inputs]
        img1, img2 = xpu.variables(*inputs)
        dist_tensor = model(img1[None, :], img2[None, :])
        dist = dist_tensor.data.cpu().numpy()
        dists.append(dist)

    dist_arr = np.squeeze(np.array(dists))

    p_same = np.exp(-dist_arr)
    p_diff = 1 - p_same

    import pandas as pd
    # hack probabilities
    probs_df = pd.DataFrame(
        np.array([p_diff, p_same]).T,
        columns=class_names,
        index=pblm_test.samples['binary_match'].indicator_df.index
        # index=pblm_test.samples.index
    )
    siam_res = vsone.clf_helpers.ClfResult()
    siam_res.probs_df = probs_df
    siam_res.probhats_df = None
    siam_res.data_key = 'SiamL2'
    siam_res.feat_dims = None
    siam_res.class_names = class_names
    siam_res.task_name = task_name
    siam_res.target_bin_df = pblm_test.samples['binary_match'].indicator_df
    siam_res.target_enc_df = pblm_test.samples['binary_match'].encoded_df

    print('--- SIAM ---')
    print('epoch = {!r}'.format(epoch))
    siam_report = siam_res.extended_clf_report()  # NOQA
    print('siam roc = {}'.format(siam_res.roc_score()))

    print('--- VAMP ---')
    vamp_report = vamp_res.extended_clf_report()  # NOQA
    print('vamp roc = {}'.format(vamp_res.roc_score()))


def siam_vsone_train():
    r"""
    CommandLine:
        python -m clab.live.siam_train siam_vsone_train --dry
        python -m clab.live.siam_train siam_vsone_train
        python -m clab.live.siam_train siam_vsone_train --db PZ_Master1
        python -m clab.live.siam_train siam_vsone_train --db PZ_MTEST --dry
        python -m clab.live.siam_train siam_vsone_train --db PZ_MTEST
        python -m clab.live.siam_train siam_vsone_train --db RotanTurtles

        python -m clab.live.siam_train siam_vsone_train --db humpbacks_fb

    Example:
        >>> # DISABLE_DOCTEST
        >>> from clab.live.siam_train import *  # NOQA
        >>> siam_vsone_train()
    """
    dbname = ub.argval('--db', default='PZ_MTEST')
    # train_dataset, vali_dataset, test_dataset = ibeis_datasets('GZ_Master')
    dim = 512
    datasets = randomized_ibeis_dset(dbname, dim=dim)
    workdir = ub.ensuredir(os.path.expanduser(
        '~/data/work/siam-ibeis2/' + dbname))
    print('workdir = {!r}'.format(workdir))

    # train_dataset, vali_dataset, test_dataset = att_faces_datasets()
    # workdir = os.path.expanduser('~/data/work/siam-att')
    for k, v in datasets.items():
        print('* len({}) = {}'.format(k, len(v)))

    batch_size = 6

    print('batch_size = {!r}'.format(batch_size))
    # class_weights = train_dataset.class_weights()
    # print('class_weights = {!r}'.format(class_weights))

    pretrained = 'resnet50'
    # pretrained = 'resnet50'
    branch = getattr(torchvision.models, pretrained)(pretrained=True)
    model = models.SiameseLP(p=2, branch=branch, input_shape=(1, 3, dim, dim))
    print(model)

    hyper = hyperparams.HyperParams(
        criterion=(criterions.ContrastiveLoss, {
            'margin': 4,
            'weight': None,
        }),
        optimizer=(torch.optim.SGD, {
            'weight_decay': .0005,
            'momentum': 0.9,
            'nesterov': True,
        }),
        scheduler=('Exponential', {
            'gamma': 0.99,
            'base_lr': 0.001,
            'stepsize': 2,
        }),
        other={
            'n_classes': 2,
            'augment': datasets['train'].augment,
        }
    )

    def custom_metrics(harn, output, label):
        from clab import metrics
        metrics_dict = metrics._siamese_metrics(output, label,
                                                 margin=harn.criterion.margin)
        return metrics_dict

    dry = ub.argflag('--dry')
    from clab.live.sseg_train import directory_structure
    train_dpath, test_dpath = directory_structure(
        workdir, model.__class__.__name__, datasets, pretrained=pretrained,
        train_hyper_id=hyper.hyper_id(), suffix='_' + hyper.other_id())

    xpu = xpu_device.XPU.from_argv()
    harn = fit_harness.FitHarness(
        model=model, hyper=hyper, datasets=datasets, xpu=xpu,
        batch_size=batch_size,
        train_dpath=train_dpath, dry=dry,
    )
    harn.add_metric_hook(custom_metrics)
    harn.run()


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m clab.live.siam_train siam_vsone_train --dry
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
