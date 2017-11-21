import cv2
import numpy as np


# def mask(mask, k=3, n_iter=2):
#     kernel = np.ones((k, k), np.uint8)
#     mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=n_iter)
#     return mask


def watershed_filter(mask, dist_thresh=5, topology=None, demo_mode=False):
    """
    References:
        https://docs.opencv.org/trunk/d3/db4/tutorial_py_watershed.html

    Example:
        >>> from clab.torch import filters
        >>> from clab import util
        >>> import ubelt as ub
        >>> pred = util.imread(ub.grabdata('https://i.imgur.com/Xdm4A24.png'))
        >>> img  = util.imread(ub.grabdata('https://i.imgur.com/M0CZ8ba.png'))

        >>> pred = util.imread(ub.grabdata('https://i.imgur.com/aZRgXpN.png'))
        >>> img = util.imread(ub.grabdata('https://i.imgur.com/7cwM5b6.jpg'))

        import plottool as pt
        pt.qtensure()

        from .tasks.urban_mapper_3d import UrbanMapper3D
        task = UrbanMapper3D('', '')

        kernel = np.ones((7, 7), np.uint8)
        mask = cv2.morphologyEx(pred, cv2.MORPH_OPEN, kernel, iterations=1)

        ut.exec_funckw(filters.watershed_filter, globals())
        demo_mode = True
        dist_thresh = 5
        ut.exec_func_src3(filters.watershed_filter, globals())

        blend_pred = task.instance_colorize(cv2.connectedComponents(pred, connectivity=4)[1], img, seed=2)
        blend_mask = task.instance_colorize(cv2.connectedComponents(mask, connectivity=4)[1], img, seed=2)
        blend_filtered = task.instance_colorize(cv2.connectedComponents(filtered, connectivity=4)[1], img, seed=2)

        sep_img = np.zeros(shape, dtype=np.uint8)
        sep_img[separator_locs] = 3
        n_seps = cv2.connectedComponents(sep_img, connectivity=8)[0]
        print('Made n_seps = {!r}'.format(n_seps))
        sep_img[locs] = 2
        sep_img[separator_locs] = 3

        pnum_ = pt.make_pnum_nextgen(nCols=4, nRows=2)
        # pt.imshow(img, fnum=1, pnum=pnum_(), title='image')
        pt.imshow(blend_pred, fnum=1, pnum=pnum_(), title='pred')
        pt.imshow(blend_mask, fnum=1, pnum=pnum_(), title='opening')
        pt.imshow(np.minimum(dist_fg, dist_thresh), norm=True, fnum=1, pnum=pnum_(), title='prob-fg')
        pt.imshow(sure_fg, fnum=1, pnum=pnum_(), title='seeds')
        pt.imshow(task.instance_colorize(markers, seed=2), fnum=1, pnum=pnum_(), title='watershed')
        pt.imshow(task.instance_colorize(mask_markers, seed=2), fnum=1, pnum=pnum_(), title='masked_watershed')
        pt.imshow(task.instance_colorize(sep_img, seed=2), fnum=1, pnum=pnum_(), title='separators')
        pt.imshow(blend_filtered, fnum=1, pnum=pnum_(), title='relabeled')

        # pt.imshow(task.instance_colorize(seed_ccs), fnum=3)
        # pt.imshow(task.instance_colorize(task.instance_label(pred, k=7, watershed=True), img, seed=2), fnum=2)
        # markers[mask == 0] = 0
        # pt.imshow(task.instance_colorize(markers))
        # cc_labels = cv2.connectedComponents(filtered, connectivity=4)[1]
        # pt.imshow(task.instance_colorize(cc_labels))

    """
    # Apply watershed to NN results
    # https://docs.opencv.org/trunk/d3/db4/tutorial_py_watershed.html
    # Finding sure foreground area

    dist_fg = cv2.distanceTransform(mask, cv2.DIST_L2, maskSize=3)

    sure_fg = (dist_fg > dist_thresh)
    # prob_fg = dist_fg / dist_fg.max()
    # sure_fg = (prob_fg > thresh)

    # Finding unknown region
    # unknown = mask - sure_fg

    # Marker labeling, mark the region of unknown with zero
    n_seeds, seed_ccs = cv2.connectedComponents(sure_fg.astype(np.uint8))
    # print('n_seeds = {!r}'.format(n_seeds))
    # Mark the unknown places with 0
    # seeds[unknown.astype(np.bool)] = 0

    # Note, anything that is not part of the foreground will be filled in by
    # the watershed. We don't want to use it for segmentation, we are only
    # using it to find the boundararies at which two seed objects meet.

    if topology is None:
        topology = np.dstack([mask, mask, mask])

    markers = seed_ccs.copy() if demo_mode else seed_ccs
    markers = cv2.watershed(topology, markers)
    # -1 indicates boundaries between objects
    boundaries = (markers == -1)

    # Remove all locations that were not part of the original mask
    mask_markers = markers.copy() if demo_mode else markers
    mask_markers[mask == 0] = 0

    # If our initial segmentation is pretty tight, removing the background will
    # hurt us. We find and remove only the boundaries that separate two
    # non-background regions.
    locs = np.where(boundaries)
    shape = mask_markers.shape

    high = [np.minimum(locs[i] + 1, shape[i] - 1) for i in range(2)]
    low  = [np.maximum(locs[i] - 1, 0) for i in range(2)]

    # high = [locs[i] + 1 for i in range(2)]
    # low  = [locs[i] - 1 for i in range(2)]
    # high_boundry = [high[i] >= shape[i] for i in range(2)]
    # low_boundry  = [low[i] <= 0 for i in range(2)]

    neighors8 = np.hstack([
        # important to use 8-cc here
        mask_markers[(high[0], locs[1])][:, None],
        mask_markers[( low[0], locs[1])][:, None],
        mask_markers[(locs[0], high[1])][:, None],
        mask_markers[(locs[0],  low[1])][:, None],
        mask_markers[(high[0], high[1])][:, None],
        mask_markers[( low[0],  low[1])][:, None],
        mask_markers[( low[0], high[1])][:, None],
        mask_markers[(high[0],  low[1])][:, None],
    ])
    non_seed_labels = {-1, 0}
    # Find boundaries that are adjacent to multiple different seed labels
    on_boundry = [len(s - non_seed_labels) > 1 for s in map(set, neighors8)]
    separator_locs = [x.compress(on_boundry) for x in locs]

    # Simply remove the multi-object boundaries from the mask
    filtered = mask.copy()
    filtered[separator_locs] = 0
    return filtered


def crf_posterior(img, log_probs, **kwargs):
    """
    Conditional Random Field posterior probabilities

    Args:
        img (np.ndarray): must be an RGB image
        log_probs (np.ndarray): [C x H x W] tensor
            must be negative log probabilities of each class

    References:
        https://github.com/lucasb-eyer/pydensecrf
        https://arxiv.org/pdf/1210.5644.pdf

    Ignore:
        import pickle
        from .torch.urban_mapper import *
        task = get_task('urban_mapper_3d')
        crf_data = pickle.load(open('crf_testdata.pkl', 'rb'))
        img, log_probs = ub.take(crf_data, ['img', 'log_probs'])

        pred_raw = log_probs.argmax(axis=0)
        blend_pred_raw = task.colorize(pred_raw, img)
        pt.imshow(blend_pred_raw, pnum=(1, 2, 1), fnum=1)

        posterior = crf_posterior(img, log_probs, n_iters=10, sigma_=4)

        pred_crf = posterior.argmax(axis=0)
        blend_pred_crf = task.colorize(pred_crf, img)
        pt.imshow(blend_pred_crf, pnum=(1, 2, 2), fnum=1)
    """
    import pydensecrf.densecrf as dcrf
    import pydensecrf.utils as dcrf_utils

    # TODO: Use L-BFGS to learn model parameters

    # Appearence (position + color) kernel parameters
    w1 = kwargs.get('w1', 4)  # weight of the kernel
    sigma_alpha = kwargs.get('sigma_alpha', 100)  # width of this kernel's position component
    sigma_beta  = kwargs.get('sigma_beta', 3)     # width of this kernel's color component

    # Smoothness kernel parameters
    w2 = kwargs.get('w2', 3)  # weight of the kernel
    sigma_gamma = kwargs.get('sigma_gamma', 3)  # width of this kernel's position component

    n_iters = kwargs.get('n_iters', 10)

    assert log_probs.max() < 0, 'must be negative log probs'
    # assert np.exp(log_probs).max() <= (1.0 + 1e-6)
    # assert np.exp(log_probs).min() >= (0.0 - 1e-6)

    [n_classes, height, width] = log_probs.shape
    assert (height, width) == img.shape[0:2]
    model = dcrf.DenseCRF(height * width, n_classes)

    # Use -log(prob) output from NN as unary energy
    unary_energy = dcrf_utils.unary_from_softmax(np.exp(log_probs))
    model.setUnaryEnergy(np.ascontiguousarray(unary_energy))

    # This adds the color-independent term, features are the locations only.
    feats = dcrf_utils.create_pairwise_gaussian(
        sdims=(sigma_gamma, sigma_gamma), shape=img.shape[:2]
    )
    model.addPairwiseEnergy(feats, compat=w2,
                            kernel=dcrf.DIAG_KERNEL,
                            normalization=dcrf.NORMALIZE_SYMMETRIC)

    # This adds the color-dependent term, i.e. features are (x,y,r,g,b).
    feats = dcrf_utils.create_pairwise_bilateral(sdims=(sigma_alpha, sigma_alpha),
                                                 schan=(sigma_beta, sigma_beta, sigma_beta),
                                                 img=img, chdim=2)
    model.addPairwiseEnergy(feats, compat=w1, kernel=dcrf.DIAG_KERNEL,
                            normalization=dcrf.NORMALIZE_SYMMETRIC)

    # Run the CRF inference
    infr_result = model.inference(n_iters)
    posterior = np.array(infr_result).reshape(log_probs.shape)
    return posterior
