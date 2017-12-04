import numpy as np
from clab import util
import ubelt as ub


def instance_fscore(gti, uncertain, dsm, pred, info=False):
    """
    path = '/home/local/KHQ/jon.crall/data/work/urban_mapper/eval/input_4224-rwyxarza/solver_4214-yxalqwdk_unet_vgg_nttxoagf_a=1,n_ch=5,n_cl=3/_epoch_00000236/restiched/pred'

    path = ub.truepath(
        '~/remote/aretha/data/work/urban_mapper2/test/input_4224-exkudlzu/'
        'solver_4214-guwsobde_unet_mmavmuou_eqnoygqy_a=1,c=RGB,n_ch=5,n_cl=4/'
        '_epoch_00000154/restiched/pred')
    mode_paths = sorted(glob.glob(path + '/*.png'))

    def instance_label(pred, k=15, n_iters=1, dist_thresh=5, watershed=False):
        mask = pred

        # noise removal
        if k > 1 and n_iters > 0:
            kernel = np.ones((k, k), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel,
                                    iterations=n_iters)

        if watershed:
            from clab.torch import filters
            mask = filters.watershed_filter(mask, dist_thresh=dist_thresh)

        mask = mask.astype(np.uint8)
        n_ccs, cc_labels = cv2.connectedComponents(mask, connectivity=4)
        return cc_labels

    from clab.tasks.urban_mapper_3d import UrbanMapper3D
    task = UrbanMapper3D('', '')

    fscores = []
    for pred_fpath in ub.ProgIter(mode_paths):
        pass
        gtl_fname = basename(pred_fpath).replace('.png', '_GTL.tif')
        gti_fname = basename(pred_fpath).replace('.png', '_GTI.tif')
        dsm_fname = basename(pred_fpath).replace('.png', '_DSM.tif')
        bgr_fname = basename(pred_fpath).replace('.png', '_RGB.tif')
        gtl_fpath = join(ub.truepath('~/remote/aretha/data/UrbanMapper3D/training/'), gtl_fname)
        gti_fpath = join(ub.truepath('~/remote/aretha/data/UrbanMapper3D/training/'), gti_fname)
        dsm_fpath = join(ub.truepath('~/remote/aretha/data/UrbanMapper3D/training/'), dsm_fname)
        bgr_fpath = join(ub.truepath('~/remote/aretha/data/UrbanMapper3D/training/'), bgr_fname)

        pred_seg = util.imread(pred_fpath)

        pred = instance_label2(pred_seg, dist_thresh=d, k=k, watershed=True)
        gti = util.imread(gti_fpath)
        gtl = util.imread(gtl_fpath)
        dsm = util.imread(dsm_fpath)
        bgr = util.imread(bgr_fpath)

        uncertain = (gtl == 65)

        fscore = instance_fscore(gti, uncertain, dsm, pred)
        fscores.append(fscore)
    print('k = {!r}'.format(k))
    print('d = {!r}'.format(d))
    print(np.mean(fscores))


    from clab import profiler
    instance_fscore_ = dynamic_profile(instance_fscore)
    fscore = instance_fscore_(gti, uncertain, dsm, pred)
    instance_fscore_.profile.profile.print_stats()
    """
    def _bbox(arr):
        # r1, c1, r2, c2
        return np.hstack([arr.min(axis=0), arr.max(axis=0)])

    def cc_locs(ccs):
        rc_locs = np.where(ccs > 0)
        rc_ids = ccs[rc_locs]
        rc_arr = np.ascontiguousarray(np.vstack(rc_locs).T)
        unique_labels, groupxs = util.group_indices(rc_ids)
        grouped_arrs = util.apply_grouping(rc_arr, groupxs, axis=0)
        id_to_rc = ub.odict(zip(unique_labels, grouped_arrs))
        return id_to_rc, unique_labels, groupxs, rc_arr

    (true_rcs_arr, group_true_labels,
     true_groupxs, true_rc_arr) = cc_locs(gti)

    (pred_rcs_arr, group_pred_labels,
     pred_groupxs, pred_rc_arr) = cc_locs(pred)

    DSM_NAN = -32767
    MIN_SIZE = 100
    MIN_IOU = 0.45
    # H, W = pred.shape[0:2]

    # --- Find uncertain truth ---
    # any gt-building explicitly labeled in the GTL is uncertain
    uncertain_labels = set(np.unique(gti[uncertain.astype(np.bool)]))
    # Any gt-building less than 100px or at the boundary is uncertain.
    for label, rc_arr in true_rcs_arr.items():
        if len(rc_arr) < MIN_SIZE:
            rc_arr = np.array(list(rc_arr))
            if (np.any(rc_arr == 0) or np.any(rc_arr == 2047)):
                uncertain_labels.add(label)
            else:
                rc_loc = tuple(rc_arr.T)
                is_invisible = (dsm[rc_loc] == DSM_NAN)
                if np.any(is_invisible):
                    invisible_rc = rc_arr.compress(is_invisible, axis=0)
                    invisible_rc_set = set(map(tuple, invisible_rc))
                    # Remove invisible pixels
                    remain_rc_set = list(set(map(tuple, rc_arr)).difference(invisible_rc_set))
                    true_rcs_arr[label] = np.array(remain_rc_set)
                    uncertain_labels.add(label)

    def make_int_coords(rc_arr, unique_labels, groupxs):
        # using nums instead of tuples gives the intersection a modest speedup
        rc_int = rc_arr.T[0] + pred.shape[0] + rc_arr.T[1]
        id_to_rc_int = ub.odict(zip(unique_labels,
                                    map(set, util.apply_grouping(rc_int, groupxs))))
        return id_to_rc_int

    # Make intersection a bit faster by filtering via bbox fist
    true_rcs_bbox = ub.map_vals(_bbox, true_rcs_arr)
    pred_rcs_bbox = ub.map_vals(_bbox, pred_rcs_arr)

    true_bboxes = np.array(list(true_rcs_bbox.values()))
    pred_bboxes = np.array(list(pred_rcs_bbox.values()))

    candidate_matches = {}
    for plabel, pb in zip(group_pred_labels, pred_bboxes):
        irc1 = np.maximum(pb[0:2], true_bboxes[:, 0:2])
        irc2 = np.minimum(pb[2:4], true_bboxes[:, 2:4])
        irc1 = np.minimum(irc1, irc2, out=irc1)
        isect_area = np.prod(np.abs(irc2 - irc1), axis=1)
        tlabels = list(ub.take(group_true_labels, np.where(isect_area)[0]))
        candidate_matches[plabel] = set(tlabels)

    # using nums instead of tuples gives the intersection a modest speedup
    pred_rcs_ = make_int_coords(pred_rc_arr, group_pred_labels, pred_groupxs)
    true_rcs_ = make_int_coords(true_rc_arr, group_true_labels, true_groupxs)

    # Greedy matching
    unused_true_rcs = true_rcs_.copy()
    FP = TP = FN = 0
    unused_true_keys = set(unused_true_rcs.keys())

    assignment = []
    fp_labels = []
    fn_labels = []
    tp_labels = []

    for pred_label, pred_rc_set in pred_rcs_.items():

        best_score = (-np.inf, -np.inf)
        best_label = None

        # Only check unused true labels that intersect with the predicted bbox
        true_cand = candidate_matches[pred_label] & unused_true_keys
        for true_label in true_cand:
            true_rc_set = unused_true_rcs[true_label]
            n_isect = len(pred_rc_set.intersection(true_rc_set))
            iou = n_isect / (len(true_rc_set) + len(pred_rc_set) - n_isect)
            if iou > MIN_IOU:
                score = (iou, -true_label)
                if score > best_score:
                    best_score = score
                    best_label = true_label

        if best_label is not None:
            assignment.append((pred_label, best_label, best_score[0]))
            unused_true_keys.remove(best_label)
            if true_label not in uncertain_labels:
                TP += 1
                tp_labels.append((pred_label, best_label, best_score[0]))
        else:
            FP += 1
            fp_labels.append(pred_label)

    # Had two bugs:
    # * used wrong variable to count false negs (all true were labeled as FN)
    #   (massivly increasing FN)
    # * Certain true building as marked as uncertain, but I was checking
    #   against the pred labels instead (possibly decreasing/increasing TP)

    fn_labels = unused_true_keys - uncertain_labels  # NOQA
    FN = len(fn_labels)

    precision = TP / (TP + FP) if TP > 0 else 0
    recall = TP / (TP + FN) if TP > 0 else 0
    if precision > 0 and recall > 0:
        f_score = 2 * precision * recall / (precision + recall)
    else:
        f_score = 0

    # They multiply by 1e6, but lets not do that.
    if info:
        infod = {
            'assign': assignment,
            'tp': tp_labels,
            'fp': fp_labels,
            'fn': fn_labels,
            'uncertain': uncertain_labels,
        }
        return (f_score, precision, recall), infod

    return (f_score, precision, recall)
