def region_normalize():
    # Normalize by the size of the original region
    # n_ccs, ccs = cv2.connectedComponents(mask)
    # prob_fg = dist_fg.copy()
    # for i in range(1, n_ccs):
    #     cc_mask = ccs == i
    #     cc_fgvals = prob_fg[cc_mask]
    #     prob_fg[cc_mask] = cc_fgvals / cc_fgvals.max()
