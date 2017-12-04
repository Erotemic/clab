# def pointset_boundary(rcs, shape):
#     offsets = [[ 0, +1], [+1, +1], [+1,  0], [ 0, -1],
#                [-1, -1], [-1,  0], [+1, -1], [-1, +1]]

#     rcs_ = np.ascontiguousarray(rcs[:, :, None])
#     offsets_ = np.ascontiguousarray(np.array(offsets).T[None, :])
#     rc_off = rcs_ + offsets_

#     # encode offset as an integer for intersection
#     rc_off_int = rc_off[:, 0, :] * shape[0] + rc_off[:, 1, :]
#     rc_int = rcs.T[0] * shape[0] + rcs.T[1]

#     # Faster than both intersect1d and the python set version
#     # Also faster than versions where we loop over the offset
#     flags = np.in1d(rc_off_int.ravel(), rc_int)
#     flags.shape = rc_off_int.shape
#     on_bound = (~np.all(flags, axis=1))

#     # might not be in a good order though
#     bound_rcs = rcs.compress(on_bound, axis=0)
#     return bound_rcs

# import ubelt
# for timer in ubelt.Timerit(1):
#     with timer:
#         grouped_contours = {}
#         for label, rcs in grouped_cc_rcs.items():
#             shape = gti.shape
#             bound_rcs = pointset_boundary(rcs, shape)
#             grouped_contours[label] = bound_rcs
