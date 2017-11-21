# if class_weights is None:
#     # HACK SHRUTIS WEIGHTS
#     class_weights = [1.6794, 1.1364, 0.9105, 1.1089, 0.4635, 0.6225]
#     ignore_label = 0
#     #class_weighting: 0.3664
#     #class_weighting: 1.1676
#     #class_weighting: 1.2135
#     #class_weighting: 23.185
#     #class_weighting: 6.8739
#     # WHY WERE THSE USED IN SHRUTIS CODE?
#     # core = core.replace('@UPSAMPLE_1_W@', '80')
#     # core = core.replace('@UPSAMPLE_1_H@', '45')
#     # core = core.replace('@UPSAMPLE_2_W@', '160')
#     # core = core.replace('@UPSAMPLE_2_H@', '90')


# if class_weights is None:
#     # HACK SHRUTIS WEIGHTS
#     class_weights = [1.1150, 5.7484, 0.6143, 4.0324, 0.7403, 0.7708,
#                      0.3041, 1.0000, 0.6813, 23.185, 6.8739]
#     ignore_label = 0


# shruti's BASIC hyper params changes
# test_interval: 20000
# stepsize: 1000
# max_iter: 25000


# shruti's PROPER hyper params changes
# base_lr: 0.001
# gamma: 0.1
# lr_policy: "multistep"
# stepvalue: 5000
# stepvalue: 7500
# stepvalue: 8500
# stepvalue: 9500
# display: 20
# momentum: 0.9
# max_iter: 30000
# weight_decay: 0.0005
# snapshot: 1000
