"""
Need to compile yolo first.

Currently its hacked into the system.


pip install cffi
cd $HOME/code/clab/clab/models/yolo2
./make.sh
"""


# TODO: VOC Dataset
label_names = ('aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair',
               'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant',
               'sheep', 'sofa', 'train', 'tvmonitor')
num_classes = len(label_names)

anchors = np.asarray([(1.08, 1.19), (3.42, 4.41),
                      (6.63, 11.38), (9.42, 5.11), (16.62, 10.52)],
                     dtype=np.float)
num_anchors = len(anchors)

