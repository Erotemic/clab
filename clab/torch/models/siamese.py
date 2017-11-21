import torch
import torchvision
from clab.torch.models.output_shape_for import OutputShapeFor


class SiameseLP(torch.nn.Module):
    """
    Siamese pairwise distance

    Example:
        >>> from clab.torch.models.siamese import *
        >>> self = SiameseLP()
    """

    def __init__(self, p=2, branch=None, input_shape=(1, 3, 224, 224)):
        super(SiameseLP, self).__init__()
        if branch is None:
            self.branch = torchvision.models.resnet50(pretrained=True)
        else:
            self.branch = branch
        assert isinstance(self.branch, torchvision.models.ResNet)
        prepool_shape = self.resnet_prepool_output_shape(input_shape)
        # replace the last layer of resnet with a linear embedding to learn the
        # LP distance between pairs of images.
        # Also need to replace the pooling layer in case the input has a
        # different size.
        pool_channels = prepool_shape[1]
        pool_kernel = prepool_shape[2:4]
        self.branch.avgpool = torch.nn.AvgPool2d(pool_kernel, stride=1)
        self.branch.fc = torch.nn.Linear(pool_channels, 500)

        self.pdist = torch.nn.PairwiseDistance(p=p)

    def resnet_prepool_output_shape(self, input_shape):
        """
        input_shape = (1, 3, 224, 224)
        input_shape = (1, 3, 416, 416)
        """
        # Figure out how big the output will be and redo the average pool layer
        # to account for it
        branch = self.branch
        shape = input_shape
        shape = OutputShapeFor(branch.conv1)(shape)
        shape = OutputShapeFor(branch.bn1)(shape)
        shape = OutputShapeFor(branch.relu)(shape)
        shape = OutputShapeFor(branch.maxpool)(shape)

        shape = OutputShapeFor(branch.layer1)(shape)
        shape = OutputShapeFor(branch.layer2)(shape)
        shape = OutputShapeFor(branch.layer3)(shape)
        shape = OutputShapeFor(branch.layer4)(shape)
        prepool_shape = shape
        return prepool_shape

    def forward(self, input1, input2):
        """
        Compute a resnet50 vector for each input and look at the LP-distance
        between the vectors.

        >>> input1 = torch.autograd.Variable(torch.rand(1, 3, 224, 224))
        >>> input2 = torch.autograd.Variable(torch.rand(1, 3, 224, 224))
        >>> self = SiameseLP()
        >>> self(input1, input2)

        >>> input1 = torch.autograd.Variable(torch.rand(1, 3, 416, 416))
        >>> input2 = torch.autograd.Variable(torch.rand(1, 3, 416, 416))
        >>> input_shape1 = input1.shape
        >>> self = SiameseLP()
        >>> self(input1, input2)
        """
        output1 = self.branch(input1)
        output2 = self.branch(input2)
        output = self.pdist(output1, output2)
        return output

    def output_shape_for(self, input_shape1, input_shape2):
        shape1 = OutputShapeFor(self.branch)(input_shape1)
        shape2 = OutputShapeFor(self.branch)(input_shape2)
        assert shape1 == shape2
        output_shape = (shape1[0], 1)
        return output_shape


# class SiameseCLF(torch.nn.Module):
#     """
#     Siamese pairwise classifier

#     Example:
#         >>> from clab.torch.models.siamese import *
#         >>> self = SiameseCLF()
#     """

#     def __init__(self, n_classes=2):
#         super(SiameseCLF, self).__init__()
#         self.branch = torchvision.models.resnet50(pretrained=True)
#         self.num_fcin = self.branch.fc.in_features
#         # Instead of learning a distance, learn to classify pairs of images

#         linembed = torch.nn.Linear(self.num_fcin, 500)
#         self.branch.fc = linembed

#         ndims  = [linembed.out_features * 2, 500, 100]
#         self.classifier = torch.nn.Sequential(
#             torch.nn.Linear(in_features=ndims[0], out_features=ndims[1]),
#             torch.nn.ReLU(inplace=True),
#             torch.nn.Linear(in_features=ndims[1], out_features=ndims[2]),
#             torch.nn.ReLU(inplace=True),
#             torch.nn.Linear(in_features=ndims[2], out_features=n_classes),
#         )

#     def forward(self, input1, input2):
#         """
#         Compute a resnet50 vector for each input and look at the LP-distance
#         between the vectors.
#         """
#         output1 = self.branch(input1)
#         output2 = self.branch(input2)
#         combined = torch.cat(output1, output2)
#         output = self.classifier(combined)
#         return output


class SiameseL2(torch.nn.Module):

    def __init__(self, input_shape=(1, 3, 512, 512)):
        super(SiameseL2, self).__init__()
        self.branch = torchvision.models.resnet50(pretrained=True)
        # Custom method to figure out how big output of the convolution will be
        prepool_shape = self.resnet_prepool_output_shape(input_shape)
        # replace the last layer of resnet with a linear embedding
        pool_channels = prepool_shape[1]
        pool_kernel = prepool_shape[2:4]
        self.branch.avgpool = torch.nn.AvgPool2d(pool_kernel, stride=1)
        self.branch.fc = torch.nn.Linear(pool_channels, 500)
        # We will learn the L2 distance between pairs of embedded descriptors
        self.pdist = torch.nn.PairwiseDistance(p=2)

    def forward(self, input1, input2):
        """
        Compute a resnet50 vector for each input and look at the LP-distance
        between the vectors.
        """
        output1 = self.branch(input1)
        output2 = self.branch(input2)
        output = self.pdist(output1, output2)
        return output


class SiameseCLF(torch.nn.Module):

    def __init__(self, input_shape=(1, 3, 512, 512), n_classes=3, n_global_feats=42):
        super(SiameseCLF, self).__init__()
        self.branch = torchvision.models.resnet50(pretrained=True)
        # Custom method to figure out how big output of the convolution will be
        prepool_shape = self.resnet_prepool_output_shape(input_shape)
        # replace the last layer of resnet with a linear embedding
        pool_channels = prepool_shape[1]
        pool_kernel = prepool_shape[2:4]
        self.branch.avgpool = torch.nn.AvgPool2d(pool_kernel, stride=1)
        self.branch.fc = torch.nn.Linear(pool_channels, 500)

        # Instead of L2 learn to classify use a classifier
        ndims  = [self.branch.fc * 2 + n_global_feats, 500, 100]
        self.classifier = torch.nn.Sequential(
            # First fully connected layer
            torch.nn.Linear(in_features=ndims[0], out_features=ndims[1]),
            torch.nn.BatchNorm1d(ndims[1]),
            torch.nn.ReLU(inplace=True),

            # Second fully connected layer
            torch.nn.Linear(in_features=ndims[1], out_features=ndims[2]),
            torch.nn.BatchNorm1d(ndims[1]),
            torch.nn.ReLU(inplace=True),

            # Final layer for classification
            torch.nn.Linear(in_features=ndims[2], out_features=n_classes),
        )

        # Optional: We can still use L2 distance for auxillary loss
        self.pdist = torch.nn.PairwiseDistance(p=2)

    def forward(self, input1, input2, global_feats):
        """
        Compute a resnet50 vector for each input and look at the LP-distance
        between the vectors.
        """
        output1 = self.branch(input1)
        output2 = self.branch(input2)
        # Compute intermediate distance between the L2 vectors
        distance = self.pdist(output1, output2)
        # Concatenate the visual descriptor vectors (with global features)
        # into a single pairwise feature and classify the result.
        combined = torch.cat(output1, output2, global_feats)
        classification = self.classifier(combined)
        return classification, distance
