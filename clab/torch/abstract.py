import torch


class DummySSegDataset(torch.utils.data.Dataset):
    """
    Example:
        >>> dset = DummySSegDataset()
        >>> loader = torch.utils.data.DataLoader(dset, batch_size=4)
        >>> inputs, target = next(iter(loader))
    """
    def __init__(self, input_shape=(36, 48, 3), n_classes=2):
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.size = 1000

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        rng = torch.get_rng_state()
        torch.manual_seed(index)

        inputs = torch.randn(*self.input_shape)
        target = torch.randn(*self.input_shape)
        torch.set_rng_state(rng)

        return inputs, target


class DummyPairwiseDataset(torch.utils.data.Dataset):
    """
    Example:

        >>> dset = DummyPairwiseDataset()
        >>> loader = torch.utils.data.DataLoader(dset, batch_size=4)
        >>> inputs, target = next(iter(loader))
        >>> input1, input2 = inputs
    """
    def __init__(self, input_shape=(36, 48, 3), n_classes=2):
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.size = 1000

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        # create random image that is consistent with the index id
        rng = torch.get_rng_state()
        torch.manual_seed(index)

        img1 = torch.randn(*self.input_shape)
        img2 = torch.randn(*self.input_shape)
        img1[:, :, 0] = 1
        img2[:, :, 0] = 2
        img1[:, :, 1] = index
        img2[:, :, 1] = index

        target = torch.Tensor(1).random_(0, self.n_classes)[0]
        torch.set_rng_state(rng)

        inputs = (img1, img2)
        return inputs, target
