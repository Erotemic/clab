import torch
from clab import xpu_device


def device_mapping(gpu_num: int):
    """
    Forces torch to load a saved snapshot onto a specific processor

    Args:
        gpu_num (int): gpu num to load to or None for CPU

    Notes:
        In order to `torch.load()` a GPU-trained model onto a CPU (or specific
        GPU), you have to supply a `map_location` function. Call this with the
        desired `gpu_num` to get the function that `torch.load()` needs.

    References:
        https://github.com/allenai/allennlp/blob/b12517c78db605eda815a0468fd92c014e37821b/allennlp/nn/util.py

    Example:
        >>> # doctest: +SKIP
        >>> # To load a tensor onto a specific gpu
        >>> import ubelt as ub
        >>> from clab.nnio import *
        >>> from clab.util import gpu_util
        >>> ngpus = gpu_util.num_gpus()
        >>> if ngpus <= 1:
        >>>     raise Exception('+SKIP')
        >>> dpath = ub.ensure_app_cache_dir('clab', 'test', 'nnio')
        >>> fpath = dpath + '/state.pt'
        >>> # Create data on gpu 0
        >>> data_cpu = torch.FloatTensor([3, 1, 4, 1, 5])
        >>> data_gpu0 = data_cpu.cuda(0)
        >>> torch.save(data_gpu0, fpath)
        >>> # Load data on gpu 1
        >>> data_gpu1 = torch.load(fpath, map_location=device_mapping(1))
        >>> assert data_cpu.storage().is_cuda is False
        >>> assert data_gpu0.storage().get_device() == 0
        >>> assert data_gpu1.storage().get_device() == 1
    """
    def map_location(storage: torch.Storage, location) -> torch.Storage:
        """
        Args:
            storage (torch.Storage) : the initial deserialization of the
                storage of the data read by `torch.load`, residing on the CPU.
            location (str): tag identifiying the location the data being read
                by `torch.load` was originally saved from.

        Returns:
            torch.Storage : the storage
        """
        xpu = xpu_device.XPU(gpu_num)
        if xpu.is_gpu():
            return storage.cuda(xpu.num)
        else:
            return storage
    return map_location


def export_model(fpath, model, input_shapes=None):
    """
    Exports a model in the ONNX format

    References:
        http://pytorch.org/tutorials/advanced/super_resolution_with_caffe2.html

    Ignore:
        >>> import ubelt as ub
        >>> from clab.nnio import *
        >>> from clab import models
        >>> model = models.MnistNet()
        >>> dpath = ub.ensure_app_cache_dir('clab', 'test', 'nnio')
        >>> fpath = dpath + '/export.onnx'
        >>> input_shapes = (1, 28, 28)
        >>> export_model(fpath, model, input_shapes)

    Ignore:
        >>> from clab import models
        >>> from torch.autograd import Variable
        >>> B, C, W, H = (4, 3, 372, 400)
        >>> n_classes = 11
        >>> input_shape = (B, C, W, H)
        >>> model = models.SegNet(in_channels=C, n_classes=n_classes)
        >>> input_shapes = [(C, W, H)]
        >>> params = None
        >>> import ubelt as ub
        >>> dpath = ub.ensure_app_cache_dir('clab', 'test', 'nnio')
        >>> fpath = dpath + '/export_net.onnx'
        >>> export_model(fpath, model, input_shapes)
        >>> onnx_model = onnx.load(fpath)

    """
    assert fpath.endswith('.onnx')

    # Make a double nested tuple so n inputs is an n-tuple
    if isinstance(input_shapes, (list, tuple)):
        first = input_shapes[0]
        if not isinstance(first, (list, tuple)):
            # 1-input was specified as a non-nested tuple make it a 1-tuple
            input_shapes = (input_shapes,)

    # Input to the model
    # model input (or a tuple for multiple inputs)
    batch_size = 1
    input_variables = tuple(
        torch.autograd.Variable(torch.randn(batch_size, *shape),
                                requires_grad=True)
        for shape in input_shapes
    )

    # fpath is where to save the model (can be a file or file-like object)

    # store the trained parameter weights inside the model file
    export_params = False

    # Export the model
    torch_out = torch.onnx._export(model, input_variables, fpath,
                                   export_params=export_params)
    torch_out


def import_model(fpath):
    assert fpath.endswith('.onnx')
    import onnx
    import onnx.helper
    import onnx.checker
    model = onnx.load(fpath)
    onnx.checker.check_model(model)
    print(onnx.helper.printable_graph(model.graph))


def _onnx_as_caffe2(fpath):
    import onnx_caffe2.backend as backend
    import numpy as np
    import onnx
    import onnx.helper
    import onnx.checker
    model = onnx.load(fpath)
    onnx.checker.check_model(model)
    rep = backend.prepare(model, device='CUDA:0')  # or "CPU"
    # For the Caffe2 backend:
    #     rep.predict_net is the Caffe2 protobuf for the network
    #     rep.workspace is the Caffe2 workspace for the network
    #       (see the class onnx_caffe2.backend.Workspace)
    outputs = rep.run(np.random.randn(10, 3, 224, 224).astype(np.float32))
    # To run networks with more than one input, pass a tuple
    # rather than a single numpy ndarray.
    print(outputs[0])
