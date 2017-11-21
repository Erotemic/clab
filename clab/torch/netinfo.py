import torchvision
import re


def get_num_gen(gen):
    return sum(1 for x in gen)


def flops_layer(layer):
    """
    Calculate the number of flops for given a string information of layer.
    We extract only resonable numbers and use them.

    References:
        https://discuss.pytorch.org/t/calculating-flops-of-a-given-pytorch-model/3711

    Args:
        layer (str) : example
            Linear (512 -> 1000)
            Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
    """
    # print(layer)
    idx_type_end = layer.find('(')
    type_name = layer[:idx_type_end]

    params = re.findall('[^a-z](\d+)', layer)
    flops = 1

    if layer.find('Linear') >= 0:
        C1 = int(params[0])
        C2 = int(params[1])
        flops = C1 * C2

    elif layer.find('Conv2d') >= 0:
        C1 = int(params[0])
        C2 = int(params[1])
        K1 = int(params[2])
        K2 = int(params[3])

        # image size
        H = 32
        W = 32
        flops = C1 * C2 * K1 * K2 * H * W

    return flops


def calculate_flops(gen):
    """
    Calculate the flops given a generator of pytorch model.
    It only compute the flops of forward pass.

    Example:
        >>> net = torchvision.models.resnet18()
        >>> calculate_flops(net.children())
    """
    flops = 0

    for child in gen:
        num_children = get_num_gen(child.children())

        # leaf node
        if num_children == 0:
            flops += flops_layer(str(child))

        else:
            flops += calculate_flops(child.children())

    return flops


def demo():
    net = torchvision.models.resnet18()
    flops = calculate_flops(net.children())
    print(flops / 10**9, 'G')
    # 11.435429919 G


def as_networkx(model, input_shapes, params=None):
    """
    Produces Graphviz representation of PyTorch autograd graph
    Blue nodes are the Variables that require grad, orange are Tensors
    saved for backward in torch.autograd.Function

    Args:
        var: output Variable
        params: dict of (name, Variable) to add names to node that
            require grad (TODO: make optional)

    Example:
        >>> from clab.torch.netinfo import *
        >>> from clab.torch.models.unet import *  # NOQA
        >>> from torch.autograd import Variable
        >>> B, C, W, H = (4, 3, 372, 400)
        >>> n_classes = 11
        >>> input_shape = (B, C, W, H)
        >>> model = UNet(in_channels=C, n_classes=n_classes)
        >>> input_shapes = [(B, C, W, H)]
        >>> params = None
        >>> graph = as_networkx(model, input_shapes)

        >>> import plottool as pt
        >>> pt.nx_helpers.dump_nx_ondisk(graph, 'torch_model.png')
    """
    import torch
    from torch.autograd import Variable
    # if params is not None:
    #     assert isinstance(params.values()[0], Variable)
    #     param_map = {id(v): k for k, v in params.items()}

    node_attr = dict(style='filled', shape='box', align='left', fontsize='12',
                     ranksep='0.1', height='0.2')

    inputs = [Variable(torch.rand(*shape), requires_grad=True) for shape in input_shapes]
    output = model(*inputs)
    # dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))

    import networkx as nx
    graph = nx.DiGraph()
    # graph.graph['size'] = '12,12'
    seen = set()

    def size_to_str(size):
        return '(' + (', ').join(['%d' % v for v in size]) + ')'

    def add_nodes(var):
        if var not in seen:
            if torch.is_tensor(var):
                graph.add_node(str(id(var)),
                               label=size_to_str(var.size()),
                               fillcolor='orange')
            elif hasattr(var, 'variable'):
                u = var.variable
                # name = param_map[id(u)] if params is not None else ''
                name = ''
                node_name = '%s\n %s' % (name, size_to_str(u.size()))
                graph.add_node(str(id(var)), label=node_name,
                               fillcolor='lightblue')
            else:
                graph.add_node(str(id(var)), label=str(type(var).__name__))
            seen.add(var)
            if hasattr(var, 'next_functions'):
                for u in var.next_functions:
                    if u[0] is not None:
                        graph.add_edge(str(id(u[0])), str(id(var)))
                        add_nodes(u[0])
            if hasattr(var, 'saved_tensors'):
                for t in var.saved_tensors:
                    graph.add_edge(str(id(t)), str(id(var)))
                    add_nodes(t)

    add_nodes(output.grad_fn)

    for k, v in node_attr.items():
        nx.set_node_attributes(graph, v, name=k)
    return graph
