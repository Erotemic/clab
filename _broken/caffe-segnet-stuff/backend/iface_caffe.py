# -*- coding: utf-8 -*-
"""
Module for dealing with weirness of caffe and especially segnet-caffe
"""
from __future__ import absolute_import, division, print_function, unicode_literals
from os.path import exists, join, basename
import glob
import re
import ubelt as ub

from pysseg import getLogger
logger = getLogger(__name__)
print = logger.info


def snapshot_iterno(path):
    """
    The iterno is considered to be the first integer not followed by an
    underscore in the basename.
    """
    # Hack, not the most robust way of parsing the iter num
    last_part = basename(path).split('_')[-1]
    match = re.search(r'(?P<n>[0-9]+)', last_part)
    gd = match.groupdict()
    return int(gd['n'])


def load_snapshot_weight_paths(snapshot_prefix):
    import os
    if os.path.isdir(snapshot_prefix):
        pattern = join(snapshot_prefix, '*.caffemodel')
    else:
        pattern = snapshot_prefix + '*.caffemodel'
    snapshots = sorted(glob.glob(pattern))
    snapshots = sorted(snapshots, key=snapshot_iterno)
    return snapshots


def load_snapshot_state_paths(snapshot_prefix):
    # Hack, not the most robust sorting scheme
    import os
    if os.path.isdir(snapshot_prefix):
        pattern = join(snapshot_prefix, '*.solverstate')
    else:
        pattern = snapshot_prefix + '*.solverstate'
    snapshots = sorted(glob.glob(pattern))
    snapshots = sorted(snapshots, key=snapshot_iterno)
    return snapshots


def parse_solver_info(solver_fpath):
    from google.protobuf import text_format
    from pysseg.backend.find_segnet_caffe import import_segnet_caffe
    segnet_caffe = import_segnet_caffe()
    # Need to user different "message" depending on the file being read
    caffe_protos = {
        # 'bn' : segnet_caffe.proto.caffe_pb2.BNParameter
        'solver': segnet_caffe.proto.caffe_pb2.SolverParameter,
        # 'model': segnet_caffe.proto.caffe_pb2.NetParameter,
    }
    message = caffe_protos['solver']()
    with open(solver_fpath, 'r') as file:
        text = str(file.read())
    message = text_format.Merge(text, message)
    info = _parse_protobuf(message)
    info['train_model_path'] = info['net']
    return info
    # info = {
    #     'train_model_path': message.net,
    #     'snapshot_prefix': message.snapshot_prefix,
    #     'max_iter': message.max_iter,
    # }
    # return info


def _parse_protobuf(message):
    # HACK TO FIX protobuf_to_dict
    import six
    if six.PY3:
        from six.moves import builtins
        builtins.long = int
        builtins.unicode = str
    from protobuf_to_dict import protobuf_to_dict
    info = protobuf_to_dict(message)
    return info


def parse_solver_state_info(solverstate_fpath):
    """
    solverstate_fpath='/home/local/KHQ/jon.crall/.cache/segnet/solvers/snapshots/segnet_proper_solver_hdfpqisoxhttsqmoqjpuizwugfcjsbya_iter_40000.solverstate'
    """
    from google.protobuf import text_format
    from pysseg.backend.find_segnet_caffe import import_segnet_caffe
    segnet_caffe = import_segnet_caffe()
    message = segnet_caffe.proto.caffe_pb2.SolverState()
    with open(solverstate_fpath, 'rb') as file:
        solver_data = str(file.read())
    message = text_format.Merge(solver_data, message)
    info = _parse_protobuf(message)
    info['train_model_path'] = info['net']
    return info
    # train_model_path = message.net
    # snapshot_prefix = message.snapshot_prefix
    # info = {
    #     'train_model_path': train_model_path,
    #     'snapshot_prefix': snapshot_prefix,
    # }
    # return info


def parse_model_info(model_fpath):
    from pysseg.backend.find_segnet_caffe import import_segnet_caffe
    from google.protobuf import text_format
    segnet_caffe = import_segnet_caffe()
    # Need to user different "message" depending on the file being read
    caffe_protos = {
        # 'bn' : segnet_caffe.proto.caffe_pb2.BNParameter
        # 'solver': segnet_caffe.proto.caffe_pb2.SolverParameter,
        'model': segnet_caffe.proto.caffe_pb2.NetParameter,
    }
    message = caffe_protos['model']()
    with open(model_fpath, 'r') as file:
        text = str(file.read())
    message = text_format.Merge(text, message)
    info = _parse_protobuf(message)
    image_input_fpath = message.layer[0].dense_image_data_param.source
    info['image_input_fpath'] = image_input_fpath
    info['batch_size'] = message.layer[0].dense_image_data_param.batch_size
    # info = {
    #     'image_input_fpath': image_input_fpath,
    # }
    # return info
    # message = text_format.Merge(text, message)
    # info['train_model_path'] = info['net']
    return info


def _model_data_flow_to_networkx(model_info):
    layers = model_info['layer']
    import networkx as nx
    G = nx.DiGraph()

    prev = None
    # Stores last node with the data for this layer in it
    prev_map = {}

    SHOW_LOOPS = False

    for layer in layers:
        name = layer.get('name')
        print('name = {!r}'.format(name))
        G.add_node(name)
        bottom = set(layer.get('bottom', []))
        top = set(layer.get('top', []))

        both = top.intersection(bottom)
        if both:
            if prev is None:
                prev = both
            for b in both:
                prev_map[b] = name
            for b in prev:
                print('  * b = {!r}'.format(b))
                G.add_edge(b, name, constraint=False)
            for b in both:
                print('  * b = {!r}'.format(b))
                kw = {}
                if not G.has_edge(b, name):
                    kw['color'] = 'red'
                G.add_edge(b, name, constraint=True, **kw)
            prev = [name]
        else:
            prev = None

        # for b in (bottom - both):
        for b in bottom:
            print('  * b = {!r}'.format(b))
            constraint = True
            G.add_edge(prev_map.get(b, b), name, constraint=constraint)
            if SHOW_LOOPS:
                G.add_edge(b, name)
        # for t in (bottom - top):
        for t in top:
            print('  * t = {!r}'.format(t))
            constraint = True
            G.add_edge(name, prev_map.get(t, t), constraint=constraint)
            if SHOW_LOOPS:
                G.add_edge(name, t)

    G.remove_edges_from(list(G.selfloop_edges()))

    import plottool as pt
    pt.qtensure()
    pt.show_nx(G, arrow_width=1)
    pt.adjust_subplots(left=0, right=1, top=1, bottom=0)
    pt.pan_factory()
    pt.zoom_factory()

    list(nx.topological_sort(G))


def convert_weights(old_proto, new_proto, old_weight_paths, new_weight_path):
    """
    Converts weights between networks with the same parameter shapes but
    different layer names.
    """
    from pysseg.backend.find_segnet_caffe import import_segnet_caffe
    caffe = import_segnet_caffe()

    print('convert weights')
    print(exists(new_proto))
    print(exists(old_proto))
    print(exists(old_weight_paths))

    new_net = caffe.Net(new_proto, old_weight_paths, caffe.TEST)
    old_net = caffe.Net(old_proto, old_weight_paths, caffe.TEST)

    import numpy as np
    old_layers = list(old_net.params.items())
    new_layers = list(new_net.params.items())
    for layerno in range(len(old_layers)):
        k1, old_layer = old_layers[layerno]
        k2, new_layer = new_layers[layerno]
        if k1 == k2:
            assert len(old_layer) == len(new_layer), 'model architectures do not agree'
            for i in range(len(old_layer)):
                assert np.all(old_layer[i].data == new_layer[i].data), 'model architectures do not agree'
        else:
            assert len(old_layer) == len(new_layer)
            for i in range(len(old_layer)):
                old_data = old_layer[i].data
                new_data = new_layer[i].data
                assert new_data.shape == old_data.shape, 'model architectures do not agree'
                new_data.flat = old_data.flat

    old_layers = list(old_net.params.items())
    new_layers = list(new_net.params.items())
    for layerno in range(len(old_layers)):
        k1, old_layer = old_layers[layerno]
        k2, new_layer = new_layers[layerno]
        assert len(old_layer) == len(new_layer)
        for i in range(len(old_layer)):
            assert np.all(old_layer[i].data == new_layer[i].data), 'model architectures do not agree'

    new_net.save(new_weight_path)


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m pysseg.iface_caffe
    """
    import ubelt as ub  # NOQA
    ub.doctest_package()
