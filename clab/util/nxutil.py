import numpy as np
import ubelt as ub
import re
import networkx as nx


def dump_nx_ondisk(graph, fpath):
    agraph = make_agraph(graph.copy())
    # agraph = nx.nx_agraph.to_agraph(graph)
    agraph.layout(prog='dot')
    agraph.draw(ub.truepath(fpath))


def nx_ensure_agraph_color(graph):
    """ changes colors to hex strings on graph attrs """
    from clab.util import mplutil
    def _fix_agraph_color(data):
        try:
            color = data.get('color', None)
            alpha = data.get('alpha', None)
            if color is None and alpha is not None:
                color = [0, 0, 0]
            if color is not None:
                data['color'] = mplutil.Color(color, alpha).ashex()
        except Exception as ex:
            raise

    for node, node_data in graph.nodes(data=True):
        data = node_data
        _fix_agraph_color(data)

    for u, v, edge_data in graph.edges(data=True):
        data = edge_data
        _fix_agraph_color(data)


def nx_delete_None_edge_attr(graph, edges=None):
    removed = 0
    if graph.is_multigraph():
        if edges is None:
            edges = list(graph.edges(keys=graph.is_multigraph()))
        for edge in edges:
            u, v, k = edge
            data = graph[u][v][k]
            for key in list(data.keys()):
                try:
                    if data[key] is None:
                        del data[key]
                        removed += 1
                except KeyError:
                    pass
    else:
        if edges is None:
            edges = list(graph.edges())
        for edge in graph.edges():
            u, v = edge
            data = graph[u][v]
            for key in list(data.keys()):
                try:
                    if data[key] is None:
                        del data[key]
                        removed += 1
                except KeyError:
                    pass
    return removed


def nx_delete_node_attr(graph, name, nodes=None):
    """
    Removes node attributes

    Doctest:
        >>> G = nx.karate_club_graph()
        >>> nx.set_node_attributes(G, name='foo', values='bar')
        >>> datas = nx.get_node_attributes(G, 'club')
        >>> assert len(nx.get_node_attributes(G, 'club')) == 34
        >>> assert len(nx.get_node_attributes(G, 'foo')) == 34
        >>> nx_delete_node_attr(G, ['club', 'foo'], nodes=[1, 2])
        >>> assert len(nx.get_node_attributes(G, 'club')) == 32
        >>> assert len(nx.get_node_attributes(G, 'foo')) == 32
        >>> nx_delete_node_attr(G, ['club'])
        >>> assert len(nx.get_node_attributes(G, 'club')) == 0
        >>> assert len(nx.get_node_attributes(G, 'foo')) == 32
    """
    if nodes is None:
        nodes = list(graph.nodes())
    removed = 0
    # names = [name] if not isinstance(name, list) else name
    node_dict = graph.nodes

    if isinstance(name, list):
        for node in nodes:
            for name_ in name:
                try:
                    del node_dict[node][name_]
                    removed += 1
                except KeyError:
                    pass
    else:
        for node in nodes:
            try:
                del node_dict[node][name]
                removed += 1
            except KeyError:
                pass
    return removed


def make_agraph(graph_, inplace=False):
    import pygraphviz
    patch_pygraphviz()

    if not inplace:
        graph_ = graph_.copy()

    # Convert to agraph format
    num_nodes = len(graph_)
    LARGE_GRAPH = 100
    is_large = num_nodes > LARGE_GRAPH

    if is_large:
        print('Making agraph for large graph %d nodes. '
              'May take time' % (num_nodes))

    nx_ensure_agraph_color(graph_)
    # Reduce size to be in inches not pixels
    # FIXME: make robust to param settings
    # Hack to make the w/h of the node take thae max instead of
    # dot which takes the minimum
    shaped_nodes = [n for n, d in graph_.nodes(data=True) if 'width' in d]
    node_dict = graph_.nodes
    node_attrs = ub.dict_take(node_dict, shaped_nodes)

    width_px = np.array([n['width'] for n in node_attrs])
    height_px = np.array([n['height'] for n in node_attrs])
    scale = np.array([n.get('scale', 1.0) for n in node_attrs])

    inputscale = 72.0
    width_in = width_px / inputscale * scale
    height_in = height_px / inputscale * scale
    width_in_dict = dict(zip(shaped_nodes, width_in))
    height_in_dict = dict(zip(shaped_nodes, height_in))

    nx.set_node_attributes(graph_, name='width', values=width_in_dict)
    nx.set_node_attributes(graph_, name='height', values=height_in_dict)
    nx_delete_node_attr(graph_, name='scale')

    # Check for any nodes with groupids
    node_to_groupid = nx.get_node_attributes(graph_, 'groupid')
    if node_to_groupid:
        groupid_to_nodes = ub.group_items(*zip(*node_to_groupid.items()))
    else:
        groupid_to_nodes = {}
    # Initialize agraph format
    nx_delete_None_edge_attr(graph_)
    agraph = nx.nx_agraph.to_agraph(graph_)
    # Add subgraphs labels
    # TODO: subgraph attrs
    group_attrs = graph_.graph.get('groupattrs', {})
    for groupid, nodes in groupid_to_nodes.items():
        # subgraph_attrs = {}
        subgraph_attrs = group_attrs.get(groupid, {}).copy()
        cluster_flag = True
        # FIXME: make this more natural to specify
        if 'cluster' in subgraph_attrs:
            cluster_flag = subgraph_attrs['cluster']
            del subgraph_attrs['cluster']
        name = groupid
        if cluster_flag:
            # graphviz treast subgraphs labeld with cluster differently
            name = 'cluster_' + groupid
        else:
            name = groupid
        agraph.add_subgraph(nodes, name, **subgraph_attrs)

    for node in graph_.nodes():
        anode = pygraphviz.Node(agraph, node)
        # TODO: Generally fix node positions
        ptstr_ = anode.attr['pos']
        if (ptstr_ is not None and len(ptstr_) > 0 and not ptstr_.endswith('!')):
            ptstr = ptstr_.strip('[]').strip(' ').strip('()')
            ptstr_list = [x.rstrip(',') for x in re.split(r'\s+', ptstr)]
            pt_list = list(map(float, ptstr_list))
            pt_arr = np.array(pt_list) / inputscale
            new_ptstr_list = list(map(str, pt_arr))
            new_ptstr_ = ','.join(new_ptstr_list)
            if anode.attr['pin'] is True:
                anode.attr['pin'] = 'true'
            if anode.attr['pin'] == 'true':
                new_ptstr = new_ptstr_ + '!'
            else:
                new_ptstr = new_ptstr_
            anode.attr['pos'] = new_ptstr

    if graph_.graph.get('ignore_labels', False):
        for node in graph_.nodes():
            anode = pygraphviz.Node(agraph, node)
            if 'label' in anode.attr:
                try:
                    del anode.attr['label']
                except KeyError:
                    pass
    return agraph


def patch_pygraphviz():
    """
    Hacks around a python3 problem in 1.3.1 of pygraphviz
    """
    import pygraphviz
    if pygraphviz.__version__ != '1.3.1':
        return
    if hasattr(pygraphviz.agraph.AGraph, '_run_prog_patch'):
        return
    def _run_prog(self, prog='nop', args=''):
        """Apply graphviz program to graph and return the result as a string.

        >>> A = AGraph()
        >>> s = A._run_prog() # doctest: +SKIP
        >>> s = A._run_prog(prog='acyclic') # doctest: +SKIP

        Use keyword args to add additional arguments to graphviz programs.
        """
        from pygraphviz.agraph import (shlex, subprocess, PipeReader, warnings)
        runprog = r'"%s"' % self._get_prog(prog)
        cmd = ' '.join([runprog, args])
        dotargs = shlex.split(cmd)
        p = subprocess.Popen(dotargs,
                             shell=False,
                             stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE,
                             close_fds=False)
        (child_stdin,
         child_stdout,
         child_stderr) = (p.stdin, p.stdout, p.stderr)
        # Use threading to avoid blocking
        data = []
        errors = []
        threads = [PipeReader(data, child_stdout),
                   PipeReader(errors, child_stderr)]
        for t in threads:
            t.start()

        self.write(child_stdin)
        child_stdin.close()

        for t in threads:
            t.join()

        if not data:
            raise IOError(b"".join(errors))

        if len(errors) > 0:
            warnings.warn(str(b"".join(errors)), RuntimeWarning)

        return b"".join(data)
    # Patch error in pygraphviz
    pygraphviz.agraph.AGraph._run_prog_patch = _run_prog
    pygraphviz.agraph.AGraph._run_prog_orig = pygraphviz.agraph.AGraph._run_prog
    pygraphviz.agraph.AGraph._run_prog = _run_prog


def nx_source_nodes(graph):
    # for node in nx.dag.topological_sort(graph):
    for node in graph.nodes():
        if graph.in_degree(node) == 0:
            yield node


def nx_sink_nodes(graph):
    # for node in nx.dag.topological_sort(graph):
    for node in graph.nodes():
        if graph.out_degree(node) == 0:
            yield node
