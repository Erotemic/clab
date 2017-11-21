    def evaluate(harn):
        """
        Uses python caffe interface to test results

        Notes:
            # Download pretrained weights from
            # https://github.com/alexgkendall/SegNet-Tutorial/blob/master/Example_Models/segnet_model_zoo.md
            http://mi.eng.cam.ac.uk/~agk34/resources/SegNet/segnet_basic_camvid.caffemodel
        """
        # Import the right version of caffe
        print(ub.color_text('[segnet] begin evaulation', 'blue'))
        # if harn.test_model_fpath is None:
        #     harn.prepare_test_model()
        # else:
        # harn.prepare_from_model(harn.test_model_fpath)

        n_iter = int(harn.n_test_input / harn.test_batch_size)

        model_info = iface.parse_model_info(harn.test_model_fpath)
        need_read = not model_info['image_input_fpath']
        need_read = True
        print('need_read = {!r}'.format(need_read))

        labels = np.arange(len(harn.task.classnames))
        im_size = cv2.imread(harn.test_gt_paths[0], cv2.IMREAD_UNCHANGED).size
        sample_weight = np.ones(im_size, dtype=np.int64)
        per_image_confusion = []

        def load_batch_data(net, input_paths, bx):
            offset = bx * harn.test_batch_size
            for jx in range(harn.test_batch_size):
                # push data into the network
                ix = offset + jx
                im_fpath = input_paths[ix]
                im_hwc = cv2.imread(im_fpath, flags=cv2.IMREAD_UNCHANGED)
                im_chw = np.transpose(im_hwc, (2, 0, 1)).astype(np.float32)
                net.blobs['data'].data[jx, :, :, :] = im_chw

        harn.make_dumpsafe_paths()
        net = harn.make_net()
        total_cfsn = pd.DataFrame(0, index=harn.task.classnames, columns=harn.task.classnames)
        for bx in ub.ProgIter(range(n_iter), label='forward batch', freq=1):
            if need_read:
                load_batch_data(net, harn.test_im_paths, bx)

            net.forward()
            blobs = net.blobs
            harn.dump_predictions(blobs, bx, have_true=True)

            assert harn.test_batch_size == 1
            jx = 0
            if 'argmax' in blobs:
                offset = bx * harn.test_batch_size
                ix = offset + jx
                gt_fpath = harn.test_gt_paths[ix]
                true = cv2.imread(gt_fpath, flags=cv2.IMREAD_UNCHANGED)
                pred = blobs['argmax'].data[jx][0].astype(np.int)
            else:
                true = blobs['label'].data[jx][0].astype(np.int)
                pred = blobs['prob'].data[jx].argmax(axis=0).astype(np.int)

            cfsn = metrics.confusion_matrix(true.ravel(), pred.ravel(),
                                            labels=labels,
                                            sample_weight=sample_weight)
            cfsn_df = pd.DataFrame(cfsn, index=harn.task.classnames, columns=harn.task.classnames)
            total_cfsn += cfsn_df

            sofar_ious = metrics.jaccard_score_from_confusion(total_cfsn)
            sofar_ious = sofar_ious.drop(harn.task.ignore_classnames)
            sofar_miou = sofar_ious.mean(skipna=True)
            print('sofar_miou = {!r}'.format(sofar_miou))
            # print('ious =\n{!r}'.format(ious))
            per_image_confusion.append(cfsn)

        # Get confusion matrix across all pixels in all images
        # from six.moves import reduce
        # total_cfsn = reduce(np.add, per_image_confusion)
        # import pandas as pd
        # cfsn_df = pd.DataFrame(total_cfsn, index=harn.task.classnames, columns=harn.task.classnames)
        ious = metrics.jaccard_score_from_confusion(total_cfsn)
        print('ious =\n{!r}'.format(ious))
        global_miou = ious.drop(harn.task.ignore_classnames).mean(skipna=True)
        print('global_miou = {!r}'.format(global_miou))

        results_fpath = join(harn.test_dump_dpath, 'results.json')
        results = {
            'total_cfsn': total_cfsn.to_json(),
            'ious': ious.to_json(),
            'global_miou': global_miou,
        }
        json.dump(results, open(results_fpath, 'w'))

        return global_miou
        # print('[segnet] end evaulation')


def shortest_unique_prefixes(items, mode=3):
    """
    Args:
        items (list of str): strings for which to find the shortest unique
            prefix
        mode (int): changes the implementation. mode=3 is the overall best,
            but requires mucking with private data in pygtrie. The modes 0 and
            1 are more naive implementations.

    References:
        http://www.geeksforgeeks.org/find-all-shortest-unique-prefixes-to-represent-each-word-in-a-given-list/
        https://github.com/Briaares/InterviewBit/blob/master/Level6/Shortest%20Unique%20Prefix.cpp

    Requires:
        pip install pygtrie

    Doctest:
        >>> from pysseg.fnameutil import *
        >>> items = ["zebra", "dog", "duck", "dove"]
        >>> shortest_unique_prefixes(items)
        ['z', 'dog', 'du', 'dov']

    Doctest:
        >>> # make numbers larger to stress test
        >>> # L = max length of a string, N = number of strings,
        >>> # C = smallest gaurenteed common length
        >>> # (the setting N=10000, L=100, C=20 is feasible we are good)
        >>> import random
        >>> rng = random.Random(0)
        >>> N, L, C = 1000, 10, 0
        >>> items = [''.join(['a' if i < C else chr(rng.randint(97, 122))
        >>>                     for i in range(L)]) for _ in range(N)]
        >>> shortest_unique_prefixes(items)

    Timeing:
        >>> import random
        >>> def make_data(N, L, C):
        >>>     rng = random.Random(0)
        >>>     return [''.join(['a' if i < C else chr(rng.randint(97, 122))
        >>>                      for i in range(L)]) for _ in range(N)]

        >>> items = make_data(N=1000, L=10, C=0)
        >>> %timeit shortest_unique_prefixes(items, mode=0)
        76 ms ± 499 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)
        >>> %timeit shortest_unique_prefixes(items, mode=1)
        25.6 ms ± 439 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)
        >>> %timeit shortest_unique_prefixes(items, mode=3)
        17.5 ms ± 244 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

        >>> items = make_data(N=1000, L=100, C=0)
        >>> %timeit shortest_unique_prefixes(items, mode=0)
        4.61 s ± 122 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
        >>> %timeit shortest_unique_prefixes(items, mode=1)
        26 ms ± 610 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)
        >>> %timeit shortest_unique_prefixes(items, mode=3)
        141 ms ± 1.05 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)

        >>> items = make_data(N=1000, L=100, C=70)
        >>> %timeit shortest_unique_prefixes(items, mode=0)
        4.61 s ± 122 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
        >>> %timeit shortest_unique_prefixes(items, mode=1)
        2.76 s ± 44.8 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
        >>> %timeit shortest_unique_prefixes(items, mode=3)
        141 ms ± 1.05 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)

        >>> items = make_data(N=10000, L=250, C=20)
        >>> # Mode 0 is too slow for this case.
        >>> %timeit shortest_unique_prefixes(items, mode=1)
        7.33 s ± 1.42 s per loop (mean ± std. dev. of 7 runs, 1 loop each)
        >>> %timeit shortest_unique_prefixes(items, mode=3)
        3.55 s ± 23 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
    """
    if len(set(items)) != len(items):
        raise ValueError('inputs must be unique')

    elif mode == 0:
        import pygtrie
        data = pygtrie.CharTrie()
        # Build a trie containing the frequency of all prefixes
        for item in items:
            for i in range(1, len(item) + 1):
                key = item[:i]
                data[key] = data.get(key, 0) + 1
    elif mode == 1:
        import pygtrie
        data = pygtrie.CharTrie()
        # Build a trie containing the frequency of all prefixes
        # (as an optimization for the average case, build the trie using only
        # the first few chars from each string, and return if the prefixes are
        # unique. Otherwise, continue until they are).
        L = max(map(len, items))
        for i in range(1, L + 1):
            # List all possible prefixes of len i
            for item in items:
                if len(item) >= i:
                    key = item[:i]
                    data[key] = data.get(key, 0) + 1

            # Stop early if all items have at least one prefix with freq=1
            if all(any(freq == 1 for _, freq in data.prefixes(item))
                   for item in items):
                break
    elif mode == 3:
        def _hack_build_pygtrie_prefix_freq(items):
            """
            Builds the trie, and then modifies the internal nodes so the
            sentinal values become the frequenceies we need.
            """
            import pygtrie
            from collections import deque

            # construct tree
            self = pygtrie.CharTrie(zip(items, [0] * len(items)))

            # Hack into the internal structure and insert frequencies at each node
            def _iternodes(self):
                """
                Generates all nodes in the trie
                """
                stack = deque([[self._root]])
                while stack:
                    for node in stack.pop():
                        yield node
                        stack.append(node.children.values())

            for node in _iternodes(self):
                if node is not self._root:   # dont do this to the root
                    node.value = 0

            # For each item trace its path and increment frequencies
            for item in items:
                final_node, trace = self._get_node(item)
                for key, node in trace[1:]:
                    node.value += 1
            return self

        data = _hack_build_pygtrie_prefix_freq(items)

    # Query for the first prefix with frequency 1 for each item.
    # This is the shortest unique prefix over all items.
    unique = []
    for item in items:
        freq = None
        for prefix, freq in data.prefixes(item):
            if freq == 1:
                break
        assert freq == 1, 'item={} has no unique prefix'.format(item)
        unique.append(prefix)
    return unique

    @classmethod
    def from_argv(Harness, argv=None):
        if argv is None:
            argv = sys.argv
        kwargs = {
            'test_model_fpath': ub.argval('--model', default=None, argv=argv),
            'test_weights_fpath': ub.argval('--weights', default=None, argv=argv),
            'workdir': ub.argval('--workdir', default='./work', argv=argv),
            # test_imdir = ub.argval('--test_imdir', default=None, argv=argv)
            # test_gtdir = ub.argval('--test_gtdir', default=None, argv=argv)
        }
        for key in list(kwargs.keys()):
            if kwargs[key] is not None:
                kwargs[key] = expanduser(kwargs[key])

        harn = Harness(**kwargs)
        return harn

    def prepare_from_model(harn, test_model_fpath):
        """
        Reads the model prototext to populate helpful info like batch size,
        number of inputs, etc...
        """
        info = iface.parse_model_info(test_model_fpath)
        image_input_fpath = info['image_input_fpath']

        harn.test_im_paths = []
        harn.test_gt_paths = []

        with open(image_input_fpath, 'r') as file:
            n_lines = 0
            for n_lines, line in enumerate(file, start=1):
                parts = line.split(' ')
                if len(parts) != 2:
                    raise IOError('cannot parse paths with spaces')
                harn.test_im_paths.append(parts[0])
                harn.test_gt_paths.append(parts[1])
        harn.test_batch_size = info['batch_size']
        harn.n_test_input = n_lines
