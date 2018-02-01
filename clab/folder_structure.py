import os
from os.path import join, normpath, dirname
import ubelt as ub
from clab import util


def symlink(real_path, link_path, overwrite=False, on_error='raise',
            verbose=2):
    """
    Attempt to create a symbolic link.

    TODO:
        Can this be fixed on windows?

    Args:
        path (str): path to real file or directory
        link_path (str): path to desired location for symlink
        overwrite (bool): overwrite existing symlinks (default = False)
        on_error (str): strategy for dealing with errors.
            raise or ignore
        verbose (int):  verbosity level (default=2)

    Returns:
        str: link path

    CommandLine:
        python -m utool.util_path symlink
    """
    path = normpath(real_path)
    link = normpath(link_path)
    if verbose:
        print('[util_path] Creating symlink: path={} link={}'.format(path, link))
    if os.path.islink(link):
        if verbose:
            print('[util_path] symlink already exists')
        os_readlink = getattr(os, "readlink", None)
        if callable(os_readlink):
            if os_readlink(link) == path:
                if verbose > 1:
                    print('[path] ... and points to the right place')
                return link
        else:
            print('[util_path] Warning, symlinks are not implemented on windows')
        if verbose > 1:
            print('[util_path] ... but it points somewhere else')
        if overwrite:
            ub.delete(link, verbose > 1)
        elif on_error == 'ignore':
            return False
    try:
        os_symlink = getattr(os, "symlink", None)
        if callable(os_symlink):
            os_symlink(path, link)
        else:
            raise NotImplementedError('')
            # win_shortcut(path, link)
    except Exception as ex:
        # import utool as ut
        # checkpath(link, verbose=True)
        # checkpath(path, verbose=True)
        do_raise = (on_error == 'raise')
        if do_raise:
            raise
    return link


class FolderStructure(object):
    def __init__(self, workdir='.', hyper=None, datasets=None):
        self.datasets = datasets
        self.workdir = workdir
        self.hyper = hyper

    def train_info(self, short=True, hashed=True):
        # TODO: needs MASSIVE cleanup and organization

        # TODO: if pretrained is another clab model, then we should read that
        # train_info if it exists and append it to a running list of train_info
        hyper = self.hyper

        arch = hyper.model_cls.__name__

        # setup a short symlink directory as well
        link_base = join(self.workdir, 'link', arch)
        arch_base = join(self.workdir, 'arch', arch)

        if 'train' in hyper.input_ids:
            # NEW WAY
            input_id = hyper.input_ids['train']
        else:
            # OLD WAY
            input_id = self.datasets['train'].input_id
            if callable(input_id):
                input_id = input_id()

        train_hyper_id_long = hyper.hyper_id()
        train_hyper_id_brief = hyper.hyper_id(short=short, hashed=hashed)
        train_hyper_hashid = ub.hash_data(train_hyper_id_long)[:8]
        other_id = hyper.other_id()

        aug_brief = 'au' + ub.hash_data(hyper.augment)[0:6]

        train_id = '{}_{}_{}_{}'.format(
            ub.hash_data(input_id)[:6], train_hyper_id_brief,
            aug_brief, other_id)

        full_dname = 'fit_{}'.format(train_id)

        train_hyper_id_hashed = hyper.hyper_id(hashed=True)
        link_dname = '{}_{}_{}_{}'.format(
            ub.hash_data(input_id)[:6], train_hyper_id_hashed,
            aug_brief, ub.hash_data(other_id)[0:4])
        # link_dname = ub.hash_data(full_dname)[0:8]

        input_dname = 'input_' + input_id
        link_dpath = join(link_base, input_dname, link_dname)
        train_dpath = join(arch_base, input_dname, full_dname)

        # make temporary initializer so we can infer the history
        temp_initializer = hyper.make_initializer()
        init_history = temp_initializer.history()

        train_info =  {
            'workdir': self.workdir,
            'train_id': train_id,
            'train_dpath': train_dpath,
            'input_id': input_id,
            'other_id': other_id,
            'hyper': hyper.get_initkw(),
            'train_hyper_id_long': train_hyper_id_long,
            'train_hyper_id_brief': train_hyper_id_brief,
            'train_hyper_hashid': train_hyper_hashid,
            'init_history': init_history,
            'init_history_hashid': ub.hash_data(util.make_idstr(init_history)),
            'link_dname': link_dname,
            'link_dpath': link_dpath,

            # TODO: add in n_classes if applicable
            # TODO: add in centering if applicable

            # HACKED IN
            'augment': hyper.augment,
            'aug_brief': aug_brief,
        }
        return train_info

    def setup_dpath(self, short=True, hashed=True):
        train_info = self.train_info(short, hashed)

        train_dpath = ub.ensuredir(train_info['train_dpath'])
        train_info_fpath = join(train_dpath, 'train_info.json')

        util.write_json(train_info_fpath, train_info)

        # setup symlinks
        ub.ensuredir(dirname(train_info['link_dpath']))
        symlink(train_info['train_dpath'], train_info['link_dpath'],
                on_error='ignore')

        print('+=========')
        # print('hyper_strid = {!r}'.format(params.hyper_id()))
        # print('train_init_id = {!r}'.format(train_info['input_id']))
        # print('arch = {!r}'.format(train_info['arch_id']))
        # print('train_hyper_hashid = {!r}'.format(train_info['train_hyper_hashid']))
        print('hyper = {}'.format(ub.repr2(train_info['hyper'], nl=3)))
        print('train_hyper_id_brief = {!r}'.format(train_info['train_hyper_id_brief']))
        print('train_id = {!r}'.format(train_info['train_id']))
        print('+=========')
        return train_info
