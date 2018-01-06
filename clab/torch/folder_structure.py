from os.path import join
import ubelt as ub
from clab import util


class FolderStructure(object):
    def __init__(self, workdir='.', hyper=None, datasets=None):
        self.datasets = datasets
        self.workdir = workdir
        self.hyper = hyper

    def train_info(self, short=True, hashed=True):
        # TODO: if pretrained is another clab model, then we should read that
        # train_info if it exists and append it to a running list of train_info
        hyper = self.hyper

        arch = hyper.model_cls.__name__
        # arch_id = hyper.model_id()

        # arch_hashid = hyper.model_id(brief=True)

        arch_dpath = join(self.workdir, 'arch', arch)
        train_base = join(arch_dpath, 'train')

        if 'train' in hyper.input_ids:
            # NEW WAY
            input_id = hyper.input_ids['train']
        else:
            # OLD WAY
            input_id = self.datasets['train'].input_id

        train_hyper_id_long = hyper.hyper_id()
        train_hyper_id_brief = hyper.hyper_id(short=short, hashed=hashed)
        train_hyper_hashid = util.hash_data(train_hyper_id_long)[:8]
        other_id = hyper.other_id()

        train_id = '{}_{}_{}'.format(
            util.hash_data(input_id)[:6], train_hyper_id_brief, other_id)

        train_dpath = join(
            train_base,
            'input_' + input_id, 'fit_{}'.format(train_id)
        )
        # TODO: needs MASSIVE cleanup and organization

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
            'init_history_hashid': util.hash_data(util.make_idstr(init_history)),
        }
        return train_info

    def setup_dpath(self, short=True, hashed=True):
        train_info = self.train_info(short, hashed)

        train_dpath = ub.ensuredir(train_info['train_dpath'])
        train_info_fpath = join(train_dpath, 'train_info.json')

        util.write_json(train_info_fpath, train_info)

        print('+=========')
        # print('hyper_strid = {!r}'.format(params.hyper_id()))
        # print('train_init_id = {!r}'.format(train_info['input_id']))
        # print('arch = {!r}'.format(train_info['arch_id']))
        # print('train_hyper_hashid = {!r}'.format(train_info['train_hyper_hashid']))
        print('hyper = {}'.format(ub.repr2(train_info['hyper'], nl=3)))
        print('train_hyper_id_brief = {!r}'.format(train_info['train_hyper_id_brief']))
        print('train_id = {!r}'.format(train_info['train_id']))
        print('+=========')
        train_dpath = ub.ensuredir(train_info['train_dpath'])
        return train_dpath


# def make_training_dpath(workdir, arch, datasets, hyper,
#                         pretrained=None,
#                         train_hyper_id=None, suffix=''):
#     """
#     from clab.torch.sseg_train import *
#     datasets = load_task_dataset('urban_mapper_3d')
#     datasets['train']._make_normalizer()
#     arch = 'foobar'
#     workdir = datasets['train'].task.workdir
#     ut.exec_funckw(directory_structure, globals())
#     """
#     # workdir = os.path.expanduser('~/data/work/pycamvid')
#     arch_dpath = ub.ensuredir((workdir, 'arch', arch))
#     train_base = ub.ensuredir((arch_dpath, 'train'))
#     test_base = ub.ensuredir((arch_dpath, 'test'))
#     test_dpath = ub.ensuredir((test_base, 'input_' + datasets['test'].input_id))

#     train_init_id = pretrained
#     train_hyper_hashid = util.hash_data(train_hyper_id)[:8]

#     train_id = '{}_{}_{}_{}'.format(
#         datasets['train'].input_id, arch, train_init_id, train_hyper_hashid) + suffix

#     train_dpath = ub.ensuredir((
#         train_base,
#         'input_' + datasets['train'].input_id, 'fit_{}'.format(train_id)
#     ))

#     train_info =  {
#         'arch': arch,
#         'train_id': datasets['train'].input_id,
#         'train_hyper_id': train_hyper_id,
#         'train_hyper_hashid': train_hyper_hashid,
#         'colorspace': datasets['train'].colorspace,
#     }
#     if hasattr(datasets['train'], 'center_inputs'):
#         # Hack in centering information
#         # TODO: better serialization
#         train_info['hack_centers'] = [
#             (t.__class__.__name__, t.__getstate__())
#             # ub.map_vals(str, t.__dict__)
#             for t in datasets['train'].center_inputs.transforms
#         ]
#     util.write_json(join(train_dpath, 'train_info.json'), train_info)

#     print('+=========')
#     # print('hyper_strid = {!r}'.format(params.hyper_id()))
#     print('train_init_id = {!r}'.format(train_init_id))
#     print('arch = {!r}'.format(arch))
#     print('train_hyper_hashid = {!r}'.format(train_hyper_hashid))
#     print('train_hyper_id = {!r}'.format(train_hyper_id))
#     print('train_id = {!r}'.format(train_id))
#     print('+=========')

#     return train_dpath, test_dpath
