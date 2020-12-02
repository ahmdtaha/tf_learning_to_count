import os
import getpass
import constants
import os.path as osp
username = getpass.getuser()

def get_checkpoint_dir(exp_name):

    project_name = osp.basename(osp.abspath('./'))
    # if username == 'ahmdtaha':
    #     ckpt_dir = '/vulcanscratch/ahmdtaha/checkpoints'
    # elif username == 'ataha':
    #     ckpt_dir = '/mnt/data/checkpoints'
    # elif username == 'ahmedtaha':
    #     ckpt_dir = '/Users/ahmedtaha/Documents/checkpoints'
    # else:
    #     raise NotImplementedError('Invalid username {}'.format(username))

    ckpt_dir = constants.checkpoint_save_dir
    assert osp.exists(ckpt_dir),('{} does not exists'.format(ckpt_dir))

    ckpt_dir = '{}/{}/{}'.format(ckpt_dir,project_name,exp_name)
    return ckpt_dir


if __name__ == '__main__':
    print(get_checkpoint_dir('test_exp'))
