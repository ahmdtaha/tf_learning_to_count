from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
import sys
import constants
import os.path as osp
import multiprocessing
import tensorflow as tf
from utils import os_utils
from utils import lr_scheduler
from nets.model import Counter_Model
import datasets.dali_tf_flow as dataset

from config.base_config import BaseConfig
from tensorpack import TensorInput
from tensorpack.utils import logger
from tensorpack.utils.gpu import get_nr_gpu
from tensorpack.callbacks.saver import ModelSaver
from tensorpack.tfutils.sessinit import SmartInit,SaverRestore
from tensorpack.callbacks.monitor import JSONWriter
from tensorpack.callbacks.inference import ScalarStats
from tensorpack.train.config import AutoResumeTrainConfig,TrainConfig
from tensorpack.input_source import QueueInput,StagingInput
from tensorpack.callbacks.inference import ClassificationError
from tensorpack.train.interface import launch_train_with_config
from tensorpack.train import SyncMultiGPUTrainerParameterServer
from tensorpack.callbacks import EstimatedTimeLeft,MinSaver,ScheduledHyperParamSetter,InferenceRunner,PeriodicTrigger,MergeAllSummaries
from tensorpack.callbacks import GPUUtilizationTracker,GPUMemoryTracker


class Trainer(object):

    def __init__(self,config):
        self.config = config

        if self.config.gpu:
            # print(self.config.gpu)
            os.environ['CUDA_VISIBLE_DEVICES'] = self.config.gpu
        os_utils.touch_dir(self.config.train_dir)


        # --- input ops ---
        self.batch_size = config.batch_size

        # --- create model ---
        self.model = Counter_Model(config,linear_classifier=True)

        all_trn_vars = tf.compat.v1.trainable_variables()
        print(len(all_trn_vars),all_trn_vars)

        logger.set_logger_dir(self.config.train_dir, 'k')

        logger.info('Training using the following parameters:')
        for key, value in sorted(vars(self.config).items()):
            logger.info('{}: {}'.format(key, value))

        lr_string = self.config.learning_rate_var_name
        START_LR = self.config.learning_rate

        lr_schedule = lr_scheduler.cls_lr_scheduler(self.config)

        callbacks = [
            ModelSaver(max_to_keep=2, keep_checkpoint_every_n_hours=1000),
            # min_saver_callback,
            EstimatedTimeLeft(),
            GPUUtilizationTracker(),
            GPUMemoryTracker(),
            MergeAllSummaries(),
            ScheduledHyperParamSetter(lr_string, lr_schedule),
        ]

        # import getpass
        # username = getpass.getuser()
        # if username == 'ataha':  ## honda machine
        #     # self.config.prefix = '/mnt/data/datasets/imagenet/ILSVRC/Data/CLS-LOC/'
        #     self.config.prefix = '/mnt/data/datasets/imagenet/ILSVRC/Data/CLS-LOC/'
        # else:
        #     # option.prefix = '/fs/vulcan-scratch/datasets/imagenet/train'
        #     # self.config.prefix = '/vulcan/scratch/ahmdtaha/datasets/imagenet/ILSVRC/Data/CLS-LOC/'
        #     self.config.prefix = '/scratch0/ahmdtaha/imagenet/'

        self.config.prefix = constants.datsets_dir
        steps_per_epoch = self.config.max_iters
        # steps_per_epoch = 5

        n_threads = multiprocessing.cpu_count()
        # n_threads = 4
        train_data = TensorInput(lambda: dataset.dali_tensors(self.config.batch_size, self.config.prefix,True,n_threads=n_threads // get_nr_gpu()), steps_per_epoch)

        infs =  [
                 # ClassificationError('alexnet_v2_conv{}_Relu_acc_top1'.format(self.config.conv_classifier_idx+1),
                 #                       'val-alexnet_v2_conv{}_Relu_acc_top1'.format(self.config.conv_classifier_idx+1)),

                 ClassificationError('alexnet_v2_conv1_Relu_acc_top1','val-alexnet_v2_conv1_Relu_acc_top1'),
                 ClassificationError('alexnet_v2_conv2_Relu_acc_top1','val-alexnet_v2_conv2_Relu_acc_top1'),
                 ClassificationError('alexnet_v2_conv3_Relu_acc_top1','val-alexnet_v2_conv3_Relu_acc_top1'),
                 ClassificationError('alexnet_v2_conv4_Relu_acc_top1','val-alexnet_v2_conv4_Relu_acc_top1'),
                 ClassificationError('alexnet_v2_conv5_Relu_acc_top1','val-alexnet_v2_conv5_Relu_acc_top1'),
                 ]
        val_batch = self.config.batch_size
        imagenet_val_size = 50000
        val_steps_per_epoch = imagenet_val_size//val_batch
        val_data = TensorInput(lambda: dataset.dali_tensors(val_batch, self.config.prefix,False,n_threads=4), val_steps_per_epoch)
        callbacks.extend([PeriodicTrigger(InferenceRunner(val_data,infs), every_k_epochs=2)])

        assert osp.exists(self.config.cnt_exp_name), ' The unsupervised-counter exp name is not valid'

        path_ckpt = tf.train.latest_checkpoint(self.config.cnt_exp_name)
        if self.config.pretrained:
            train_config =  AutoResumeTrainConfig(
                model=self.model,
                data=train_data,
                session_init=SmartInit(path_ckpt),
                callbacks=callbacks,
                steps_per_epoch=int(steps_per_epoch),
                max_epoch=self.config.epoch,
                # optimizer=self.model.model_optimizer,
            )
        else:
            train_config = AutoResumeTrainConfig(
                model=self.model,
                data=train_data,
                callbacks=callbacks,
                steps_per_epoch=int(steps_per_epoch),
                max_epoch=self.config.epoch,
            )


        nr_gpu = get_nr_gpu()
        self.config.nr_gpu = nr_gpu
        print('Number of GPUs {}'.format(nr_gpu))
        launch_train_with_config(train_config, SyncMultiGPUTrainerParameterServer(nr_gpu))


def check_data_path(path):
    if os.path.isfile(os.path.join(path, 'data.hy')) \
           and os.path.isfile(os.path.join(path, 'id.txt')):
        return True
    else:
        return False


def main():
    print(tf.test.is_gpu_available())
    assert tf.test.is_gpu_available(), 'Something is wrong with GPU usage'

    nr_gpu = get_nr_gpu()
    if nr_gpu == 1:
        arg_gpus = '0'
    elif nr_gpu == 2:
        arg_gpus = '0,1'
    elif nr_gpu == 3:
        arg_gpus = '0,1,2'
    elif nr_gpu == 4:
        arg_gpus = '0,1,2,3'
    else:
        raise NotImplementedError('Something is wrong with num_gpus {}'.format(nr_gpu))

    arg_batch_size = 256
    imagenet_size = 1281167

    num_gpus = len(arg_gpus.split(','))
    arg_num_iters = imagenet_size // (arg_batch_size * num_gpus)


    arg_logits_dim = 1000
    # arg_unsupervised_exp_name = 'counter_WHL_BATCH_randresize_256_DROPFalse_WoReg_BNFalse_LRNFalse_b256_09lr0.0001_dim1000_svdlm0_gpu0_1_optadam_200e'

    arg_unsupervised_exp_name = 'counter_padding_DROPFalse_WoReg_BNFalse_LRNFalse_b256_200Plr0.0001_dim128_svdlm0_gpu0_1_2_3_optadam'
    # arg_unsupervised_exp_name = 'counter_wojitter_256_DROPFalse_WoReg_BNFalse_LRNFalse_b256_09lr1e-05_dim1000_svdlm0_gpu0_1_2_3_optmomentum'

    #arg_unsupervised_exp_name = 'count_fix_loss_halfBatch_WoReg_BNFalse_b256_09lr5e-05_dim1000_svdlm0_nr_gpu0_1'
    arg_pretrained = True
    # arg_conv_classifier_idx = 2 # 0 index [0:conv1, 1:conv2]
    arg_enable_batch_norm = False
    arg_lrn_enabled = False

    for arg_optimizer in ['adam']:#['adam','momentum','gd']:
        for arg_lr in [0.001]:#[0.1,0.01,0.001,0.0001]:
    # if arg_optimizer == 'adam':
    #     arg_lr = 0.001 #0.0001 # 0.00008
    # else:
    #     arg_lr = 0.01

            arg_epochs = str(300)
            # arg_exp_name = 'classifier_{}c_WoReg_train{}_bn{}_LRN{}_opt{}_b{}_100_50e_0.9lr{}_gpu{}_avg_pool_{}'\
            #               classifier_bilinear_5cls_WoReg_trainTrue_bnFalse_LRNFalse_optmomentum_b256_step_Sch_m1_7_lr1_gpu0_1_e300_avg_pool_
            arg_exp_name = 'classifier_fix_dim_5cls_WoReg_train{}_bn{}_LRN{}_opt{}_b{}_Polych_pwr2_m1_9_lr{}_gpu{}_e{}_avg_pool_{}'\
                .format(arg_pretrained,arg_enable_batch_norm,arg_lrn_enabled,arg_optimizer,
                        arg_batch_size, arg_lr, arg_gpus.replace(',', '_'),arg_epochs,arg_unsupervised_exp_name)

            # arg_epochs = str(350)
            # arg_epochs = str(100)
            # arg_lr = 0.0001 * 0.1

            base_args = [
                '--batch_size', str(arg_batch_size),
                '--max_iters', str(arg_num_iters),
                '--exp_name', arg_exp_name,
                '--cnt_exp_name', arg_unsupervised_exp_name,
                '--learning_rate', str(arg_lr),
                '--gpu', arg_gpus,
                '--opt',arg_optimizer,
                '--net', 'alexnet',
                '--epoch', arg_epochs,
                # '--logits_dim', str(arg_logits_dim),
                # '--conv_classifier_idx',str(arg_conv_classifier_idx),
                '--learning_rate_var_name', 'lr_cls',
            ]
            if arg_pretrained:
                base_args.extend(['--pretrained'])
            # if arg_enable_batch_norm:
            #     base_args.extend(['--enable_num_threadsbatch_norm'])
            # if arg_lrn_enabled:
            #     base_args.extend(['--lrn_enabled'])

            config  = BaseConfig().parse_lst(base_args)
            config.img_height, config.img_width, config.img_ch = 256, 256, 3
            config.num_cls = 1000


            Trainer(config)
            tf.compat.v1.reset_default_graph()


if __name__ == '__main__':
    if len(sys.argv) == 1:
        main()
    else:
        cfg = BaseConfig().parse()

        cfg.img_height, cfg.img_width, cfg.img_ch = 256, 256, 3
        cfg.num_cls = 1000

        imagenet_size = 1281167

        num_gpus = len(cfg.gpu.split(','))
        cfg.max_iters = imagenet_size // (cfg.batch_size * num_gpus)

        Trainer(cfg)
        tf.compat.v1.reset_default_graph()
