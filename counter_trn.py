import os
import sys
import json
import time
import constants
import numpy as np
import os.path as osp
import multiprocessing
import tensorflow as tf

from utils import os_utils
from utils import  lr_scheduler
from nets.model import Counter_Model
import datasets.dali_tf_flow as dataset
from config.base_config import BaseConfig

from tensorpack import TensorInput
from tensorpack.utils import logger
from tensorpack.utils.gpu import get_nr_gpu
from tensorpack.callbacks.saver import ModelSaver
from tensorpack.callbacks.monitor import JSONWriter
from tensorpack.train.config import AutoResumeTrainConfig
from tensorpack.input_source import QueueInput,StagingInput
from tensorpack.train.interface import launch_train_with_config
from tensorpack.train import SyncMultiGPUTrainerParameterServer
from tensorpack.callbacks import EstimatedTimeLeft,MinSaver,ScheduledHyperParamSetter,InferenceRunner,PeriodicTrigger,MergeAllSummaries
from tensorpack.callbacks import GPUUtilizationTracker,GPUMemoryTracker


class Trainer(object):

    def __init__(self,config):
        self.config = config

        # hyper_parameter_str = config.dataset+'_lr_'+str(config.learning_rate)

        if self.config.gpu:
            os.environ['CUDA_VISIBLE_DEVICES'] = self.config.gpu

        nr_gpu = get_nr_gpu()
        self.config.nr_gpu = nr_gpu
        os_utils.touch_dir(self.config.train_dir)

        # log_file = osp.join(self.config.train_dir, 'train_log.txt')
        # self.logger = log_utils.create_logger(log_file)
        # self.logger.info("Train Dir: %s", self.config.train_dir)

        args_file = osp.join(self.config.train_dir, 'args.json')
        with open(args_file, 'w') as f:
            json.dump(vars(config), f, ensure_ascii=False, indent=2, sort_keys=True)

        # --- input ops ---
        self.batch_size = config.batch_size
        # --- create model ---
        self.model = Counter_Model(config)



        TOTAL_BATCH_SIZE = int(config.batch_size)
        BATCH_SIZE = TOTAL_BATCH_SIZE // nr_gpu
        config.batch = BATCH_SIZE

        logger.set_logger_dir(self.config.train_dir, 'k')
        config = get_tp_config(self.model, config)
        launch_train_with_config(config, SyncMultiGPUTrainerParameterServer(nr_gpu))


def get_tp_config(model, option):

    lr_string = option.learning_rate_var_name





    # Also show all parameter values at the start, for ease of reading logs.
    logger.info('Training using the following parameters:')
    for key, value in sorted(vars(option).items()):
        logger.info('{}: {}'.format(key, value))

    lr_schedule = lr_scheduler.cnt_lr_scheduler(option)
    callbacks = [
        ModelSaver(max_to_keep=2, keep_checkpoint_every_n_hours=1000),
        # min_saver_callback,
        EstimatedTimeLeft(),
        GPUUtilizationTracker(),
        GPUMemoryTracker(),
        MergeAllSummaries(),
        ScheduledHyperParamSetter(lr_string, lr_schedule),
    ]

    steps_per_epoch = option.max_iters


    option.prefix = constants.datsets_dir
    # train_data = QueueInput(dataset.dali_tensors(option.batch_size,option.prefix))
    train_data =  TensorInput(lambda  : dataset.dali_tensors(option.batch_size,option.prefix,True,n_threads = multiprocessing.cpu_count() // get_nr_gpu()),steps_per_epoch)
    # input = QueueInput(train_data)
    # input = StagingInput(train_data, nr_stage=1)
    # steps_per_epoch = 5
    return AutoResumeTrainConfig(
        model=model,
        data=train_data,
        # dataflow=dataset_train_flow,
        # data=input,
        callbacks=callbacks,
        steps_per_epoch=int(steps_per_epoch),
        max_epoch=option.epoch,
    )


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

    imagenet_size = 1281167
    arg_batch_size = 256
    arg_svd_lm = 0
    num_gpus = len(arg_gpus.split(','))
    arg_num_iters = (imagenet_size)// ((arg_batch_size * num_gpus)) #  Multiple by 2 because A batch b has b/2 xs and b/2 ys

    for arg_optimizer in  ['adam']: #['adam','momentum','gd']:
        for arg_lr in [0.0001]: #[0.1,0.01,0.001,0.0001]:
    # arg_lr = 0.0001
            arg_logits_dim = '128'
            arg_enable_batch_norm = False
            arg_lrn_enabled = False
            arg_drop_enabled = False
            # arg_optimizer = 'adam'
            arg_exp_name = 'counter_padding_DROP{}_WoReg_BN{}_LRN{}_b{}_200Plr{}_dim{}_svdlm{}_gpu{}_opt{}'.format(
                arg_drop_enabled,arg_enable_batch_norm,arg_lrn_enabled,arg_batch_size, arg_lr,arg_logits_dim,arg_svd_lm,arg_gpus.replace(',','_'),arg_optimizer)

            # arg_lr = arg_lr*0.1
            # arg_exp_name = 'debug'
            base_args = [
                '--batch_size',str(arg_batch_size),
                '--max_iters',str(arg_num_iters),
                '--exp_name',arg_exp_name,
                '--net', 'alexnet',
                '--learning_rate',str(arg_lr),
                '--gpu',arg_gpus,
                '--svd_lm',str(arg_svd_lm),
                '--logits_dim',arg_logits_dim,
                '--opt', arg_optimizer,
                # '--gpu', '0',
                '--epoch', '250',
                '--learning_rate_var_name','lr_cnt',
            ]

            if arg_enable_batch_norm:
                base_args.extend(['--enable_batch_norm'])

            if arg_lrn_enabled:
                base_args.extend(['--lrn_enabled'])

            if arg_drop_enabled:
                base_args.extend(['--drop_enabled'])

            config  = BaseConfig().parse_lst(base_args)

            config.img_height,config.img_width,config.img_ch = 256,256,3
            config.num_cls = 1000

            trainer = Trainer(config)
            tf.compat.v1.reset_default_graph()


if __name__ == '__main__':
    # print(len(sys.argv))
    if len(sys.argv) == 1:
        main()
    else:
        config = BaseConfig().parse()

        config.img_height, config.img_width, config.img_ch = 256, 256, 3
        config.num_cls = 1000

        # print(config.learning_rate_var_name)

        trainer = Trainer(config)
        tf.compat.v1.reset_default_graph()

