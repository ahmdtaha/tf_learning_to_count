import argparse
import os.path as osp
import tensorflow as tf
from utils import path_utils

class BaseConfig:

    def __init__(self):

        parser = argparse.ArgumentParser()
        parser.add_argument('--batch_size', type=int, default=32)
        parser.add_argument('--prefix', type=str, default='default')
        parser.add_argument('--checkpoint', type=str,default=None)
        parser.add_argument('--dataset', type=str, default='ImageNet', choices=['ImageNet'])
        parser.add_argument('--learning_rate', type=float, default=1e-4)
        parser.add_argument('--learning_rate_var_name', type=str, required=True)
        parser.add_argument('--lr_weight_decay', action='store_true', default=False)
        parser.add_argument('--net', type=str,default='alexnet')
        parser.add_argument('--exp_name', type=str, required=True)
        parser.add_argument('--cnt_exp_name', type=str,default='', required=False)
        parser.add_argument('--gpu', type=str, default='0')
        parser.add_argument('--max_iters', type=int)
        parser.add_argument('--epoch', type=int,default=100)
        parser.add_argument('--logits_dim', type=int, default=1000)
        parser.add_argument('--opt', type=str, default='momentum')
        parser.add_argument('--pretrained', action='store_true', default=False)


        # parser.add_argument('--enable_batch_norm', action='store_true', default=False)
        # parser.add_argument('--lrn_enabled', action='store_true', default=False)
        # parser.add_argument('--drop_enabled', action='store_true', default=False,help='Dropout enabled')
        # parser.add_argument('--conv_classifier_idx', type=int, default=0)




        self.parser = parser

    def parse_lst(self,args):
        cfg = self.parser.parse_args(args)
        exp_name = cfg.exp_name
        cfg.train_dir = path_utils.get_checkpoint_dir(exp_name)
        cfg.cnt_exp_name = path_utils.get_checkpoint_dir(cfg.cnt_exp_name)
        if osp.exists(cfg.train_dir):
            ## set the checkpoint
            cfg.checkpoint = tf.train.latest_checkpoint(cfg.train_dir)

        return cfg

    def parse(self):
        cfg = self.parser.parse_args()
        exp_name = cfg.exp_name
        cfg.train_dir = path_utils.get_checkpoint_dir(exp_name)
        cfg.cnt_exp_name = path_utils.get_checkpoint_dir(cfg.cnt_exp_name)
        if osp.exists(cfg.train_dir):
            ## set the checkpoint
            cfg.checkpoint = tf.train.latest_checkpoint(cfg.train_dir)

        return cfg