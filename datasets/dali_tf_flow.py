import os
import cv2
import time
import numpy as np
import os.path as osp
import multiprocessing
import tensorflow as tf
import nvidia.dali.ops as ops
import nvidia.dali.types as types
import nvidia.dali.plugin.tf as dali_tf
from nvidia.dali.pipeline import Pipeline
from timeit import default_timer as timer


class SimplePipeline(Pipeline):
    def __init__(self, prefix,is_train,batch_size, num_threads, device_id):
        super(SimplePipeline, self).__init__(batch_size, num_threads, device_id, seed = -1)

        self.device = 'gpu'
        if is_train:
            imagenet_fileroot = osp.join(prefix, 'train')
            imagenet_filelist = osp.join(prefix, 'caffe_list'+'/train.txt')



            self.input = ops.FileReader(file_root=imagenet_fileroot, random_shuffle=is_train,
                                        initial_fill=batch_size * num_threads,
                                        # file_list=imagenet_filelist
                                        )

            self.resize = ops.RandomResizedCrop(size=256,
                                                device=self.device,
                                                # image_type=types.RGB,
                                                interp_type=types.INTERP_LINEAR
                                                )
        else:
            imagenet_fileroot = osp.join(prefix, 'val')
            imagenet_filelist = osp.join(prefix, 'caffe_list' + '/val.txt')

            self.input = ops.FileReader(file_root=imagenet_fileroot, random_shuffle=is_train,
                                        initial_fill=batch_size,
                                        )


            self.resize = ops.Resize(device=self.device,
                                     image_type=types.RGB,
                                     interp_type=types.INTERP_LINEAR,resize_shorter = 256)

        assert osp.exists(imagenet_fileroot), '{} does not exist'.format(imagenet_fileroot)
        self.decode = ops.ImageDecoder(device='cpu', output_type=types.RGB)


        self.cmn = ops.CropMirrorNormalize(device=self.device,
                                           output_dtype=types.FLOAT,
                                           crop=(256, 256),
                                           image_type=types.RGB,
                                           output_layout=types.NHWC,
                                           mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                                           std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
                                     )
        self.uniform = ops.Uniform(range=(0.0, 1.0))

        self.resize_rng = ops.Uniform(range=(256, 480))
        #self.img_jitter = ops.Jitter(device="gpu")
        self.mirror_coin = ops.CoinFlip(probability=0.5)

        self.color_jitter = ops.ColorTwist(device="gpu")
        self.sat_rnd = ops.Uniform(range=[0.25, 1.0])
        self.cont_rnd = ops.Uniform(range=[0.25, 1.75])
        self.brig_rnd = ops.Uniform(range=[0.5, 1.5])

        self.hue_rnd = ops.Uniform(range=[-180, 180])
        # rotates the image, enlarging the canvas

        self.rotation = ops.Rotate(device="gpu", interp_type=types.INTERP_LINEAR)
        self.rotation_rng = ops.NormalDistribution()

        self.sphere_aug = ops.Sphere(device="gpu")
        self.sphere_coin = ops.CoinFlip(probability=0.25)

        self.jitter_aug = ops.Jitter(device="gpu")
        self.jitter_coin = ops.CoinFlip(probability=0.5)

        self.water_aug = ops.Water(device = "gpu")
        self.water_coin = ops.CoinFlip(probability=0.25)

        self.is_train = is_train

        self.rng1 = ops.Uniform(range=[0.5, 1.5])
        self.rng2 = ops.Uniform(range=[0.875, 1.125])
        self.rng3 = ops.Uniform(range=[-0.5, 0.5])

    def define_graph(self):
        jpegs, labels = self.input()
        images = self.decode(jpegs)
        if self.device == 'gpu':
            labels = labels.gpu()
            images = images.gpu()
        if self.is_train:
            # images = self.water_aug(images,mask=self.water_coin())

            # images = self.resize(images, resize_shorter=self.resize_rng())
            # images = self.rotation(images,angle=5*self.rotation_rng())

            # images = self.resize(images)
            # images = self.color_jitter(images, saturation=self.sat_rnd(), contrast=self.cont_rnd(), brightness=self.brig_rnd())


            images = self.resize(images)

            images = self.color_jitter(images, saturation=self.rng1(), contrast=self.rng1(), brightness=self.rng2(),
                                       hue=self.rng3())

            #images = self.jitter_aug(images,mask=self.jitter_coin())

            # images = self.color_jitter(images, saturation=self.sat_rnd(), contrast=self.cont_rnd(), brightness=self.brig_rnd(), hue=self.hue_rnd())

            images = self.cmn(images,mirror = self.mirror_coin())
        else:
            images = self.resize(images)
            images = self.cmn(images)

        print(labels)
        return (images, labels)

def dali_tensors(batch_size,prefix,is_train,n_threads = 4,device='0'):
    device = '/device:GPU:{}'.format(device)

    # if is_train:
    #
    # else:
    #     device = '/device:CPU:0'
    with tf.device(device):
        # n_threads = multiprocessing.cpu_count()
        pipes = [SimplePipeline(prefix,is_train,batch_size=batch_size, num_threads=n_threads, device_id=device_id) for device_id in
                 range(1)]
        serialized_pipes = [pipe.serialize() for pipe in pipes]
        del pipes
        daliop = dali_tf.DALIIterator()

        image, label = daliop(serialized_pipeline=serialized_pipes[0],
                              shapes=[(batch_size, 256, 256, 3), (batch_size,1)], dtypes=[tf.float32, tf.int32])
        print(image, label)
        # image = tf.Print(image,[tf.shape(image)],'print image shape')
    return image, label


def grap_dali_tensors(batch_size,prefix,is_train):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    tf.compat.v1.disable_eager_execution()

    with tf.device('/device:CPU:0'):
        batch_size = 32
        n_threads = 1  # multiprocessing.cpu_count() // 2
        pipes = [SimplePipeline(prefix,True,batch_size=batch_size, num_threads=n_threads, device_id=device_id) for device_id in
                 range(1)]
        serialized_pipes = [pipe.serialize() for pipe in pipes]
        del pipes
        daliop = dali_tf.DALIIterator()

        image, label = daliop(serialized_pipeline=serialized_pipes[0],
                              shapes=[(batch_size, 256, 256, 3), (batch_size, 1)], dtypes=[tf.float32, tf.int32])

        print(image, label)
        gpu_options = tf.GPUOptions()
        config = tf.ConfigProto(gpu_options=gpu_options)
        with tf.Session(config=config) as sess:
            _imgs,_lbls = sess.run([image, label])
            # if not is_train:
                # read_img = cv2.imread('/mnt/data/datasets/imagenet/ILSVRC/Data/CLS-LOC/val/ILSVRC2012_val_00000001.JPEG')
                # cv2.imwrite('./test.jpg', read_img)
            for i in range(batch_size):
                cv2.imwrite('./test_{}.jpg'.format(i+1), _imgs[i, :, :, :].astype(np.uint8))

            _imgs, _lbls = sess.run([image, label])
            # if not is_train:
            # read_img = cv2.imread('/mnt/data/datasets/imagenet/ILSVRC/Data/CLS-LOC/val/ILSVRC2012_val_00000001.JPEG')
            # cv2.imwrite('./test.jpg', read_img)
            for i in range(batch_size):
                cv2.imwrite('./test_1_{}.jpg'.format(i + 1), _imgs[i, :, :, :].astype(np.uint8))


if __name__ == '__main__':
    # print(dali_tensors(32, '/mnt/data/datasets/imagenet/ILSVRC/Data/CLS-LOC/train',True))
    # print(dali_tensors(32, '/mnt/data/datasets/imagenet/ILSVRC/Data/CLS-LOC/val', False))

    grap_dali_tensors(32, '/mnt/data/datasets/imagenet/ILSVRC/Data/CLS-LOC', True)