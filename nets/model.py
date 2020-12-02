from nets.ops import fc
import numpy as np
import tensorflow as tf
# from nets.vgg19 import vgg19
from nets.alexnet import alexnet_v2 as alexnet

from tensorpack import imgaug, dataset, ModelDesc
from tensorpack.tfutils.summary import add_moving_summary
from tensorpack.tfutils.tower import get_current_tower_context


class Counter_Model(ModelDesc):
    def __init__(self,config,
        linear_classifier = False,
        # is_train = True
                 ):

        ModelDesc.__init__(self)

        self.config = config
        self.batch_size = self.config.batch_size
        self.input_height = self.config.img_height
        self.input_width = self.config.img_width
        self.c_dim = self.config.img_ch
        self.num_class = self.config.num_cls

        # self.is_training = tf.compat.v1.placeholder_with_default(bool(is_train), [], name='is_training')
        self.linear_classifier = linear_classifier
        # self.batch_norm_layers = None

        lr = tf.get_variable(self.config.learning_rate_var_name, initializer=self.config.learning_rate, trainable=False)
        if self.config.opt == 'momentum':
            self.model_optimizer = tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True)
            self.current_lr = self.model_optimizer._learning_rate
        elif self.config.opt == 'adam':
            self.model_optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=lr)
            self.current_lr = self.model_optimizer._lr
        elif self.config.opt == 'adadelta':
            self.model_optimizer = tf.compat.v1.train.AdadeltaOptimizer(learning_rate=lr)
            self.current_lr = self.model_optimizer._learning_rate
        elif self.config.opt == 'rms':
            self.model_optimizer = tf.compat.v1.train.RMSPropOptimizer(learning_rate=lr, momentum=0.5)
            self.current_lr = self.model_optimizer._learning_rate
        elif self.config.opt == 'gd':
            self.model_optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=lr)
            self.current_lr = self.model_optimizer._learning_rate
        else:
            raise NotImplementedError('{} opt invalid'.format(self.config.opt))

    def inputs(self):
        return [tf.placeholder(tf.float32,(None, 256, 256, 3), 'imgs'),
                tf.placeholder(tf.int32, [None], 'label')]

    def build_graph(self, images, lbls,tower_enabled=True):
        if tower_enabled:
            ctx = get_current_tower_context()
            is_training = ctx.is_training

        is_train = True
        h = self.input_height
        w = self.input_width
        M = 10
        if self.linear_classifier:

            lbls = tf.squeeze(lbls)

            dh = int(h / 2)
            dw = int(w / 2)

            def prediction_incorrect(logits, label, topk=1, name='correct_vector'):
                with tf.name_scope('prediction_correct'):
                    x = tf.nn.in_top_k(logits, label, topk)
                return tf.cast(x, tf.float32, name=name)

            def build_loss(logits, labels,name_step):
                # Cross-entropy loss
                loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)

                # Classification accuracy
                correct_prediction = tf.equal(tf.argmax(logits, 1,output_type=tf.dtypes.int32),labels)
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32),name='accmeanahmed')

                print('{}_acc_top1'.format(name_step)) # This operation is used for logging the val accuracy
                prediction_incorrect(logits, labels, 1, name='{}_acc_top1'.format(name_step))

                return tf.reduce_mean(loss,name='loss_sum'), accuracy

            def Classifier(f,stride_size,reuse=False, scope='Classifier'):
                # f_resized = tf.compat.v1.image.resize_bilinear(f,[stride_size,stride_size])
                f_resized_normalized = tf.compat.v1.layers.AveragePooling2D(stride_size, stride_size,padding='same')(f)
                assert np.prod(f_resized_normalized.shape[1:]) < 10000, "Invalid dim for layer {}".format(f)

                return fc(tf.reshape(f_resized_normalized, [self.batch_size, -1]), self.num_class, is_train,
                          info=not reuse, batch_norm=False, activation_fn=None, name=scope)

            D_x = images
            if self.config.net == 'alexnet':
                features = alexnet(D_x, is_train=not self.config.pretrained,num_classes=self.config.logits_dim,
                                   first_stride=4)
            else:
                raise NotImplementedError('Invalid network name {}'.format(self.config.net))
            output = []
            name = []

            resize_conv = [7,6,4,4,3] # avg_pooling
            for feat_idx,feat in enumerate(features):
                name_step = '_'.join(feat.name.rsplit('/')[1:]).rsplit(':')[0]
                name.append(name_step)
                output.append(Classifier(feat, resize_conv[feat_idx], scope=name_step))

            loss = []
            accuracy = []
            for idx,logits in enumerate(output):
                name_step = '_'.join(logits.name.rsplit('/')[1:]).rsplit(':')[0]
                loss_step, accuray_step = build_loss(logits, lbls,name_step)
                loss.append(loss_step)
                accuracy.append(accuray_step)


            self.loss = tf.reduce_mean(tf.stack(loss),name='loss')
            self.accuracy = tf.reduce_mean(tf.stack(accuracy),name='acc_reduce_mean')
            tf.summary.scalar("acc/acc", self.accuracy)
            self.trainables = [var for var in tf.compat.v1.trainable_variables()]
            print([print(var) for var in tf.compat.v1.trainable_variables()])
            tf.summary.scalar("opt/lr", self.current_lr)
            tf.summary.scalar("loss/loss", self.loss)
            add_moving_summary(self.loss)
            # wd_cost = regularize_cost('.*/weights', l2_regularizer(10**-5),
            #                          name='l2_regularize_loss')
            # tf.summary.scalar("wd_cost", wd_cost)
            # self.loss += wd_cost

            return self.loss
        else:
            image_x, image_y = tf.split(images, 2, axis=0)
            image_x = tf.reshape(image_x, [self.config.batch_size // 2 , self.config.img_height, self.config.img_width, self.config.img_ch])
            image_y = tf.reshape(image_y, [self.config.batch_size // 2 , self.config.img_height, self.config.img_width, self.config.img_ch])

            def build_loss(phi_D_x_tmp, phi_D_y_tmp, phi_T_x_1_tmp, phi_T_x_2_tmp, phi_T_x_3_tmp, phi_T_x_4_tmp,prefix):
                x_tile_distance = tf.reduce_sum((phi_D_x_tmp - (phi_T_x_1_tmp + phi_T_x_2_tmp + phi_T_x_3_tmp + phi_T_x_4_tmp)) ** 2,
                              axis=1)
                y_tile_distance = tf.reduce_sum((phi_D_y_tmp - (phi_T_x_1_tmp + phi_T_x_2_tmp + phi_T_x_3_tmp + phi_T_x_4_tmp)) ** 2,
                              axis=1)
                tf.summary.scalar("tile_distance_{}".format(prefix), tf.reduce_mean(y_tile_distance))
                pair = x_tile_distance
                unpair_raw = M - y_tile_distance

                # condition = tf.less(unpair_raw, 0.)
                unpair = tf.math.maximum(unpair_raw,0.0)
                pr_mean = tf.reduce_mean(pair)
                unpr_mean = tf.reduce_mean(unpair)
                loss = pr_mean + unpr_mean
                return loss, pr_mean, unpr_mean

            dh = int(h / 2)
            dw = int(w / 2)


            def random_resize(image_xy):
                rnd_val = tf.random_uniform([self.config.batch_size // 2], 0, 1.0)

                x_08 = tf.where(tf.less(rnd_val, 0.8), tf.compat.v2.image.resize(image_xy, [dh, dw], method='nearest'),
                                         tf.compat.v2.image.resize(image_xy, [dh, dw], method='bicubic'))

                x_06 = tf.where(tf.less(rnd_val, 0.6), tf.compat.v2.image.resize(image_xy, [dh, dw], method='area'), x_08)
                x_04 = tf.where(tf.less(rnd_val, 0.4), tf.compat.v2.image.resize(image_xy, [dh, dw], method='lanczos5'), x_06)
                return tf.where(tf.less(rnd_val, 0.2), tf.compat.v2.image.resize(image_xy, [dh, dw], method='bilinear'), x_04)

            D_x = random_resize(image_x)
            D_y = random_resize(image_y)



            T_x_1 = image_x[:, :dh, :dw, :]
            T_x_2 = image_x[:, dh:, :dw, :]
            T_x_3 = image_x[:, :dh, dw:, :]
            T_x_4 = image_x[:, dh:, dw:, :]

            T_y_1 = image_y[:, :dh, :dw, :]
            T_y_2 = image_y[:, dh:, :dw, :]
            T_y_3 = image_y[:, :dh, dw:, :]
            T_y_4 = image_y[:, dh:, dw:, :]

            input = [D_x, D_y, T_x_1, T_x_2, T_x_3, T_x_4,T_y_1, T_y_2, T_y_3, T_y_4]
            output = []
            for t in range(len(input)):
                if self.config.net == 'alexnet':
                    output.append(alexnet(input[t], is_train=True,num_classes=self.config.logits_dim,
                                          first_stride=4)[-1])
                else:
                    raise NotImplementedError('Invalid network name {}'.format(self.config.net))

            [phi_D_x, phi_D_y, phi_T_x_1, phi_T_x_2, phi_T_x_3, phi_T_x_4,phi_T_y_1, phi_T_y_2, phi_T_y_3, phi_T_y_4] = output

            loss_x, loss_pr_x, loss_unpr_x = build_loss(phi_D_x, phi_D_y, phi_T_x_1, phi_T_x_2, phi_T_x_3, phi_T_x_4,prefix='x')
            loss_y, loss_pr_y, loss_unpr_y = build_loss(phi_D_y, phi_D_x, phi_T_y_1, phi_T_y_2, phi_T_y_3, phi_T_y_4,prefix='y')

            self.output = output
            self.loss = loss_x + loss_y
            self.loss_pair = loss_pr_x + loss_pr_y
            self.loss_unpair = loss_unpr_x + loss_unpr_y
            add_moving_summary(tf.identity(self.loss,name='loss_sum'))
            tf.summary.scalar("loss/loss", self.loss)
            tf.summary.scalar("loss/pair", self.loss_pair)
            tf.summary.scalar("loss/unpair", self.loss_unpair)
            tf.summary.scalar("opt/lr", self.current_lr)

            tf.summary.scalar("count/D_x", tf.reduce_mean(output[0]))
            tf.summary.scalar("count/D_y", tf.reduce_mean(output[1]))
            tf.summary.scalar("count/T_x_1", tf.reduce_mean(output[2]))
            tf.summary.scalar("count/T_x_2", tf.reduce_mean(output[3]))
            tf.summary.scalar("count/T_x_3", tf.reduce_mean(output[4]))
            tf.summary.scalar("count/T_x_4", tf.reduce_mean(output[5]))

            x_y_distance = tf.reduce_sum((phi_D_x - phi_D_y) ** 2,axis=1)
            tf.summary.scalar("tile_distance_{}".format('x_y'), tf.reduce_mean(x_y_distance))


            self.trainables = [var for var in tf.compat.v1.trainable_variables()]
            print([print(var) for var in tf.compat.v1.trainable_variables()])

            # wd_cost = regularize_cost('.*/weights', l2_regularizer(10 ** -5),
            #                           name='l2_regularize_loss')
            #
            # tf.summary.scalar("wd_cost", wd_cost)
            # self.loss += wd_cost


        return self.loss

    def optimizer(self):
        return self.model_optimizer