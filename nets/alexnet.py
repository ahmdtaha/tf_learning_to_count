from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import slim as contrib_slim

slim = contrib_slim


def bn_act(input, is_train, batch_norm=True, activation_fn=None):
    _ = input
    if activation_fn is not None:
      _ = activation_fn(_)
    if batch_norm is True:
      _ = tf.contrib.layers.batch_norm(
        _, center=True, scale=True, decay=0.9,
          trainable=is_train,
          is_training=is_train, updates_collections=None
      )
    return _

def my_fc(input, output_shape, trainable=True, info=False, batch_norm=True,
         activation_fn=tf.nn.relu, name="fc"):
    fc_result = slim.fully_connected(input, output_shape, activation_fn=None,trainable=trainable)
    fc_result_bn = bn_act(fc_result, trainable, batch_norm=batch_norm, activation_fn=activation_fn)
    return tf.identity(fc_result_bn, name=name)
def alexnet_v2(inputs,
               is_train,
               enable_batch_norm,
               num_classes=1000,
               first_stride=4,
               is_training=True,
               lrn_enabled=False,
               drop_enabled=False,
               dropout_keep_prob=0.5,
               spatial_squeeze=True,
               scope='alexnet_v2',
               global_pool=False):
  """AlexNet version 2.
  Described in: http://arxiv.org/pdf/1404.5997v2.pdf
  Parameters from:
  github.com/akrizhevsky/cuda-convnet2/blob/master/layers/
  layers-imagenet-1gpu.cfg
  Note: All the fully_connected layers have been transformed to conv2d layers.
        To use in classification mode, resize input to 224x224 or set
        global_pool=True. To use in fully convolutional mode, set
        spatial_squeeze to false.
        The LRN layers have been removed and change the initializers from
        random_normal_initializer to xavier_initializer.
  Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    num_classes: the number of predicted classes. If 0 or None, the logits layer
    is omitted and the input features to the logits layer are returned instead.
    is_training: whether or not the model is being trained.
    dropout_keep_prob: the probability that activations are kept in the dropout
      layers during training.
    spatial_squeeze: whether or not should squeeze the spatial dimensions of the
      logits. Useful to remove unnecessary dimensions for classification.
    scope: Optional scope for the variables.
    global_pool: Optional boolean flag. If True, the input to the classification
      layer is avgpooled to size 1x1, for any input size. (This is not part
      of the original AlexNet.)
  Returns:
    net: the output of the logits layer (if num_classes is a non-zero integer),
      or the non-dropped-out input to the logits layer (if num_classes is 0
      or None).
    end_points: a dict of tensors with intermediate activations.
  """
  with tf.compat.v1.variable_scope(scope, 'alexnet_v2', [inputs],reuse=tf.compat.v1.AUTO_REUSE) as sc:
    end_points_collection = sc.original_name_scope + '_end_points'
    # Collect outputs for conv2d, fully_connected and max_pool2d.
    with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                        outputs_collections=[end_points_collection]):
      if enable_batch_norm:
        batch_norm_fn = slim.batch_norm
      else:
        batch_norm_fn = None

      output_layers =[]


      net = slim.conv2d(inputs, 96, [11, 11], stride=4, padding='SAME',
                        scope='conv1', trainable=is_train, normalizer_fn=batch_norm_fn)
      output_layers.append(net)
      net = slim.max_pool2d(net, [3, 3], 2, scope='pool1',padding='SAME')
      if lrn_enabled:
         net = tf.nn.local_response_normalization(net,depth_radius=5, bias=1, alpha=0.0001, beta=0.75)

      net = slim.conv2d(net, 256, [5, 5], scope='conv2', trainable=is_train, normalizer_fn=batch_norm_fn,padding='SAME')
      output_layers.append(net)
      net = slim.max_pool2d(net, [3, 3], 2, scope='pool2',padding='SAME')
      if lrn_enabled:
         net = tf.nn.local_response_normalization(net,depth_radius=5, bias=1, alpha=0.0001, beta=0.75)


      net = slim.conv2d(net, 384, [3, 3], scope='conv3', trainable=is_train, normalizer_fn=batch_norm_fn,padding='SAME')
      output_layers.append(net)

      net = slim.conv2d(net, 384, [3, 3], scope='conv4', trainable=is_train, normalizer_fn=batch_norm_fn,padding='SAME')
      output_layers.append(net)

      net = slim.conv2d(net, 256, [3, 3], scope='conv5', trainable=is_train, normalizer_fn=batch_norm_fn,padding='SAME')
      output_layers.append(net)
      net = slim.max_pool2d(net, [3, 3], 2, scope='pool5')

      if is_train:

          fc6 = my_fc(tf.reshape(net, [tf.shape(inputs)[0], -1]),
                   4096, trainable=is_train,batch_norm=False,  name='fc_6')
          if drop_enabled:
            fc6 = slim.dropout(fc6, dropout_keep_prob, is_training=is_training,scope='dropout6')
          output_layers.append(fc6)


          fc7 = my_fc(fc6, 4096, trainable=is_train,batch_norm=False,  name='fc_7')
          if drop_enabled:
            fc7 = slim.dropout(fc7, dropout_keep_prob, is_training=is_training,scope='dropout7')
          output_layers.append(fc7)


          fc8 = my_fc(fc7, num_classes, trainable=is_train,batch_norm=False,  name='fc_8')
          output_layers.append(fc8)


      return output_layers
