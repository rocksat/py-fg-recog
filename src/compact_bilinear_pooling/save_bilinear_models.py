from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from lib.nets import nets_factory

'''
1, load two pre-trained inception-v3 models into one graph
2, save the graph as into checkpoint
'''
def name_in_checkpoint(prefix, var):
    if prefix in var.op.name:
        _, name = var.op.name.split('/', 1)
    else:
        name = var.op.name
    return name


def load_two_models(image_size, net, num_classes, checkpoint):
    '''
    Load the model n times from checkpoint
    '''
    # Create input placeholder for the networks
    input_placeholder = tf.placeholder(tf.float32, shape=[None, image_size, image_size, 3])

    # Create factory_fn for network
    net_fn = nets_factory.get_network_fn(name=net, num_classes=num_classes)

    # load ops from two networks into single graph
    with tf.variable_scope('model_1'):
        logits_1, end_points_1 = net_fn(input_placeholder)

    with tf.variable_scope('model_2'):
        logits_2, end_points_2 = net_fn(input_placeholder)

    # load models
    variables_to_restore_1 = slim.get_variables_to_restore(exclude=["model_2", "model_1/InceptionV3/AuxLogits"])
    variables_to_restore_1 = {name_in_checkpoint('model_1', var):var for var in variables_to_restore_1}
    restorer_1 = tf.train.Saver(variables_to_restore_1)

    variables_to_restore_2 = slim.get_variables_to_restore(exclude=["model_1", "model_2/InceptionV3/AuxLogits"])
    variables_to_restore_2 = {name_in_checkpoint('model_2', var):var for var in variables_to_restore_2}
    restorer_2 = tf.train.Saver(variables_to_restore_2)

    # Save two subgraph as a graph
    variables_to_save = slim.get_variables_to_restore(exclude=["model_1/InceptionV3/AuxLogits", "model_2/InceptionV3/AuxLogits"])
    saver = tf.train.Saver(variables_to_save)

    with tf.Session() as sess:
        # Restore variables from disk
        restorer_1.restore(sess, checkpoint)
        restorer_2.restore(sess, checkpoint)
        save_path = saver.save(sess, "./data/imagenet_models/bilinear_inception_v3.ckpt")

if __name__ == '__main__':
    load_two_models(299, 'inception_v3', 1001, './data/imagenet_models/inception_v3.ckpt')
