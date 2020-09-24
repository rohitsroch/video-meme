from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
FLAGS = tf.app.flags.FLAGS


def conv_block(scope, x, kernel_size, i):
    """
            Function to run a conv1d block on the sentence input

            params:
                scope(string)
                x(float32): sentence input [batch_size, max_words, emb_size]
                kernel_size(int): n_gram size
                i(int): multiplier i.e. unit_filter * i will be num_filters

            returns:
                    tensor after maxpooling [batch_size, 1, num_filters]
    """
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        conv = tf.layers.conv1d(x, filters=100,
                                kernel_size=kernel_size, padding='same',
                                activation=tf.nn.relu
                                # kernel_initializer=tf.contrib.layers.xavier_initializer()
                                )
        pool = tf.layers.max_pooling1d(
            conv, pool_size=FLAGS.max_words, strides=1, padding='valid')
    return pool


def model_cnn(x, ngrams, dropout=0.5):
    """
            Function to run a series of conv1d block on the sentence input
            and concat the outputs

            params:
                x(float32): sentence input [batch_size, max_words, emb_size]
                ngrams(int32): list of word ngrams to consider

            returns:
                    logits
    """
    print('embeddings shape ', x.get_shape())
    with tf.variable_scope('Conv1D'):
        blocks = []
        for i, ngram in enumerate(ngrams):
            block = conv_block('Conv1D_A_' + str(ngram),
                               x, ngram, i + 1)
            blocks.append(block)
        flatten = tf.concat(blocks, axis=-1)
        flatten = tf.reshape(flatten, [-1,(100 * len(ngrams) )])
        with tf.variable_scope('DENSE_A'):
            x=tf.layers.dense(flatten,units=(100 * len(ngrams)),activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
        x = tf.nn.dropout(x, keep_prob=1 - dropout)
        with tf.variable_scope('DENSE_B'):
            x=tf.layers.dense(x,units=(100),activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
        with tf.variable_scope('Sigmoid_Layer_A'):
            x=tf.layers.dense(x,units=FLAGS.num_classes,kernel_initializer=tf.contrib.layers.xavier_initializer())
    return x

def compute_cross_entropy_loss(logits, labels):
    """
      Cross Entropy loss

    """
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels,logits=logits)
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy') 

    correct = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))


    tf.add_to_collection('losses', cross_entropy_mean)
    tf.add_to_collection('accuracy', accuracy)
    
    return cross_entropy_mean, accuracy