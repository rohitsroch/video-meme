from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import os.path
from multiprocessing import cpu_count

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import model
import utils

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('ckpt_dir', './train_logs',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('num_epochs', 50,
                            """Number of epochs to run.""")
# tf.app.flags.DEFINE_integer('val_steps', 144,
#                             """Number of batches to run.""")
tf.app.flags.DEFINE_integer('num_gpus', 1,
                            """How many GPUs to use.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_float('lr', 0.001,
                          """learning rate""")
tf.app.flags.DEFINE_float('beta1', 0.9,
                          'beta 1 parameter of Adam optimizer')
tf.app.flags.DEFINE_float('beta2', 0.999,
                          'beta 2 parameter of Adam optimizer')
tf.app.flags.DEFINE_float('epsilon', 1e-8,
                          'epsilon parameter of Adam optimizer')
tf.app.flags.DEFINE_integer('batch_size', 5,
                            """size of val data""")
tf.app.flags.DEFINE_integer('NUM_THREADS', 32,
                            """no of CPU cores to use""")
tf.app.flags.DEFINE_boolean('restore', False, 'restore checkpoint')
tf.app.flags.DEFINE_integer('max_words', 28, 'Max words in a sentence')
tf.app.flags.DEFINE_integer('num_classes', 11, 'num_classes')
tf.app.flags.DEFINE_integer('emb_size', 100, 'size of random embeddings')
tf.app.flags.DEFINE_integer(
    'num_units', 512, 'num of units to use in one lstm cell')

# word ngrams to consider for 1dCNN
ngrams = [1, 2, 3, 4, 5]

# list of train and val_files
train_files = pd.read_csv('./demo_emo_txt_clean.csv')
val_files = pd.read_csv('./demo_emo_txt_clean_val.csv')

print('Processing train files')
train_sentences, train_labels = utils.get_sen_and_labels(train_files)
print('Processing val files')
val_sentences, val_labels = utils.get_sen_and_labels(val_files)

train_size = len(train_sentences)
val_size = len(val_sentences)
print_after = train_size // (FLAGS.num_gpus * FLAGS.batch_size)
val_steps = val_size // (FLAGS.num_gpus * FLAGS.batch_size)
max_steps = FLAGS.num_epochs * print_after

sentences = train_sentences + val_sentences
labels = train_labels + val_labels

word2idx, idx2word = utils.build_dict(sentences, True)
label2idx, idx2label = utils.build_dict(labels, False)
vocabulary_size = len(word2idx)


train_gen = utils.batches_generator(train_size, train_sentences,
                                    train_labels, word2idx,
                                    label2idx)
val_gen = utils.batches_generator(val_size, val_sentences,
                                  val_labels, word2idx,
                                  label2idx)

X_train, y_train = next(train_gen)
X_val, y_val = next(val_gen)

#print(X_train, y_train)

assert X_train.shape[0] == y_train.shape[0], 'train vectors shape mismatch'
assert X_val.shape[0] == y_val.shape[0], 'val vectors shape mismatch'

def tower_loss(scope, embeddings, labels, is_train):
    """Calculate the total loss on a single tower running the model.

    params:
       scope: unique prefix string identifying the CIFAR tower, e.g. 'tower_0'
       embeddings(float32): embeddings of each word in a sentence of shape
                            [batch_size, max_words, emb_size]
       labels: Labels. 1D tensor of shape [batch_size, num_classes].
       is_train: a bool variable which can be toggled during train and test
    """
    is_train = tf.Print(is_train, [is_train], 'Value of is_train is: ')

    # get the embeddings of the sentence using 1d cnns
    logits = model.model_cnn(embeddings, ngrams)
    print('AB',logits.get_shape(), labels.get_shape())
    
   
    # Build the portion of the Graph calculating the losses. Note that we will
    # assemble the total_loss using a custom function below.
    loss, accuracy = model.compute_cross_entropy_loss(logits, labels)

    # Assemble all of the losses for the current tower only.
    losses = tf.get_collection('losses', scope)
    accues = tf.get_collection('accuracy', scope)

    # Calculate the total loss for the current tower.
    total_loss = tf.add_n(losses, name='total_loss')
    total_accuracy = tf.add_n(accues, name='total_accuracy')

    # Attach a scalar summary to all individual losses and the total loss;
    # do the same for the averaged version of the losses.
    # for l in losses + [total_loss]:
    #   # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU
    #   # training session. This helps the clarity of presentation
    #   # on tensorboard.
    #   loss_name = re.sub('%s_[0-9]*/' % 'Tower', '', l.op.name)
    #   tf.summary.scalar(loss_name, l)

    return total_loss, total_accuracy


def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.

    Note that this function provides a synchronization point across all towers.

    params:
           tower_grads: List of lists of (gradient, variable) tuples.
                        The outer list is over individual gradients.
                        The inner list is over the gradient
                        calculation for each tower.

    returns:
                 List of pairs of (gradient, variable) where the
                 gradient has been averagedacross all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        # print(grad_and_vars)
        for g, v in grad_and_vars:
            # print(g, v)
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def input_fn():
    """
    Function to get train and test dataset objects

    Returns:
            _it: iterator to get train and test batch after every batch
            train_init_op: op to initialise iterator to train dataset
            test_init_op: op to initialise iterator to test dataset
    """
    # from_tensor_slices takes tensors as input and generates dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    # do not shuffle - mixes up sen order
    # train_dataset = train_dataset.shuffle(buffer_size=FLAGS.batch_size)
    # use tf 1.9 for drop_reminder
    train_dataset = train_dataset.batch(FLAGS.batch_size, drop_remainder=True)
    train_dataset = train_dataset.repeat(FLAGS.num_epochs)
    train_dataset = train_dataset.prefetch(FLAGS.num_gpus * FLAGS.batch_size)

    test_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    # tf 1.9 for drop_reminder
    test_dataset = test_dataset.batch(FLAGS.batch_size, drop_remainder=True)
    test_dataset = test_dataset.repeat(FLAGS.num_epochs)
    test_dataset = test_dataset.prefetch(FLAGS.num_gpus * FLAGS.batch_size)

    # create a iterator of the correct shape and type
    _it = tf.data.Iterator.from_structure(train_dataset.output_types,
                                          train_dataset.output_shapes)
    # create the initialisation operations
    train_init_op = _it.make_initializer(train_dataset)
    test_init_op = _it.make_initializer(test_dataset)

    return _it, train_init_op, test_init_op


def train():
    with tf.Graph().as_default(), tf.device('/cpu:0'):

        _it, train_init_op, test_init_op = input_fn()

        global_step = tf.get_variable(
            'global_step', [],
            initializer=tf.constant_initializer(0), trainable=False)

        # Create an optimizer that performs gradient descent.
        opt = tf.train.AdamOptimizer(learning_rate=FLAGS.lr,
                                     beta1=FLAGS.beta1,
                                     beta2=FLAGS.beta2,
                                     epsilon=FLAGS.epsilon)

        # Calculate the gradients for each model tower.
        tower_grads = []
        loss_main = []
        acc_main = []
        # intra_op_paralellism forces tf to use NUM_threads CPU cores
        # ref: https://stackoverflow.com/questions/39395198/
        #                                 configuring-tensorflow-to-use-all-cpus
        config = tf.ConfigProto(allow_soft_placement=True,
                                log_device_placement=False
                                # intra_op_parallelism_threads=cpu_count()
                                )
        # allow_growth lets tf to take up gpu mem as per req
        # config.gpu_options.allow_growth = True

        with tf.variable_scope(tf.get_variable_scope()):
            for i in xrange(FLAGS.num_gpus):
                print('running on gpu {}'.format(i))
                with tf.device('/GPU:%d' % i):
                    with tf.name_scope('%s_%d' % ('Tower', i)) as scope:
                        # Dequeues one batch for the GPU
                        sen_batch, label_batch = _it.get_next()
                        # reshape to [batch_size, seq_len] for calculating loss
                        #label_batch = tf.reshape(label_batch, [-1, FLAGS.num_classes])

                        initial_embedding_matrix = np.random.randn(
                            vocabulary_size,
                            FLAGS.emb_size) / np.sqrt(FLAGS.emb_size)
                        initial_embedding_matrix = \
                            initial_embedding_matrix.astype(
                                'float32')

                        embedding_matrix_variable = tf.get_variable(
                            'embedding_matrix',
                            initializer=initial_embedding_matrix,
                            dtype=tf.float32)
                        print('embedding_matrix size: ',
                              embedding_matrix_variable.get_shape())
                        embeddings = tf.nn.embedding_lookup(
                            embedding_matrix_variable, sen_batch)
                        is_train = tf.get_variable('is_train',
                                                   dtype=tf.bool,
                                                   shape=(),
                                                   trainable=False)

                    # Calculate the loss for one tower of the model.
                    # This function constructs the entire model
                    # but shares the variables across all towers.
                        loss, accuracy = tower_loss(scope,embeddings,label_batch,
                                                            is_train)
                        # Reuse variables for the next tower.
                        tf.get_variable_scope().reuse_variables()

                        # Retain the summaries from the final tower.
                        # summaries = tf.get_collection(tf.GraphKeys.SUMMARIES,
                        #                               scope)
                        loss_main.append(loss)
                        acc_main.append(accuracy)
                        grads = opt.compute_gradients(loss)
                        # grads = _compute_gradients(
                        # loss, tf.trainable_variables())
                        # Keep track of the gradients across all towers.
                        tower_grads.append(grads)

        # We must calculate the mean of each gradient. Note that this is the
        # synchronization point across all towers.
        grads = average_gradients(tower_grads)
        avg_loss = tf.reduce_mean(loss_main, 0)
        avg_acc = tf.reduce_mean(acc_main, 0)

        # Add a summary to track the learning rate.
        # summaries.append(tf.summary.scalar('learning_rate', lr))

        # Add histograms for gradients.
        # for grad, var in grads:
        #   if grad is not None:
        #       summaries.append(tf.summary.histogram(
        #               var.op.name + '/gradients', grad))

        # Apply the gradients to adjust the shared variables.
        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

        # Add histograms for trainable variables.
        # for var in tf.trainable_variables():
        #   summaries.append(tf.summary.histogram(var.op.name, var))

        # Track the moving averages of all trainable variables.
        variable_averages = tf.train.ExponentialMovingAverage(
            0.9999, global_step)
        variables_averages_op = variable_averages.apply(
            tf.trainable_variables())

        # Group all updates to into a single train op.
        train_op = tf.group(apply_gradient_op, variables_averages_op)
        # loss_averages_op = model._add_loss_summaries(loss)
        # Create a saver.
        saver = tf.train.Saver(tf.global_variables())

        # Build the summary operation from the last tower summaries.
        # summary_op = tf.summary.merge(summaries)

        # Build an initialization operation to run below.
        init = [tf.global_variables_initializer(), tf.tables_initializer()]

        # use these assign btw train and test mode
        # this is done to toggle dropout on/off
        train_mode = is_train.assign(True)
        val_mode = is_train.assign(False)

        train_mode_init = tf.group(train_init_op, train_mode)
        val_mode_init = tf.group(test_init_op, val_mode)

        # Start running operations on the Graph. allow_soft_placement must be
        # set to True to build towers on GPU,as some of the ops do not have GPU
        # implementations.
        sess = tf.Session(config=config)
        # set to use train_data
        sess.run([init, train_mode_init])

        step1 = 0
        # summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)
        ckpt = tf.train.get_checkpoint_state(FLAGS.ckpt_dir)
        if ckpt and ckpt.model_checkpoint_path and FLAGS.restore:
            # Restores from checkpoint
            saver.restore(sess, ckpt.model_checkpoint_path)
            # Assuming model_checkpoint_path looks something like:
            #   /my-favorite-path/cifar10_train/model.ckpt-0,
            # extract global_step from it.
            global_step = ckpt.model_checkpoint_path.split(
                '/')[-1].split('-')[-1]
            print (type(global_step))
            step1 = int(global_step)
        loss_sum = 0
        acc_sum = 0
        flag = 1
        print('Starting Training')
        # get inv map for decoding labels
        total_parameters = 0
        for variable in tf.trainable_variables():
            # shape is an array of tf.Dimension
            shape = variable.get_shape()
            print(variable)
        for step in tqdm(xrange(step1, max_steps)):
            # sess.run(_input)
            # print('calculating loss')
            _, loss_value = sess.run([train_op, avg_loss])
            # duration = time.time() - start_time
            # epoch_time += duration
            loss_sum = loss_sum + loss_value
            # print(loss_value.shape)
            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
            if step % print_after == 0 and step > 0:
                # num_examples_per_epoch = FLAGS.train_size /
                # examples_per_sec = num_examples_per_step / duration
                # sec_per_batch = duration / FLAGS.num_gpus
                format_str = ('%s: Epoch %d avg_loss = %.2f')
                train_loss = loss_sum / print_after
                print(format_str % (datetime.now(), flag,
                                    train_loss))
                loss_sum = 0
                acc_sum = 0
                flag = flag + 1
                # switch to val data
                sess.run(val_mode_init)
                for _step in tqdm(range(val_steps)):
                    # select a random batch and print deocded preds for
                    # that batch
                    loss_value, acc_value = sess.run([avg_loss, avg_acc])
                    loss_sum = loss_sum + loss_value
                    acc_sum = acc_sum + acc_value
                format_str = ('%s: avg_val_loss = %.2f avg_acc_val= %.2f')
                val_loss = loss_sum / val_steps
                val_acc =  (acc_sum * 100) / val_steps
                print(format_str % (datetime.now(),
                                    val_loss, val_acc))
                loss_sum = 0
                # start using train data
                sess.run(train_mode_init)

            # Save the model checkpoint periodically.
            if step % print_after == 0 or (step + 1) == max_steps:
                checkpoint_path = os.path.join(FLAGS.ckpt_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)


def main(argv=None):
    train()
    print("Done!")


if __name__ == '__main__':
    tf.app.run()
