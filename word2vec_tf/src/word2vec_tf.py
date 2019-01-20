from word2vec_tf.utils import data, weights
import argparse
import sys
import os
import time
import logging
import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

tf.logging.set_verbosity(tf.logging.INFO)
tf.set_random_seed(123)

parser = argparse.ArgumentParser()
parser.add_argument('--train_file', type=str, help='The training file containing the data.')
parser.add_argument('--log_file', type=str, help='The log file.')
parser.add_argument('--dict_file', type=str, help='The file to store the dictionary.')
parser.add_argument('--w1_file', type=str, help='The file to store the final w1 matrix.')
parser.add_argument('--w2_file', type=str, help='The file to store the final w2 matrix.')
parser.add_argument('--x_file', type=str, help='The file to store the x matrix for training.')
parser.add_argument('--y_file', type=str, help='The file to store the y matrix for training.')
parser.add_argument('--yneg_file', type=str, help='The file to store the yneg matrix for training.')
parser.add_argument('--vocab_size', type=int, default=50000, help='The vocabulary size.')
parser.add_argument('--emb_size', type=int, default=300, help='The size of the embeddings.')
parser.add_argument('--epochs', type=int, default=5, help='The number of epochs to train on.')
parser.add_argument('--num_neg_samples', type=int, default=5, help='The number of negative samples to use.')
parser.add_argument('--context_window', type=int, default=20, help='The context window size (on each side).')
parser.add_argument('--batch_size', type=int, default=1, help='number of training samples in a batch.')
parser.add_argument('--ns_param', type=float, default=0.75, help='The negative sampling parameter.')
parser.add_argument('--ds_param', type=float, default=0.001, help='The down-sampling parameter.')
parser.add_argument('--start_learning_rate', type=float, default=0.05, help='The starting learning rate.')
parser.add_argument('--decay_rate', type=float, default=0.5,
                    help='The decay rate, if doing our own adaptive learning.')
parser.add_argument('--use_w2v_weights', action='store_true', help='Use word2vec generated weights and dictionary.')
parser.add_argument('--load_data', action='store_true', help='Load pre-generated data.')
parser.add_argument('--w2v_dict_file', type=str, help='File where the word2vec dictionary is stored.')
parser.add_argument('--w2v_w1_file', type=str, help='File where the word2vec weights are stored.')
parser.add_argument('--log_dir', type=str, help='The directory where to store all the logs')


def main():
    args = parser.parse_args()
    np.random.seed(1)
    logger = setup_logger(args)
    logger.info('Processing data.')
    start_time = time.time()

    # Get a list of the sentences from the corpus
    sentences = data.get_sentences_from_file(args.train_file)
    # Get the list of the words from the corpus
    words = data.get_words_from_sentences(sentences)
    # Get the total number of words in the corpus
    num_total_words = len(words)
    # Get the dictionaries from the corpus
    dictionaries = data.get_dictionaries(words, args)
    # Get the vocabulary size
    vocab_size = len(dictionaries['word_count'])
    # Get the sentences with words replaced by their dictionary indices
    sentences, n_training_samples = data.get_indexed_sentences(sentences, dictionaries, downsample=True)

    elapsed_time = time.time() - start_time
    logger.info('Data processed in %d seconds' % elapsed_time)

    # Log the parameters for the run.
    logger.debug('Train file                   : %s' % args.train_file)
    logger.debug('Log file                     : %s' % args.log_file)
    logger.debug('W1 file                      : %s' % args.w1_file)
    logger.debug('W2 file                      : %s' % args.w2_file)
    logger.debug('x file                       : %s' % args.x_file)
    logger.debug('y file                       : %s' % args.y_file)
    logger.debug('yneg file                    : %s' % args.yneg_file)
    logger.debug('Vocabulary size              : %d' % vocab_size)
    logger.debug('Total number of sentences    : %d' % len(sentences))
    logger.debug('Total number of words        : %d' % num_total_words)
    logger.debug('Embedding size               : %d' % args.emb_size)
    logger.debug('Batch size                   : %d' % args.batch_size)
    logger.debug('Number of epochs             : %d' % args.epochs)
    logger.debug('Number of training examples  : %d' % n_training_samples)
    logger.debug('Context words                : %d' % args.context_window)
    logger.debug('Negative samples             : %d' % args.num_neg_samples)
    logger.debug('Initial learning rate        : %f' % args.start_learning_rate)
    logger.debug('Decay rate                   : %f' % args.decay_rate)
    logger.debug('Negative sampling param      : %f' % args.ns_param)
    logger.debug('Down sampling param          : %f' % args.ds_param)

    # Define the TensorFlow graph starting with placeholders.
    with tf.name_scope('context'):
        # Instead of one-hot we feed directly the indices of the context words.
        context_words = tf.placeholder(shape=[args.batch_size, 2 * args.context_window],
                                       dtype=tf.int32,
                                       name='context')

    with tf.name_scope('target'):
        # Define the target. This will be the index of the target words
        target_words = tf.placeholder(shape=[args.batch_size, 1],
                                      dtype=tf.int32,
                                      name='positive')

        # Define the negative_samples
        negative_words = tf.placeholder(shape=[args.batch_size, args.num_neg_samples],
                                        dtype=tf.int32,
                                        name='negative')

        # For convenience we concatenate target words and negative words
        Y = tf.concat((target_words, negative_words), axis=1, name='target')

    with tf.name_scope('weights'):
        # Initialize the weights
        W1, W2 = weights.init_tf(vocab_size, args)

    with tf.name_scope('non_zero_mask'):
        # Create boolean array of non-zero elements in context_words.
        non_zero_bool_mask = tf.not_equal(context_words, 0, name='non_zero_bool_mask')
        # Cast it to 1's and 0's
        non_zero_mask = tf.cast(non_zero_bool_mask, tf.float32, name='non_zero_mask')
        # Compute number of non-zero elements for each training example in the batch.
        num_non_zero = tf.reduce_sum(non_zero_mask, axis=1, keepdims=True, name='num_non_zero')

    with tf.name_scope('hidden'):
        # Propagate to the hidden layer by gathering the corresponding embeddings from the embedding matrix.
        H = tf.nn.embedding_lookup(W1, context_words, name='H')
        assert H.get_shape().as_list() == [args.batch_size, 2 * args.context_window, args.emb_size]

        # Expand the non_zero_mask
        non_zero_mask_x = tf.expand_dims(non_zero_mask, axis=2, name='non_zero_mask_x')
        assert non_zero_mask_x.get_shape().as_list() == [args.batch_size, 2 * args.context_window, 1]

        # Make sure the rows corresponding to the PAD symbols are all zeroed before averaging them
        HZ = tf.multiply(H, non_zero_mask_x, name='HZ')
        assert HZ.get_shape().as_list() == [args.batch_size, 2 * args.context_window, args.emb_size]

        # Sum up all the context word embeddings for each training example
        HSUM = tf.reduce_sum(HZ, axis=1, name='HSUM')
        assert HSUM.get_shape().as_list() == [args.batch_size, args.emb_size]

        # Divide by the number of context words for each training example to compute the average.
        HAVG = tf.divide(HSUM, tf.cast(num_non_zero, dtype=tf.float32), name='HAVG')
        assert HAVG.get_shape().as_list() == [args.batch_size, args.emb_size]

        # Expand the dimensions for later
        HX = tf.expand_dims(HAVG, axis=1, name='HX')
        assert HX.get_shape().as_list() == [args.batch_size, 1, args.emb_size]

    with tf.name_scope('Output'):
        # Gather the embeddings for positive and negative target words from the second embedding matrix
        W2R = tf.nn.embedding_lookup(W2, Y, name='W2R')
        assert W2R.get_shape().as_list() == [args.batch_size, args.num_neg_samples + 1, args.emb_size]

        # Multiply W2R [batch_size, neg_samples + 1, emb_size] element-wise with HR [batch_size, 1, emb_size]
        UR = tf.multiply(W2R, HX, name='UR')
        assert UR.get_shape().as_list() == [args.batch_size, args.num_neg_samples + 1, args.emb_size]

        # And sum over the second axis (this effectively computes the dot product)
        U = tf.reduce_sum(UR, axis=2)
        assert U.get_shape().as_list() == [args.batch_size, args.num_neg_samples + 1]

    with tf.name_scope('Loss'):
        # Define the negative sampling loss
        neg_sampling_loss = tf.reduce_mean(-tf.log(tf.sigmoid(U[:, 0])) -
                                           tf.reduce_sum(tf.log(tf.sigmoid(-U[:, 1:])), axis=1), axis=0, name='loss')

        # Alternatively we can use nce_loss
        nce_bias = tf.Variable(tf.zeros(vocab_size - 1))
        nce_loss = tf.reduce_mean(tf.nn.nce_loss(weights=W2,
                                                 biases=nce_bias,
                                                 labels=target_words,
                                                 inputs=HAVG,
                                                 num_sampled=args.num_neg_samples,
                                                 num_classes=vocab_size - 1))

        tf.summary.scalar('neg_loss_summary', neg_sampling_loss)
        # tf.summary.scalar('nce_loss_summary', nce_loss)

    with tf.name_scope('Train'):
        # Define the training step
        train_step = tf.train.AdamOptimizer(0.001).minimize(neg_sampling_loss)

    with tf.Session() as sess:
        # Merge all the summary ops into one.
        merged_summary = tf.summary.merge_all()
        # Define the writer for the summaries.
        writer = tf.summary.FileWriter(args.log_dir, sess.graph)
        # Define a saver for persisting the variables to checkpoints
        saver = tf.train.Saver()
        # Initialize the variables
        sess.run(tf.global_variables_initializer())

        # Write corresponding labels for the embeddings.
        with open(os.path.join(args.log_dir, 'metadata.tsv'), 'w') as f:
            reversed_dictionary = dictionaries['reversed_dictionary']
            for i in range(vocab_size + 1):
                f.write(reversed_dictionary[i] + '\n')

        # Create a configuration for visualizing embeddings with the labels in TensorBoard.
        config = projector.ProjectorConfig()
        embedding_conf = config.embeddings.add()
        embedding_conf.tensor_name = W1.name
        embedding_conf.metadata_path = 'metadata.tsv'

        start_time = time.time()
        for n_epoch in range(0, args.epochs):
            epoch_cost = 0
            for ind in range(n_training_samples // args.batch_size):
                # Retrieve a training batch
                X, YT, YNEG = data.get_training_batch(sentences, dictionaries, args)
                feed_dict = {context_words: X,
                             target_words: YT,
                             negative_words: YNEG}

                cost, _ = sess.run([neg_sampling_loss, train_step], feed_dict=feed_dict)
                epoch_cost += cost

                if ind % 10 == 0:
                    # Every 10-th step write the summary
                    summary = sess.run(merged_summary, feed_dict=feed_dict)
                    writer.add_summary(summary, ind)
                elif ind % 99 == 0:
                    # Every 99-th step, write the summary and the metadata to the writer.
                    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()
                    summary = sess.run(merged_summary,
                                       feed_dict=feed_dict,
                                       options=run_options,
                                       run_metadata=run_metadata)
                    writer.add_run_metadata(run_metadata, 'ep%dstep%d' % (n_epoch, ind))
                    writer.add_summary(summary, ind)
                    # Save the model in the checkpoint
                    saver.save(sess, os.path.join(args.log_dir, "model.ckpt"), ind)
                    projector.visualize_embeddings(writer, config)
                    logger.info('Epoch %d/%d: Training examples processed: %.3f%%'
                                % (n_epoch, args.epochs, (float(ind * 100) / n_training_samples)))

            # Log the average cost per epoch, every epoch
            logger.info('At epoch %d cost is %f' %
                        (n_epoch, epoch_cost / n_training_samples))

        final_W1 = sess.run(W1)
        final_W2 = sess.run(W2)

        np.save(args.w1_file, final_W1)
        np.save(args.w2_file, final_W2)
        writer.close()
        elapsed_time = time.time() - start_time
        logger.info('Optimization time for %d epochs was %d seconds' % (args.epochs, elapsed_time))


def setup_logger(args):
    logger = logging.getLogger('main')
    logger.setLevel(logging.DEBUG)

    # Create handlers. We output to the file everything up to DEBUG.
    file_handler = logging.FileHandler(args.log_file)
    file_handler.setLevel(logging.DEBUG)
    # Stdout gets everything up to DEBUG.
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.DEBUG)

    # Create logging formats
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    stdout_formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    stdout_handler.setFormatter(stdout_formatter)

    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(stdout_handler)

    return logger


main()
