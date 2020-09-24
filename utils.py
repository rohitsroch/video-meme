from __future__ import print_function
import os
from tqdm import tqdm
from collections import defaultdict
import numpy as np
import string
import tensorflow as tf
import json
from os.path import join, dirname
FLAGS = tf.app.flags.FLAGS


def get_txt_files(directory):
    """
        Function to get all text files from a directory

        params:
            directory(string): path of directory

        returns:
                txt_files: list of all txt_file paths
    """

    txt_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.txt'):
                txt_files.append(root + file)
    return txt_files


def get_sen_and_labels(df):
    """
        Function to get all sentences, labels 
    """
    sentences = []
    labels = []

    for txt,lab in zip(df.text,df.label):
        sentences.append(txt.split())
        labels.append([lab]) 

    return sentences, labels

def build_dict(sen_or_labels, include_pad=True):
    """
            Function to construct dicts for word 2 idx and also return reverse
            maps

            params:
                sen_or_labels(string): list of lists of sentences or labels
                include_pad(bool): whether <pad> token should be added to dict

            returns:
                    word2idx(dict) and its reverse map
    """

    # Create a dictionary with default value 0
    word2idx = defaultdict(lambda: 0)
    idx = 0
    if include_pad:
        pad_token = '<pad>'
        word2idx[pad_token] = idx
        idx += 1
    for word_list in sen_or_labels:
        for word in word_list:
            if word in word2idx:
                continue
            word2idx[word] = idx
            idx += 1
    # py3
    idx2word = {v: k for k, v in word2idx.items()}
    # py2
    # idx2word = {v: k for k, v in word2idx.iter_items()}
    return word2idx, idx2word


def sen2idxs(sentences, word2idx):
    # converts string sentences to list of word ids
    return [word2idx[word] for word in sentences]


def lab2idxs(label, label2idx):
    # converts string labels to list of word ids
    one_hot= [0]*FLAGS.num_classes
    one_hot[ label2idx[label[0]] ]=1
    return one_hot


def batches_generator(batch_size, sentences, labels,
                      word2idx, label2idx,
                      shuffle=False, allow_smaller_last_batch=False,
                      get_lengths=False):
    """
            Generates padded batches of sen and labels.

            params:
                batch_size(int): batch_size of sen or labels to be returned
                sentences(string): list of lists
                labels(string): list of lists
                shuffle(bool): whether data should be shuffled
                allow_smaller_batch(bool): self explanatory
                get_lengths(bool): whether seq_lengths of each sen to be
                                   returned

            returns:
                    A generator obj which can yield a sen and a label batch
    """
    n_samples = len(sentences)
    if shuffle:
        order = np.random.permutation(n_samples)
    else:
        order = np.arange(n_samples)

    n_batches = n_samples // batch_size
    if allow_smaller_last_batch and n_samples % batch_size:
        n_batches += 1
   
    for k in range(n_batches):
        batch_start = k * batch_size
        batch_end = min((k + 1) * batch_size, n_samples)
        current_batch_size = batch_end - batch_start
        x_list = []
        y_list = []
        seq_lengths = []
        max_len_sentence = 0
        for idx in order[batch_start: batch_end]:
            if len(sentences[idx]) > FLAGS.max_words:
                sentences[idx] = sentences[idx][:FLAGS.max_words]
            seq_lengths.append(len(sentences[idx]))
            x_list.append(sen2idxs(sentences[idx], word2idx))
            y_list.append(lab2idxs(labels[idx], label2idx))
            max_len_sentence = max(max_len_sentence, len(sentences[idx]))
        # Fill in the data into numpy nd-arrays filled with padding indices.
        x = np.ones([current_batch_size, FLAGS.max_words],
                    dtype=np.int32) * word2idx['<pad>']
        # lengths = np.zeros(current_batch_size, dtype=np.int32)
        for n in range(current_batch_size):
            # print(x.shape, x_list[n])
            utt_len = len(x_list[n])
            x[n, :utt_len] = x_list[n]
            # lengths[n] = utt_len
        # yield x, np.array(y_list), lengths
        # print(y_list)
        if get_lengths is True:
            yield x, np.array(y_list), np.asarray(seq_lengths,
                                                  dtype=np.float64)
        else:
            yield x, np.array(y_list)



    


   
