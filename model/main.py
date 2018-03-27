from __future__ import division
import os
import numpy as np
import pprint
import tensorflow as tf
import tensorflow.contrib.slim as slim
import pickle, csv

from utils import *
from model import Unet3D

flags = tf.app.flags
flags.DEFINE_integer("epoch",100, "Epoch to train [4]")
flags.DEFINE_string("train_data_dir", "train/", "Directory of the training data [train/]")
flags.DEFINE_boolean("split_train", True, "Whether to split the train data into train and val [False]")
flags.DEFINE_string("lidc_dir", "../lidc_padding_samples/", "Directory of the lidc-idri data [../lidc_padding_samples/]")
flags.DEFINE_string("deploy_data_dir", "../lidc_padding_samples/test/sample/", "Directory of the test data [test/sample/]")
flags.DEFINE_string("deploy_output_dir", "../lidc_padding_samples/test/result/", "Directory of the result in test [test/result/]")
flags.DEFINE_integer("batch_size", 1, "Batch size [1]")
flags.DEFINE_integer("seg_features_root", 48, "Number of features in the first filter in 3D-Unet [48]")
flags.DEFINE_integer("conv_size", 3, "Convolution kernel size in encoding and decoding paths [3]")
flags.DEFINE_integer("layers", 3, "Encoding and decoding layers number [3]")
flags.DEFINE_string("loss_type", "cross_entropy", "Loss type in the model [cross_entropy]")
flags.DEFINE_float("dropout", 0.5, "Drop out ratio [0.5]")
flags.DEFINE_string("checkpoint_dir", "checkpoint/", "Directory name to save the checkpoint [checkpoint/]")
flags.DEFINE_string("log_dir", "logs/", "Directory name to save the log files [logs/]")
flags.DEFINE_boolean("train", True, "True for training, False for deploy [False]")
flags.DEFINE_boolean("run_seg", True, "True if run segmentation [True]")
FLAGS = flags.FLAGS


def main(_):
    pp = pprint.PrettyPrinter()
    pp.pprint(flags.FLAGS.__flags)

    all_paths = os.listdir(os.path.join(FLAGS.lidc_dir, FLAGS.train_data_dir + "sample/"))
    #all_label_path = os.listdir(os.path.join(FLAGS.lidc_dir, FLAGS.train_data_dir + "label/"))

    if FLAGS.split_train:
        np.random.shuffle(all_paths)
        num_training = int(len(all_paths) * 4 / 5)
        train_paths = all_paths[:num_training]
        val_paths = all_paths[num_training:]
    else:
        np.random.shuffle(all_paths)
        train_paths = all_paths

    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)

    if not os.path.exists(FLAGS.log_dir):
        os.makedirs(FLAGS.log_dir)

    if FLAGS.run_seg:
        run_config = tf.ConfigProto()
        with tf.Session(config=run_config) as sess:
            unet = Unet3D(sess, checkpoint_dir=FLAGS.checkpoint_dir, log_dir=FLAGS.log_dir, training_paths=train_paths,
                          testing_paths=val_paths, batch_size=FLAGS.batch_size, layers=FLAGS.layers, 
                          features_root=FLAGS.seg_features_root, conv_size=FLAGS.conv_size, dropout=FLAGS.dropout,
                          loss_type=FLAGS.loss_type)

            if FLAGS.train:
                model_vars =  tf.trainable_variables()
                slim.model_analyzer.analyze_vars(model_vars, print_info=True)

                train_config = {}
                train_config['epoch'] = FLAGS.epoch
                train_config['train_data_dir'] = os.path.join(FLAGS.lidc_dir, FLAGS.train_data_dir)

                unet.train(train_config)

            else:
                #deploy
                if not os.path.exists(FLAGS.deploy_output_dir):
                    os.makedirs(FLAGS.deploy_output_dir)
                unet.deploy(FLAGS.deploy_data_dir, FLAGS.deploy_output_dir)

        tf.reset_default_graph()




if __name__ == '__main__':
    tf.app.run()

