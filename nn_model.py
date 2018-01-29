# -*- coding: utf-8 -*-

__version__ = "0.5"

''' 
=== history ===
201703XX(0.1): first draft
20170329(0.2): model version implementation added
               threshold added
20171121(0.3): cfg file option added
20171213(0.4): functionalities for long-time running daemon is added
20171222(0.5): mini batch added
               data set padding moved to preprocessing.py 
'''

import os, sys
import csv
import time
import json
import signal
import datetime
import traceback

import numpy as np
import tensorflow as tf
from scipy.io import wavfile

import nn_model_ver
import preprocessing
import common

from batch import BATCH

g_sess = None
g_bp_ml = None

def sig_handler(signum, frame):
    
    global g_sess, g_bp_ml
    
    if g_sess and g_bp_ml:
        g_bp_ml.save_model_if_exists(g_sess)
        
    sys.exit()
    

class NN_MODEL:

    def __init__( self, prog_name, version=None, save_name=None, restore_path=None, 
                  board_dir=None, cfg_path=None, add_sighandler=True ):
        
        global g_bp_ml
        
        # for sig handling
        g_bp_ml = self
        
        # configuration setting
        if cfg_path:
            with open(cfg_path, 'r') as f:
                cfg_data = json.load(f)
            self.learning_rate = cfg_data['learning_rate']
            self.epoch = cfg_data['epoch']
            self.epoch_start_idx = cfg_data.get('epoch_start_idx', 0)
            self.dropout_prob_val = cfg_data['dropout_prob_val']
            self.tot_data_size = cfg_data.get("tot_data_size", 0)
            self.x_shape = cfg_data['x_shape']
            version = cfg_data['version']
            self.cfg_data = cfg_data
        else:
            self.learning_rate = 0.01
            self.epoch = 20000
            self.epoch_start_idx = 0
            self.dropout_prob_val = 0.5
            self.tot_data_size = 0
            self.x_shape = None
            if not version:
                raise Exception("No version info found")
        
        # parse parameter
        self.prog_name = prog_name
        self.model_version = version
        self.get_nn_name = "get_{}".format(self.model_version)
        self.preprocessing_name = "preprocess_{}".format(self.model_version)
        if save_name:
            self.save_name = save_name
        else:
            self.save_name = self.model_version
        self.restore_path = restore_path
        self.board_dir = board_dir
        self.board_on = board_dir is not None
        
        # create model check point directory
        self.model_checkpoint_dir_path = os.path.abspath("model_checkpoint")
        if not os.path.exists(self.model_checkpoint_dir_path):
            os.system('mkdir ' + self.model_checkpoint_dir_path)
        
        
        # set summary board
        if self.board_on:
            self.summary_path = os.path.abspath(self.board_dir)
            if not os.path.exists(self.summary_path):
                os.mkdir(self.summary_path, 0o755)
            print("[{}] summary dir path is {}".format(prog_name, self.summary_path))
        
        # get nn model function
        if self.get_nn_name not in dir(nn_model_ver):
            raise Exception("[{}] no matching model version is found : {}".format(self.prog_name, self.model_version))
        self.get_nn = getattr(nn_model_ver, self.get_nn_name)
        
        # get preprocessing function
        if self.preprocessing_name not in dir(preprocessing):
            raise Exception("[{}] no matching preprocessing version is found : {}".format(self.prog_name, self.model_version))
        self.preprocess = getattr(preprocessing, self.preprocessing_name)
        
        # set signal handler
        if add_sighandler:
            print("[nn_model] setting signal handler")
            signal.signal(signal.SIGINT, sig_handler)
            signal.signal(signal.SIGQUIT, sig_handler)
            signal.signal(signal.SIGTERM, sig_handler)
            signal.signal(signal.SIGALRM, sig_handler)
    
    
    def __del__(self):
        if 'sess' in dir(self) and self.sess:
            self.sess.close()
        
        
    def __enter__(self):
        pass
        
        
    def __exit__(self, exc_type, exc_value, traceback):
        if 'sess' in dir(self) and self.sess:
            self.sess.close()

        
    def save_model_if_exists(self, sess):
        
        if self.save_name is None:
            return
        
        now_datetime_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_name = "{}_{}.ckpt".format(self.save_name, now_datetime_str)
        save_path = os.path.join(self.model_checkpoint_dir_path, save_name)
        
        saver = tf.train.Saver()
        saver.save(sess, save_path)
        print("[{}] file saved : {}".format(self.prog_name, save_path))
        
        
    def restore_model_if_exists(self, sess):
        
        if self.restore_path is None:
            return
        
        restore_path = os.path.abspath(self.restore_path)
        saver = tf.train.Saver()
        try:
            saver.restore(sess, restore_path)
            print("[{}] session restored : {}".format(self.prog_name, restore_path))
        except:
            print("[{}] Error restoring old model {}".format(self.prog_name, restore_path))
        
        
    """
    trains data
    Parameters:
    - data_dir: data directory path
    - split_ratio: split ratio of training, testing, validation data.
      validation data ratio could be 0.
      separator of ratio is ':'
      ex) 7:3, 3:1:1
    """
    def train(self, data_dir=None, split_ratio="7:3"):
        
        global g_sess
        
        if not data_dir or not os.path.exists(data_dir):
            print("[{}] data directory doesn't exist : {}".format(self.prog_name, data_dir))
            sys.exit()
        
        train_data_list, test_data_list, valid_data_list = common.process_input_files(data_dir, split_ratio, limit=self.tot_data_size)
        #train_data_list, test_data_list, valid_data_list = common.process_input_files(data_dir, split_ratio)
        
        x_train_path, y_train_path = common.preprocess_data_to_file_path_label_set(train_data_list)
        x_test_path, y_test_path = common.preprocess_data_to_file_path_label_set(test_data_list)
        x_valid_path, y_valid_path = common.preprocess_data_to_file_path_label_set(valid_data_list)

        mini_batch = self.cfg_data['train_set'].get('mini_batch', False)
        preload = self.cfg_data['train_set'].get('preload', False)
        batch_size = self.cfg_data['train_set'].get('batch_size', 0)
        train_batch = BATCH( mini_batch, preload, batch_size, 
                               None, x_train_path, y_train_path, 
                               self.preprocess, self.x_shape )

        mini_batch = self.cfg_data['test_set'].get('mini_batch', False)
        preload = self.cfg_data['test_set'].get('preload', False)
        batch_size = self.cfg_data['test_set'].get('batch_size', 0)
        test_batch = BATCH( mini_batch, preload, batch_size, 
                               None, x_test_path, y_test_path, 
                               self.preprocess, self.x_shape )

        mini_batch = self.cfg_data.get('validation_set', {}).get('mini_batch', False)
        preload = self.cfg_data.get('validation_set', {}).get('preload', False)
        batch_size = self.cfg_data.get('validation_set', {}).get('batch_size', 0)
        valid_batch = BATCH( mini_batch, preload, batch_size, 
                               None, x_valid_path, y_valid_path, 
                               self.preprocess, self.x_shape )

        # print("{} {} {} {}".format(len(x_train_path), len(x_test_path), len(x_valid_path), len(x_train_path) + len(x_test_path) + len(x_valid_path)))
        # sys.exit()

        # for x_list, y_list in train_batch.batch():
        #     print("{}".format(len(x_list)))
        #     print os.popen("free -h").read()
        # sys.exit()

        # train here
        hypothesis, train, cost, keep_prob, X, Y = self.get_nn(self.x_shape, self.learning_rate)

        # Test model
        correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
        # Calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
        init = tf.global_variables_initializer()
        
        # with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True
        with tf.Session(config=config) as sess:
            sess.run(init)
            
            # for sig handling
            g_sess = sess
            
            # Create summary writer
            if self.board_on:
                merged = tf.summary.merge_all()
                writer = tf.summary.FileWriter(self.summary_path, sess.graph)
                accuracy_var = tf.placeholder(tf.float32, shape=())
                # accuracy_summary = tf.summary.scalar("accuracy", accuracy)
                train_accuracy_summary = tf.summary.scalar("trainset_accuracy", accuracy_var)
                valid_accuracy_summary = tf.summary.scalar("validationset_accuracy", accuracy_var)
            
            self.restore_model_if_exists(sess)
            
            old_train_accuracy = 0.0
            old_valid_accuracy = 0.0
            
            tot_train_batch_size = train_batch.total_batch_size()
            tot_valid_batch_size = valid_batch.total_batch_size()
            tot_test_batch_size = test_batch.total_batch_size()
            
            tot_batch_size = max([tot_train_batch_size, tot_valid_batch_size, tot_test_batch_size])
            print("tot_batch_size {}".format(tot_batch_size))

            for epoch_idx in range(self.epoch_start_idx, self.epoch):

                print("[{}] epoch {}".format(datetime.datetime.now(), epoch_idx))

                for x_list, y_list in train_batch.batch():
                    start_t = time.time()
                    print("process mini batch. elapsed_t {}".format(time.time() - start_t))
                    start_t = time.time()
                    sess.run(train, feed_dict={X: x_list, Y: y_list, keep_prob: self.dropout_prob_val})
                    print("train mini batch success. elapsed_t {}".format(time.time() - start_t))


                if epoch_idx and epoch_idx % 100 == 0:

                    a_list = []
                    c_list = []
                    
                    i = 0
                    for x_list, y_list in train_batch.batch():
                        if self.board_on:
                            summary, a = sess.run([merged, accuracy], feed_dict={X: x_list, Y: y_list, keep_prob: 1})
                            writer.add_summary(summary, global_step = epoch_idx + i)
                            i += 1
                        else:
                            c, a = sess.run([cost, accuracy], feed_dict={X: x_list, Y: y_list, keep_prob: 1})
                            c_list.append(c)
                        a_list.append(a)
                    accuracy_val = np.mean(a_list)

                    if self.board_on:
                        print("count:{} accyracy:{}".format(epoch_idx, accuracy_val))
                    else:
                        cost_val = np.mean(c_list)
                        print("count:{} cost:{} accyracy:{}".format(epoch_idx, cost_val, accuracy_val))

                    if self.board_on:
                        summary = sess.run(train_accuracy_summary, feed_dict={accuracy_var: accuracy_val})
                        writer.add_summary(summary, global_step = epoch_idx)
                    if accuracy_val > old_train_accuracy:
                        old_train_accuracy = accuracy_val
                        self.save_model_if_exists(sess)
                        
                    if valid_data_list:
                        # run validation set accuracy test
                        a_list = []
                        for x_list, y_list in valid_batch.batch():
                            a = sess.run(accuracy, feed_dict={X: x_list, Y: y_list, keep_prob: 1})
                            a_list.append(a)
                        accuracy_val = np.mean(a_list)  
                        if self.board_on:
                            summary = sess.run(valid_accuracy_summary, feed_dict={accuracy_var: accuracy_val})
                            writer.add_summary(summary, global_step = epoch_idx)
                        if accuracy_val > old_valid_accuracy:
                            old_valid_accuracy = accuracy_val
                            self.save_model_if_exists(sess)

            print("[{}] Optimization Finished!".format(self.prog_name))
            
            train_batch.clear()
            valid_batch.clear()
            
            self.save_model_if_exists(sess)

            a_list = []
            for x_list, y_list in test_batch.batch():
                a = sess.run(accuracy, feed_dict={X: x_list, Y: y_list, keep_prob: 1})
                a_list.append(a)
            accuracy_val = np.mean(a_list)
            print("test accuracy: {}".format(accuracy_val))



    def scan(self, file_path_list):
        
        global g_sess

        blank_labal_list = [0] * len(file_path_list)

        mini_batch = self.cfg_data['scan_set'].get('mini_batch', False)
        preload = self.cfg_data['scan_set'].get('preload', False)
        batch_size = self.cfg_data['scan_set'].get('batch_size', 0)
        scan_batch = BATCH( mini_batch, preload, batch_size, 
                               None, file_path_list, blank_labal_list, 
                               self.preprocess, self.x_shape )
            
        # model
        hypothesis, _, _, keep_prob, X, _ = self.get_nn(self.x_shape, self.learning_rate)
        
        init = tf.global_variables_initializer()
        
        out_predict_list = []
        with tf.Session() as sess:
            sess.run(init)
            
            # for sig handling
            g_sess = sess
            
            self.restore_model_if_exists(sess)

            for x_list, y_list in scan_batch.batch():
                predict_list = sess.run(hypothesis, feed_dict={X: x_list, keep_prob: 1})
                out_predict_list.extend(predict_list)

        return out_predict_list



       
    def scan_for_submission(self, data_dir=None, ori_submission_path=None, out_submission_path=None):
        
        global g_sess
        
        if not data_dir or not os.path.exists(data_dir):
            print("[{}] data directory doesn't exist : {}".format(self.prog_name, data_dir))
            sys.exit()

        # load file list
        tmp_file_list = os.listdir(data_dir)
        file_name_list = [ file_name for file_name in tmp_file_list if file_name.endswith('.wav') ]
        file_path_list = [ os.path.join(data_dir, file_name) for file_name in tmp_file_list if file_name.endswith('.wav') ]
        data_set = {}
        for file_name in file_name_list:
            data_set[file_name] = None

        out_predict_list = self.scan(file_path_list)

        for idx, predict in enumerate(out_predict_list):
            file_name = file_name_list[idx]
            data_set[file_name] = common.one_hot_to_label(list(predict))
            #print predict.shape, predict, data_set[file_name]

        with open(ori_submission_path, 'r') as r_f:
            with open(out_submission_path, 'w') as w_f:
                w_f.write("fname,label\n")
                for line in r_f:
                    line = line.strip()
                    if line == "fname,label":
                        continue
                    file_name = line.split(',')[0]
                    w_f.write("{},{}\n".format(file_name, data_set[file_name]))








if __name__ == "__main__":

    prog_name = "bp_ml"
    save_name = "cnn_mfcc_noise"
    restore_path = None
    board_dir = "summary/cnn_mfcc_noise"
    #board_dir = None
    cfg_path = "cfg/cnn_mfcc_noise.cfg"
    dir_path = "../dataset/train/audio"
    scan_dir_path = "../dataset/test/audio"
    ori_submission_path = "../dataset/sample_submission.csv"
    out_submission_path = "../dataset/sample_submission_{}.csv".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    nn = NN_MODEL(prog_name, save_name=save_name, restore_path=restore_path, board_dir=board_dir, cfg_path=cfg_path)
    nn.train(dir_path, "4:1:1")
    #nn.scan_for_submission(data_dir=scan_dir_path, ori_submission_path=ori_submission_path, out_submission_path=out_submission_path)










