# -*- coding: utf-8 -*-

__version__ = "0.2"

''' 
=== history ===
20171220(0.1): first draft
20171222(0.2): mini batch 
'''

import os, sys
import csv
import json
import signal
import datetime
import traceback

import numpy as np
import tensorflow as tf
from scipy.io import wavfile


label_list = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go', 'silence', 'unknown']

label_len = len(label_list)

label_idx = {}
for idx, label in enumerate(label_list):
    label_idx[label] = idx

    
"""
processes input files into data

Parameters:
- data_dir: data directory path
- split_ratio: split ratio of training, testing, validation data.
  validation data ratio could be 0.
  separator of ratio is ':'
  ex) 7:3, 3:1:1
- limit: limits data count. disabled if 0 is passed

Returns:
- returns train_data_list, test_data_list, valid_data_list
- data format is "{"file_path":<file_path>, "label_name":<label string>, "label_idx":<label index>}"
"""
def process_input_files(data_dir, split_ratio, limit=0):

    global label_list

    if limit:
        limit = int(limit / float(len(label_list)))

    # read all files
    label_data_dict = {}
    for label in label_list:
        label_data_dict[label] = []
    dir_list = os.listdir(data_dir)
    for dir in dir_list:
        if dir == "_background_noise_":
            continue
        dir_path = os.path.join(data_dir, dir)
        if dir in label_list:
            label = dir
        else:
            label = "unknown"
        file_list = os.listdir(dir_path)
        for file in file_list:
            if not file.endswith('.wav'):
                continue
            file_path = os.path.join(dir_path, file)
            label_data_dict[label].append({"file_path":file_path, "label_name":label, "label_idx":label_idx[label]})
    

    # calculate ratio
    ratios = split_ratio.split(':')
    if len(ratios) < 2 or len(ratios) > 3:
        raise Exception("ratio format error")
        
    train_ratio = int(ratios[0])
    test_ratio = int(ratios[1])
    valid_ratio = 0
    if len(ratios) == 3:
        valid_ratio = int(ratios[2])
    tot = train_ratio + test_ratio + valid_ratio
    train_ratio = train_ratio / float(tot)
    test_ratio = test_ratio / float(tot)
    valid_ratio = valid_ratio / float(tot)

    # split data into ratio
    train_data_list = []
    test_data_list = []
    valid_data_list = []
    
    for label_data in label_data_dict:
        data_len = len(label_data_dict[label_data])
        if limit and data_len >= limit:
            data_len = limit
        train_cnt = int(round(data_len * train_ratio))
        test_cnt = int(round(data_len * test_ratio))
        valid_cnt = int(round(data_len * valid_ratio))
        print("[{}] train:{} test:{} valid:{}".format(label_data, train_cnt, test_cnt, valid_cnt))
        train_data_list.extend(label_data_dict[label_data][:train_cnt])
        test_data_list.extend(label_data_dict[label_data][train_cnt:train_cnt+test_cnt])
        valid_data_list.extend(label_data_dict[label_data][train_cnt+test_cnt:train_cnt+test_cnt+valid_cnt])
    
    return train_data_list, test_data_list, valid_data_list
    
    

def label_idx_to_one_hot(y_label_idx):

    global label_len

    one_hot_list = [0] * label_len
    one_hot_list[y_label_idx] = 1
    return one_hot_list


def one_hot_to_label(one_hot):

    global label_list

    max_val = max(one_hot)
    idx = one_hot.index(max_val)
    return label_list[idx]



def preprocess_data_to_file_path_label_set(data_list):

    x_list = []
    y_list = []
    for data in data_list:
        x_list.append(data['file_path'])
        y_list.append(label_idx_to_one_hot(data['label_idx']))

    return x_list, y_list


def preprocess_data_to_x_y_set(data_list):

    x_list = []
    y_list = []
    for data in data_list:
        x_list.append(data['data'])
        y_list.append(label_idx_to_one_hot(data['label_idx']))

    return x_list, y_list




def add_padding_to_x_1d(data, x_len):
    data_len = len(data)
    if data_len != x_len:
        data.extend([0] * (x_len - data_len))
    return data



def mini_batch(x_list, y_list, batch_size):

    if batch_size == 0:
        yield x_list, y_list

    else:
        tot_len = len(x_list)
        for i in xrange(0, tot_len, batch_size):
            yield x_list[i:i+batch_size], y_list[i:i+batch_size]








if __name__ == "__main__":

    # dir_path = sys.argv[1]
    # ratio = sys.argv[2]

    # train_data, test_data, valid_data = process_input_files(dir_path, ratio)


    print(one_hot_to_label([0, 0, 0, 1]))










