
import os, sys

from common import preprocess_data_to_file_path_label_set, process_input_files
from preprocessing import preprocess_cnn_nn
from batch import BATCH

dir_path = sys.argv[1]
split_ratio = "4:1:1"

train_data_list, test_data_list, valid_data_list = process_input_files(dir_path, split_ratio, 20000)

x_train_path, y_train_path = preprocess_data_to_file_path_label_set(train_data_list)

#batch = BATCH(False, True, 0, None, x_train_path, y_train_path, preprocess_cnn_nn, [None, 99, 41])

print y_train_path[:5]
print x_train_path[:5][:2][:2]


