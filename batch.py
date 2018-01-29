# -*- coding: utf-8 -*-

import math



class BATCH:

    def __init__( self, mini_batch=False, preload=True, batch_size=0, 
                   x_data_list=None, file_path_list=None, 
                   label_list=None, preprocess_func=None, x_shape=None ):

        self.mini_batch = mini_batch
        self.preload = preload
        self.batch_size = batch_size
        self.file_path_list = file_path_list
        self.x_data_list = x_data_list
        self.label_list = label_list
        self.preprocess = preprocess_func
        self.x_shape = x_shape

        if self.preload and not self.x_data_list:
            self.x_data_list = self.preprocess(self.file_path_list, self.x_shape)


    def batch(self):

        if not self.mini_batch:
            if self.preload:
                yield self.x_data_list, self.label_list
            else:
                x_list = self.preprocess(self.file_path_list, self.x_shape)
                yield x_list, self.label_list

        else:
            tot_len = len(self.label_list)
            for i in range(0, tot_len, self.batch_size):
                if self.preload:
                    yield self.x_data_list[i:i+self.batch_size], self.label_list[i:i+self.batch_size]
                else:
                    x_list = self.preprocess(self.file_path_list[i:i+self.batch_size], self.x_shape)
                    yield x_list, self.label_list[i:i+self.batch_size]


    def clear(self):

        del self.x_data_list
        del self.label_list


    def total_batch_size(self):
    
        if self.mini_batch:
            return int(math.ceil(len(self.label_list) / float(self.batch_size)))
        else:
            return 1



if __name__ == "__main__":

    mini_batch = False
    preload = False
    batch_size = 0,
    x_data_list = []
    file_path_list = []
    label_list = []
    preprocess_func = None
    x_shape = None

    import common
    import preprocessing

    dir_path = "dataset/train/audio"
    ratio = "1:0"

    preprocess_func = preprocessing.preprocess_spectrogram_nn
    x_shape = 4059

    batch_size = 100

    data_list, _, _ = common.process_input_files(dir_path, ratio, limit=1000)
    file_path_list, label_list = common.preprocess_data_to_file_path_label_set(data_list)
    # x_data_list = preprocess_func(file_path_list, x_shape)

    #print len(file_path_list)
    #print len(x_data_list)

    batch = BATCH( mini_batch, preload, batch_size, 
                    x_data_list, file_path_list, 
                    label_list, preprocess_func, x_shape )

    for x_list, y_list in batch.batch():

        #print len(x_list), x_list[0][0], y_list[0]
        pass



