# -*- coding: utf-8 -*-

import os
import sys

import wave
# from scipy.io import wavfile


def get_wav_length_in_sec(file_path):

    f = wave.open(file_path, 'r')
    frame_num = f.getnframes()
    frame_rate = f.getframerate()
    f.close()

    return frame_num / float(frame_rate)


def slice(input_file_path, output_file_path, start_ms, end_ms):

    in_f = wave.open(input_file_path, 'r')

    width = in_f.getsampwidth()
    rate = in_f.getframerate()
    fpms = rate / 1000 # frames per ms
    length = int((end_ms - start_ms) * fpms)
    start_index = start_ms * fpms

    # print width, rate, fpms, length, start_index

    out_f = wave.open(output_file_path, "w")
    out_f.setparams((in_f.getnchannels(), width, rate, length, in_f.getcomptype(), in_f.getcompname()))
    
    in_f.rewind()
    anchor = in_f.tell()
    in_f.setpos(anchor + start_index)
    out_f.writeframes(in_f.readframes(length))

    out_f.close()
    in_f.close()
    

if __name__ == "__main__":

    audio_dir_path = sys.argv[1]

    background_noise_path = os.path.join(audio_dir_path, "_background_noise_")
    silence_path = os.path.join(audio_dir_path, "silence")

    # make silence directory
    os.system("mkdir {}".format(silence_path))

    file_list = os.listdir(background_noise_path)
    for file_name in file_list:
        if not file_name.endswith(".wav"):
            continue
        file_path = os.path.join(background_noise_path, file_name)
        print file_path

        file_length_in_sec = int(get_wav_length_in_sec(file_path))

        for idx in xrange(file_length_/in_sec):
            out_name = "{}_{}.wav".format(os.path.splitext(file_name)[0], idx)
            out_path = os.path.join(silence_path, out_name)
            slice(file_path, out_path, idx*1e3, (idx+1)*1e3)



