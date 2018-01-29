
import math
import numpy as np
from scipy import signal
from scipy.io import wavfile
import librosa

from common import add_padding_to_x_1d
from noise_reduct import reduce_noise_power, trim_silence

"""
Interface:

In order to interact with nn_model file,
please follow following key points.

1. function name should be start with "preprocess_",
   followed by model name
2. function must take file_list, which is list of file path
3. function must return list of data to be trained/predicted.

ex) preprocess_spectrogram_nn

"""


def log_specgram(audio, sample_rate, window_size=20,
                 step_size=10, eps=1e-10):
    nperseg = int(round(window_size * sample_rate / 1e3))
    noverlap = int(round(step_size * sample_rate / 1e3))
    freqs, times, spec = signal.spectrogram(audio,
                                    fs=sample_rate,
                                    window='hann',
                                    nperseg=nperseg,
                                    noverlap=noverlap,
                                    detrend=False)
    return freqs, times, np.log(spec.T.astype(np.float32) + eps)


def subsample_spectrogram(spectrogram, divided_val):

    shape = spectrogram.shape
    shape = (shape[0], int(math.ceil(shape[1] / float(divided_val))))
    tmp_array = np.zeros(shape)
    for idx, arr in enumerate(spectrogram):
        tmp_array[idx] = arr[::divided_val]

    return tmp_array



def to_mfcc(audio, sample_rate):

    # mel-scaled power spectrogram
    S = librosa.feature.melspectrogram(audio, sr=sample_rate, n_mels=128)

    # Convert to log scale
    log_S = librosa.power_to_db(S, ref=np.max)

    # mfcc (top 13 Mel-frequency cepstral coefficients)
    mfcc = librosa.feature.mfcc(S=log_S, n_mfcc=50)
    #print("mfcc shape : {}".format(mfcc.shape))
    return mfcc

def reduce_noise_n_trim(audio, sample_rate):

    reduced_wav = reduce_noise_power(audio, sample_rate)
    trimed_wav, _ = trim_silence(reduced_wav)
    return trimed_wav



def preprocess_spectrogram_nn(file_list, x_len):

    data_list = []
    for file_path in file_list:
        # read file
        sample_rate, samples = wavfile.read(file_path)

        # convert to spectrogram
        _, _, spectrogram = log_specgram(samples, sample_rate)

        # normalize
        mean = np.mean(spectrogram, axis=0)
        std = np.std(spectrogram, axis=0)
        spectrogram = (spectrogram - mean) / std

        # subsample
        spectrogram = subsample_spectrogram(spectrogram, 4)

        # flat
        data = list(spectrogram.ravel())  # transform 2d array to 1d

        # add padding
        data = add_padding_to_x_1d(data, x_len)

        # add to list
        data_list.append(data)

    return data_list


def preprocess_cnn_nn(file_list, x_len):

    x_len = x_len[1] * x_len[2]
    print("x len {}".format(x_len))
    data_list = []
    for file_path in file_list:
        # read file
        sample_rate, samples = wavfile.read(file_path)

        # convert to spectrogram
        _, _, spectrogram = log_specgram(samples, sample_rate)

        # normalize
        mean = np.mean(spectrogram, axis=0)
        std = np.std(spectrogram, axis=0)
        spectrogram = (spectrogram - mean) / std

        # subsample
        spectrogram = subsample_spectrogram(spectrogram, 4)

        # flat
        data = list(spectrogram.ravel())  # transform 2d array to 1d
        #print("data {}".format(data[:10]))
        if len(data) > x_len:
            print("wrong data len: {} file path: {}".format(len(data), file_path))

        # add padding
        data = add_padding_to_x_1d(data, x_len)

        # reshape
        data = np.reshape(data, [99,41])
        #print("data shape {} len {} data: {}".format(data.shape, len(data[0]), data[0][:10]))

        # add to list
        data_list.append(data)

    return data_list


def preprocess_cnn_mfcc_noise(file_list, x_shape):

    x_len = x_shape[1] * x_shape[2]
    print("x len {}".format(x_len))
    data_list = []
    for file_path in file_list:
        # read file
        #sample_rate, samples = wavfile.read(file_path)
        samples, sample_rate = librosa.load(file_path, None)
        #print("sample shape {} sample rate {}".format(samples.shape, sample_rate))

        # noise reduction and silence trimming
        samples = reduce_noise_n_trim(samples, sample_rate)
        #print("reduce_noise_n_trim {}".format(samples.shape))

        # convert to mfcc
        mfcc = to_mfcc(samples, sample_rate)
        mfcc = np.transpose(mfcc)
        #print("mfcc transpose shape : {}".format(mfcc.shape))

        # flat
        data = list(mfcc.ravel())  # transform 2d array to 1d
        #print("data {}".format(data[:10]))
        #print("flatedned data len : {}".format(len(data)))
        if len(data) > x_len:
            print("wrong data len: {} file path: {}".format(len(data), file_path))

        # add padding
        data = add_padding_to_x_1d(data, x_len)
        #print("pad added len : {}".format(len(data)))

        # reshape
        data = np.reshape(data, x_shape[1:3])
        #print("data shape {} len {} data: {}".format(data.shape, len(data[0]), data[0][:10]))

        # add to list
        data_list.append(data)

    return data_list




# def preprocess_CNN_nn(file_list):

#     data_list = []
#     for file_path in file_list:
#         sample_rate, samples = wavfile.read(file_path)
#         _, _, spectrogram = log_specgram(samples, sample_rate)
#         data = list(spectrogram.ravel())  # transform 2d array to 1d


if __name__ == "__main__":

    import sys
    #from scipy.io import wavfile
    #import matplotlib.pyplot as plt

    wav_file_path = sys.argv[1]

    preprocess_cnn_mfcc_noise([wav_file_path], [-1, 32, 50])
    sys.exit()

    sample_rate, samples = wavfile.read(wav_file_path)
    print("{} {}".format(type(sample_rate), type(samples)))
    freqs, times, spectrogram = log_specgram(samples, sample_rate)
    print("{} {} {}".format(type(freqs), type(times), type(spectrogram)))
    print("{} {} {}".format(len(freqs), len(times), len(spectrogram)))
    # flated_data = spectrogram.ravel()
    # print("{} {}".format(len(flated_data), flated_data[0]))

    fig = plt.figure(figsize=(14, 8))
    ax1 = fig.add_subplot(211)
    ax1.set_title('Raw wave of ' + wav_file_path)
    ax1.set_ylabel('Amplitude')
    ax1.plot(np.linspace(0, sample_rate/len(samples), sample_rate), samples)

    mean = np.mean(spectrogram, axis=0)
    std = np.std(spectrogram, axis=0)
    spectrogram = (spectrogram - mean) / std

    spectrogram = subsample_spectrogram(spectrogram, 4)
    # spectrogram = subsample_spectrogram(spectrogram, 1)

    ax3 = fig.add_subplot(212)
    ax3.imshow(spectrogram.T, aspect='auto', origin='lower', 
               extent=[times.min(), times.max(), freqs.min(), freqs.max()])
    ax3.set_yticks(freqs[::16])
    ax3.set_xticks(times[::16])
    ax3.set_title('Spectrogram of ' + wav_file_path)
    ax3.set_ylabel('Freqs in Hz')
    ax3.set_xlabel('Seconds')

    plt.show()

    flated_data = spectrogram.ravel()
    print("{} {}".format(len(flated_data), flated_data[0]))
