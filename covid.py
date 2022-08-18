import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras import Sequential, Model
from keras.layers import concatenate, Activation, Dense, Dropout, Conv2D, Flatten, MaxPooling2D, GlobalMaxPooling2D, \
    GlobalAveragePooling2D, AveragePooling2D, Input, Add, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.models import model_from_json
from sklearn.metrics import roc_curve
from keras.utils import np_utils
from keras.models import load_model
from tqdm.auto import tqdm
import pandas as pd
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import cv2
import os
import wave
import shutil
import socket
from _thread import *
import io
import subprocess
import os
import threading
import pickle
import sys
import time

def load_wav(path):
    return librosa.core.load(path, sr=24000)[0]

def trim_silence(wav, top_db=23, fft_size=512, hop_size=128):
    return librosa.effects.trim(wav, top_db=top_db, frame_length=fft_size, hop_length=hop_size)[0]

def spectral_features(y, n_fft, window_size):
    S, phase = librosa.magphase(librosa.stft(y=y, n_fft=n_fft, hop_length=window_size + 1))

    centroid = librosa.feature.spectral_centroid(S=S, n_fft=n_fft)
    bandwidth = librosa.feature.spectral_bandwidth(S=S)
    flatness = librosa.feature.spectral_flatness(S=S)
    rolloff = librosa.feature.spectral_rolloff(S=S, sr=24000)

    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=window_size + 1))
    contrast = librosa.feature.spectral_contrast(S=S, sr=24000)

    return np.concatenate([centroid, bandwidth, flatness, rolloff, contrast], axis=0)

def segment_cough(x, fs, cough_padding=0.2, min_cough_len=0.2, th_l_multiplier=0.1, th_h_multiplier=2):
    cough_mask = np.array([False] * len(x))

    # Define hysteresis thresholds
    rms = np.sqrt(np.mean(np.square(x)))
    seg_th_l = th_l_multiplier * rms
    seg_th_h = th_h_multiplier * rms
    # 기침소리 크기 컷팅을 위한 최저,최고 임계값

    # Segment coughs-> 기침소리 + 작은기침소리(+잡음)에서 앞부분만 자르기 위함
    coughSegments = []
    padding = round(fs * cough_padding)
    # 기침 컷팅 후 전후의 추가되는 샘플수
    min_cough_samples = round(fs * min_cough_len)
    # 기침의 최소샘플수
    cough_start = 0
    cough_end = 0
    cough_in_progress = False
    tolerance = round(0.01 * fs)
    below_th_counter = 0
    # 기침 진행 중 sample 값이 low임계값보다 작은 횟수

    for i, sample in enumerate(x ** 2):
        # enumerate를 사용해 현재 반복 횟수(i)와 현재 반복의 항목 값(sample) 저장
        if cough_in_progress:
            if sample < seg_th_l:
                below_th_counter += 1
                if below_th_counter > tolerance:
                    cough_end = i + padding if (i + padding < len(x)) else len(x) - 1
                    cough_in_progress = False
                    if (cough_end + 1 - cough_start - 2 * padding > min_cough_samples):
                        coughSegments.append(x[cough_start:cough_end + 1])
                        cough_mask[cough_start:cough_end + 1] = True
            elif i == (len(x) - 1):
                cough_end = i
                cough_in_progress = False
                if (cough_end + 1 - cough_start - 2 * padding > min_cough_samples):
                    coughSegments.append(x[cough_start:cough_end + 1])
            else:
                below_th_counter = 0
        else:
            if sample > seg_th_h:
                cough_start = i - padding if (i - padding >= 0) else 0
                cough_in_progress = True

            # think- 앞에 패딩을 해주기위한 과정인 거 같은데, 만약 기침 시작을 녹음과 동시에 시작으로 해서 기침 시작 sample의 index가 padding의 못 미치면?

    return coughSegments, cough_mask

def compute_SNR(x, fs):
    """Compute the Signal-to-Noise ratio of the audio signal x (np.array) with sampling frequency fs (float)"""
    segments, cough_mask = segment_cough(x, fs)
    RMS_signal = 0 if len(x[cough_mask]) == 0 else np.sqrt(np.mean(np.square(x[cough_mask])))
    RMS_noise = np.sqrt(np.mean(np.square(x[~cough_mask])))
    SNR = 0 if (RMS_signal == 0 or np.isnan(RMS_noise)) else 20 * np.log10(RMS_signal / RMS_noise)
    return SNR

# librosa를 사용한 7가지 오디오 특징 추출 및 특징 set 만들기
def feature_extractor_rec(row):
    recpath = 'C:/Users/kungm/Desktop/test/custom_input'
    sr = 24000

    name = row[0]
    s=[]
    c=[]
    try:
        # audio,sr = librosa.load(row[-2])
        # For MFCCS
        audio = row[1]

        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        mfccsscaled = np.mean(mfccs.T, axis=0)
        s.append(0)
        # 열을 따라 산술 평균을 계산

        cent = librosa.feature.spectral_centroid(y=audio, sr=sr)
        centscaled = np.mean(cent.T, axis=0)
        s.append(1)

        bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
        bandwidthscaled = np.mean(bandwidth.T, axis=0)
        s.append(2)

        contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
        contrastscaled = np.mean(contrast.T, axis=0)
        s.append(3)

        flatness = librosa.feature.spectral_flatness(y=audio)
        flatnessscaled = np.mean(flatness.T, axis=0)
        s.append(4)

        rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
        rolloffscaled = np.mean(rolloff.T, axis=0)
        s.append(5)

        chroma_vec = librosa.feature.chroma_stft(y=audio, sr=sr)
        chroma_vecscaled = np.mean(chroma_vec.T, axis=0)  # 12
        s.append(6)

        Feature_Set_rec = np.concatenate(
            (mfccsscaled, centscaled, bandwidthscaled, contrastscaled, flatnessscaled, rolloffscaled, chroma_vecscaled),
            axis=None)

        # Spectogram
        plt.axis('off')  # no axis
        c.append(0)

        plt.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[])
        c.append(1)
        melspec = librosa.feature.melspectrogram(y=audio, sr=sr)
        c.append(2)
        s_db = librosa.power_to_db(melspec, ref=np.max)
        c.append(3)
        librosa.display.specshow(s_db)
        c.append(4)
        savepath_rec = os.path.join(recpath, name + '.png')
        c.append(5)
        plt.savefig(savepath_rec)
        # 여백 있게 그래프 저장
        c.append(6)
        plt.close()

    except:
        print('File cannot open:', name)
        print('s: ',s)
        print('c: ',c)

        return None, None
    return Feature_Set_rec, savepath_rec



class CustomDataset(tf.keras.utils.Sequence):
    def __init__(self, imgfiles, labels, batch_size, target_size=(224, 224), shuffle=False, scale=255, n_classes=1,
                 n_channels=3):
        self.batch_size = batch_size
        self.dim = target_size
        self.labels = labels
        self.imgfiles = imgfiles
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.n_channels = n_channels
        self.scale = scale

        self.c = 0
        self.on_epoch_end()

    def __len__(self):
        # returns the number of batches
        if len(self.imgfiles) <= 10:
            return int(len(self.imgfiles))
        else:
            int(np.floor(len(self.imgfiles) / self.batch_size))

    def __getitem__(self, index):
        # returns one batch
        if len(self.indexes) <= 10:
            indexes = self.indexes
        else:
            indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Generate data
        X, y = self.__data_generation(indexes)
        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.imgfiles))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        if len(self.indexes) <= 10:
            X = np.empty((len(self.indexes), *self.dim, self.n_channels))
            y = np.empty(len(self.indexes), dtype=int)
        else:
            X = np.empty((self.batch_size, *self.dim, self.n_channels))
            y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store list_IDs_temp
            img = cv2.imread(self.imgfiles[ID])
            img = cv2.resize(img, self.dim, interpolation=cv2.INTER_CUBIC)
            X[i,] = img / self.scale

            # Store class
            y[i] = self.labels[ID]

            self.c += 1
        return X, y  # keras.utils.to_categorical(y, num_classes=self.n_classes)

class CustomPipeline(tf.keras.utils.Sequence):
    def __init__(self, data_x, data_y, batch_size=48, shuffle=False, n_classes=1):
        self.features = data_x
        self.labels = data_y
        self.batch_size = 48
        self.shuffle = shuffle
        self.n_features = self.features.shape[1]  # 36
        self.n_classes = 1
        self.on_epoch_end()

    def __len__(self):
        if len(self.features) <= 10:
            return int(len(self.features))
        else:
            return int(np.floor(len(self.features) / self.batch_size))

    def __getitem__(self, index):
        if len(self.indexes) <= 10:
            indexes = self.indexes
        else:
            indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        X, y = self.__data_generation(indexes)
        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.features))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        if len(self.indexes) <= 10:
            X = np.empty((len(self.indexes), self.n_features))
            y = np.empty(len(self.indexes), dtype=int)
        else:
            X = np.empty((self.batch_size, self.n_features))
            y = np.empty((self.batch_size), dtype=int)

        for i, ID in enumerate(indexes):
            X[i,] = self.features[ID]
            y[i,] = self.labels[ID]
        return X, y

class TripleInputGenerator(tf.keras.utils.Sequence):
    # Wrapper of two generatos for the combined input model
    def __init__(self, X1, X2, Y, batch_size, target_size=(224, 224)):
        self.genX1 = CustomPipeline(X1, Y, batch_size=batch_size, shuffle=False)
        self.genX2 = CustomDataset(X2, Y, batch_size=batch_size, shuffle=False, target_size=target_size)

    def __len__(self):
        return self.genX1.__len__()

    def __getitem__(self, index):
        X1_batch, Y_batch = self.genX1.__getitem__(index)
        X2_batch, Y_batch = self.genX2.__getitem__(index)

        X_batch = [X1_batch, X2_batch]
        return X_batch, Y_batch



def pcm2wav(pcm_path, wav_path, channels=2, bit_depth=16, sampling_rate=48000):
    # Check if the options are valid.
    if bit_depth % 8 != 0:
        raise ValueError("bit_depth " + str(bit_depth) + " must be a multiple of 8.")

    # Read the .pcm file as a binary file and store the data to pcm_data
    with open(pcm_path, 'rb') as opened_pcm_file:
        pcm_data = opened_pcm_file.read()

        obj2write = wave.open(wav_path, 'wb')
        obj2write.setnchannels(channels)
        obj2write.setsampwidth(bit_depth // 8)
        obj2write.setframerate(sampling_rate)
        obj2write.writeframes(pcm_data)
        obj2write.close()

def cough_s(wav_path):
    sr = 24000
    wav = load_wav(wav_path)
    cough_segments, cough_mask = segment_cough(wav, sr, cough_padding=0)
    return cough_segments, cough_mask

def mod(cough_segments, addr):
    MERGM_f = load_model('C:/Users/kungm/Desktop/test/MERGM_len10.h5')

    DATA = {}
    DATA['MFCCS'] = {}
    DATA['MEL'] = {}
    DATA['LABELS'] = {}

    with open('C:/Users/kungm/Desktop/test/loaded_data_model2_seg_1041_2480_mfcc13.pickle', 'rb') as af:
        # 파일을 열고 닫는 것을 자동으로 처리
        DATA = pickle.load(af)

    # 받은 데이터 잘라내기
    data_seg_rec = {}
    data_seg_rec['id'] = []
    data_seg_rec['data'] = []
    data_seg_rec['covid_status'] = []

    for j in range(len(cough_segments)):
        data_seg_rec['id'].append(str(addr[0]) + str(addr[1]))
        data_seg_rec['covid_status'].append('Unknown')
        data_seg_rec['data'].append(cough_segments[j])

    ds_rec = pd.DataFrame.from_dict(data_seg_rec)
    # data segement를 받아 DataFrame반환 = ds_rec

    features_rec = []
    imgpaths_rec = []

    for row in ds_rec.values:
        feature_set_rec, savepath_rec = feature_extractor_rec(row)

        if any(feature_set_rec) is None:
            return None

        features_rec.append(feature_set_rec)
        imgpaths_rec.append(savepath_rec)

    isnone = lambda x: x is not None
    label = lambda x: 3 if x == 'Unknown' else 0
    cast_x = list(map(isnone, features_rec))
    data_y = list(map(label, ds_rec['covid_status']))

    data_x = [features_rec[i] for i in range(len(features_rec)) if cast_x[i] == True]
    data_xx = [imgpaths_rec[i] for i in range(len(imgpaths_rec)) if cast_x[i] == True]
    data_y = [data_y[i] for i in range(len(features_rec)) if cast_x[i] == True]

    indices = np.arange(len(data_x))

    # 녹음된 data에 대한 처리
    DATA_rec = {}
    DATA_rec['MFCCS'] = {}
    DATA_rec['MEL'] = {}
    DATA_rec['LABELS'] = {}

    DATA_rec['MFCCS'] = np.array([data_x[i] for i in indices])
    DATA_rec['MEL'] = [data_xx[i] for i in indices]
    DATA_rec['LABELS'] = np.array([data_y[i] for i in indices])

    ###
    test_features = DATA_rec['MFCCS']
    test_imgs = DATA_rec['MEL']
    test_labels = DATA_rec['LABELS']

    TEST = TripleInputGenerator(test_features, test_imgs, test_labels, batch_size=48, target_size=(224, 224))

    np.set_printoptions(precision=6, suppress=True)

    return MERGM_f.predict(TEST)

def mkfile(addr, result):
    rec_data_dir = 'C:/Users/kungm/Desktop/test/cough/user_input_p/'
    wav_dir = 'C:/Users/kungm/Desktop/test/cough/user_input_w/'

    path = str(addr[0]) + str(addr[1])
    pcm_path = rec_data_dir + path + ".pcm"
    wav_path = wav_dir + path + ".wav"

    with open(pcm_path, "wb") as writer:
        writer.write(result)

    return pcm_path, wav_path

def recv_data(client_sock, addr, client_sockets):
    start = time.time()
    n = 0
    result = bytearray()

    while 1:
        n += 1
        data = client_sock.recv(48000)
        result += data
        # 전체 소켓을 받고 첫 번째 소켓을 제외한 나머지만 저장

        if n == 20:
            del result[0:48000]

            client_sockets.append(client_sock)

            pcm, wav = mkfile(addr, result)

            pcm2wav(pcm, wav)

            cough_segments, cough_mask = cough_s(wav)

            if cough_segments == []:
                client_sock.send('Retry'.encode('utf-8'))

                if client_sock in client_sockets:
                    client_sockets.remove(client_sock)
                print(f"{addr}와 연결 중단: NONE data")
                client_sock.close()
                return

            y_preds_test = mod(cough_segments, addr)

            if any(y_preds_test) is None:
                client_sock.send('Retry'.encode('utf-8'))

                if client_sock in client_sockets:
                    client_sockets.remove(client_sock)
                print(f"{addr}와 연결 중단: Feature extractor error")
                client_sock.close()
                return

            preds = [pred for pred in y_preds_test if pred > 0.9]

            if (len(y_preds_test) / 2) < len(preds):
                client_sock.send('positive'.encode('utf-8'))
            else:
                client_sock.send('negative'.encode('utf-8'))

            print('진단 결과를 보냈습니다.')

            if client_sock in client_sockets:
                client_sockets.remove(client_sock)
                print('client list : ', len(client_sockets))

            print(f'{addr}와 통신 종료')
            client_sock.close()
            finish = time.time()
            print(finish - start,addr)
            os.remove(wav)
            return

        elif len(data) == 0:
            client_sock.send('Retry'.encode('utf-8'))

            if client_sock in client_sockets:
                client_sockets.remove(client_sock)
            print(f"Disconnected from {addr}")
            client_sock.close()
            return



def main():
    plt.switch_backend('agg')

    host = '117.16.123.50'  #home com:'172.30.1.86'
    port = 9999

    # 소켓 생성(IP4v)
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    # 서버가 소켓을 포트에 맵핑하는 행위
    server_sock.bind((host, port))
    server_sock.listen()
    client_sockets = []

    print('>> Server Start')

    try:
        # server_sock.accept 때문에 while
        while 1:
            print("기다리는 중")
            client_sock, addr = server_sock.accept()
            print(addr)
            t = threading.Thread(target=recv_data, args=(client_sock, addr,client_sockets))
            t.start()

    except:
        print("Fail")
        server_sock.close()


if __name__ == "__main__":
    main()
