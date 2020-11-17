""" Extracting features of the audio data set. """
import os
import time
import joblib
import librosa
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import json
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank
import scipy.io.wavfile as wav
from us52_utility import UtilityClass

# Package metadata
__authors__ = "Raha Sadeghi, Parima Mirshafiei"
__email__ = "r.sadeghi@tue.nl; P.mirshafiei@tue.nl"
__copyright__ = "TU/e ST2019"
__version__ = "1.0"
__status__ = "Prototype"


class FeatureExtractor:

    def __init__(self):
        self.crr_path = Path(os.getcwd())
        self.path = str(self.crr_path) + '/data/tone_cnn_8_emotions_dataset/'

    def __calculate_nfft(samplerate, winlen):
        """
        Calculates the FFT size as a power of two greater than or equal to
        the number of samples in a single window length.

        Having an FFT less than the window length loses precision by dropping
        many of the samples; a longer FFT than the window allows zero-padding
        of the FFT buffer which is neutral in terms of frequency domain conversion.
        :param samplerate: The sample rate of the signal we are working with, in Hz.
        :param winlen: The length of the analysis window in seconds.
        """
        window_length_samples = winlen * samplerate
        n_fft = 1
        while n_fft < window_length_samples:
            n_fft *= 2
        return n_fft

    # todo ... change the location/file
    def preprocessing(self):
        """
        changing the file length
        """
        utility = UtilityClass()

        for subdir, dirs, files in os.walk(self.path):
            for file in files:
                try:
                    utility.audio_padding(str(os.path.join(subdir, file)))
                except ValueError as err:
                    print(err)
                    continue

    @staticmethod
    def lmfe_feature_extractor(sig, rate):
        
        nfft = FeatureExtractor.__calculate_nfft(rate, winlen=0.025)
        fbank_feat = logfbank(sig, rate, nfft=nfft)
        transposed_fbank_feat = fbank_feat.T
        return transposed_fbank_feat


    def tone_mfcc_features_creator(self):
        """
        capturing mfcc feature.
        """
        lst_mfcc = []
        lst_y = []
        save_dir = str(self.crr_path) + '/data/features/tone';
        start_time = time.time()

        for subdir, dirs, files in os.walk(self.path):
            for file in files:
                try:
                    # Load librosa array, obtain mfcss, store the file and the mcss information in a new array
                    X, sample_rate = librosa.load(os.path.join(subdir, file),
                                                  res_type='kaiser_fast')
                                                  
                    mfccs = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13)
                    lst_mfcc.append(mfccs)

                    # The instruction below converts the labels (from 1 to 8) to a series from 0 to 7
                    # This is because our predictor needs to start from 0 otherwise it will try to predict also 0.
                    label = int(file[7:8]) - 1
                    lst_y.append(label)

                # If the file is not valid, skip it
                except ValueError as err:
                    print(err)
                    continue

        print("--- Data loaded. Loading time: %s seconds ---" % (time.time() - start_time))

        Xmfcc, Ymfcc = FeatureExtractor.saving_features(save_dir, lst_mfcc, lst_y, "mfcc")
        return Xmfcc, Ymfcc

    def tone_lmfe_features_creator(self):
        """
        capturing log filterbank energy feature.
        """
        lst_y = []
        lst_lmfe = []
        save_dir = str(self.crr_path) + '/data/features/tone';
        start_time = time.time()

        for subdir, dirs, files in os.walk(self.path):
            for file in files:
                try:
                    # Load librosa array, obtain lmfe, store the file and the lmfe information in a new array
                    X, sample_rate = librosa.load(os.path.join(subdir, file),
                                                  res_type='kaiser_fast')

                    lmfe = FeatureExtractor.lmfe_feature_extractor(X, sample_rate)

                    lst_lmfe.append(lmfe)

                    # The instruction below converts the labels (from 1 to 8) to a series from 0 to 7
                    # This is because our predictor needs to start from 0 otherwise it will try to predict also 0.
                    label = int(file[7:8]) - 1
                    lst_y.append(label)
                    

                # If the file is not valid, skip it
                except ValueError as err:
                    print(err)
                    continue

        print("--- Data loaded. Loading time: %s seconds ---" % (time.time() - start_time))

        FeatureExtractor.saving_features(save_dir, lst_lmfe, lst_y, "lmfe")

    @staticmethod
    def saving_features(save_dir, lst_x, lst_y, type_):
        """
        saving features for later use. in case we want to retrain our model by only changing
        the hyper parameters, there is no need to extract features again.
        :param save_dir the directory to save the features
        :param lst_y the lables
        :param lst_x the feature (mfcc or lmfe)
        :param type_ either "mfcc" or "lmfe"
        """

        # Array conversion
        X, y = np.asarray(lst_x), np.asarray(lst_y, dtype=np.float32)

        # Preparing  features dump
        X_name, y_name = 'X' + type_ + '.joblib', 'y' + type_ + '.joblib'

        joblib.dump(X, os.path.join(save_dir, X_name))
        joblib.dump(y, os.path.join(save_dir, y_name))

        return X, y
