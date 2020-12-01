import os
import librosa
import numpy as np
import json
from python_speech_features import mfcc as psf_mfcc
from python_speech_features import delta
from python_speech_features import logfbank
from librosa.feature import mfcc as lb_mfcc

"""
Copyright (c) 2020 TU/e - PDEng Software Technology C2019. All rights reserved. 
@ Authors: Raha Sadeghi r.sadeghi@tue.nl; Parima Mirshafiei P.mirshafiei@tue.nl; Jorge Cordero j.a.cordero.cruz@tue.nl;
@ Contributors: Niels Rood n.rood@tue.nl; 
Description: Module that contains several tone sentiment analyzers.
Last modified: 01-12-2020
"""

class SiameseToneSentimentPredictor:
    """Feature extractor that computes audio features required for emotion detection using siamese tone models based on the architecture described in [ref].
    """

    def __init__(self, model, parameters):
        self.model = model
        self.sample_rate = parameters['audio_frequency']
        self.n_mfcc = parameters['n_mfcc']

    def __calculate_nfft(self, samplerate, winlen):
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

    def __compute_lmfe_features(self, audio_array, sample_rate):
        """Computes lmfe features used as one of the inputs of siamese tone models.

        Args:
            audio_array : Array containing an audio signal
            sample_rate : Sample rate used to record the audio signal

        Returns:
            The lmfe features corresponding to the input audio signal.
        """
        nfft = self.__calculate_nfft(sample_rate, winlen=0.025)
        fbank_feat = logfbank(audio_array, sample_rate, nfft=nfft)
        transposed_fbank_feat = fbank_feat.T
        return transposed_fbank_feat

    def __compute_mfcc_features(self, audio_array, sample_rate, _n_mfcc=13):
        """Computes mfcc features used as one of the inputs of siamese tone models.

        Args:
            audio_array : Array containing an audio signal.
            sample_rate : Sample rate used to record the audio signal.
            _n_mfcc (int, optional): Number of mffc values computed per audio signal. Defaults to 13.

        Returns:
            The mfcc features corresponding to the input audio signal
        """
        return librosa.feature.mfcc(y=audio_array, sr=sample_rate, n_mfcc=_n_mfcc)

    def __compute_features(self, audio_array):
        """Computes mfcc and lmfe features for an audio signal.

        Args:
            audio_array : Array containing an audio signal.
            sample_rate : Sample rate used to record the audio signal.

        Returns:
            The mfcc and lmfe features computed from the input audio signal.
        """
        mfcc_features = self.__compute_mfcc_features(
            audio_array, self.sample_rate, self.n_mfcc)
        lmfe_features = self.__compute_lmfe_features(
            audio_array, self.sample_rate)
        return mfcc_features, lmfe_features

    def predict(self, audio_array):
        """Predicts the sentiment in an audio

        Args:
            audio_array : array containing an audio to be analyzed

        Returns:
            probabilities of the emotions that can be predicted by a siamese tone model
        """
        mfcc_features, lmfe_features = self.__compute_features(audio_array)
        new_mfcc=np.expand_dims(mfcc_features,axis=0)
        new_lmfe=np.expand_dims(lmfe_features,axis=0)
        print("mfcc shape: {}".format(new_mfcc.shape))
        print("lmfe shape: {}".format(new_lmfe.shape))
        return np.squeeze(self.model.predict([new_mfcc, new_lmfe]))


class SequentialToneSentimentPredictor:
    """Feature extractor that computes audio features required for emotion detection using sequential tone models.
    """

    def __init__(self, model, parameters):
        self.model = model
        self.sample_rate = parameters['audio_frequency']
        self.n_mfcc = parameters['n_mfcc']

    def __compute_mfcc_features(self, audio_array, sample_rate, _n_mfcc=40):
        """Computes mfcc features used as one of the inputs of siamese tone models.

        Args:
            audio_array : Array containing an audio signal.
            sample_rate : Sample rate used to record the audio signal.
            _n_mfcc (int, optional): Number of mffc values computed per audio signal. Defaults to 13.

        Returns:
            The mfcc features corresponding to the input audio signal
        """
        mfccs = np.mean(lb_mfcc(y=audio_array, sr=np.array(
            sample_rate), n_mfcc=_n_mfcc).T, axis=0)
        x = np.expand_dims(mfccs, axis=0)
        x = np.expand_dims(x, axis=2)
        return x

    def __compute_features(self, audio_array):
        """Wrapper function used to compute mfcc features corresponding to an audio signal.

        Args:
            audio_array : Array containing an audio signal.
            sample_rate : Sample rate used to record the audio signal.

        Returns:
            The mfcc features computed from the input audio signal.
        """
        return self.__compute_mfcc_features(audio_array, self.sample_rate, self.n_mfcc)

    def predict(self, audio_array):
        """Predicts the sentiment in an audio

        Args:
            audio_array : array containing an audio to be analyzed

        Returns:
            Probabilities of the emotions that can be predicted by a sequential tone model
        """

        mfcc_features = self.__compute_features(audio_array)
        return np.squeeze(self.model.predict(mfcc_features))


