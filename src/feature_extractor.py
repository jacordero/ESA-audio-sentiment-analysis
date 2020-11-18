""" Extracting features of the audio data set. """
import os
import librosa
import numpy as np
import json
from python_speech_features import mfcc as psf_mfcc
from python_speech_features import delta
from python_speech_features import logfbank
from librosa.feature import mfcc as lb_mfcc
import speech_recognition as sr

# Package metadata
__authors__ = "Raha Sadeghi, Parima Mirshafiei, Jorge Cordero"
__email__ = "r.sadeghi@tue.nl; P.mirshafiei@tue.nl; j.a.cordero.cruz@tue.nl"
__copyright__ = "TU/e ST2019"
__version__ = "1.0"
__status__ = "Prototype"


class SiameseToneModelFeatureExtractor:
    """Feature extractor that computes audio features required for emotion detection using siamese tone models based on the architecture described in [ref].
    """    

    def calculate_nfft(self, samplerate, winlen):
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
        nfft = self.calculate_nfft(sample_rate, winlen=0.025)
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

    def compute_features(self, audio_array, sample_rate):
        """Computes mfcc and lmfe features for an audio signal.

        Args:
            audio_array : Array containing an audio signal.
            sample_rate : Sample rate used to record the audio signal.

        Returns:
            The mfcc and lmfe features computed from the input audio signal.
        """        
        mfcc_features = self.__compute_mfcc_features(audio_array, sample_rate)
        lmfe_features = self.__compute_lmfe_features(audio_array, sample_rate)
        return mfcc_features, lmfe_features


class SequentialToneModelFeatureExtractor:
    """Feature extractor that computes audio features required for emotion detection using sequential tone models.
    """    

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

    def compute_features(self, audio_array, sample_rate):
        """Wrapper function used to compute mfcc features corresponding to an audio signal.

        Args:
            audio_array : Array containing an audio signal.
            sample_rate : Sample rate used to record the audio signal.

        Returns:
            The mfcc features computed from the input audio signal.
        """        
        return self.__compute_mfcc_features(audio_array, sample_rate)


class TextModelFeatureExtractor:
    """Feature extractor that computes text features required for emotion detection using text models.
    """    

    def compute_features(self, input_audio_filename):
        """Computes a text array from an audio signal stored in the filesystem.

        Args:
            input_audio_filename : Path of the wav file used to extract a sentence from the audio.

        Returns:
            Text array containing a sentence obtained from an audio file.
        """
        try:
            r = sr.Recognizer()
            with sr.AudioFile(input_audio_filename) as source:
                audio_content = r.record(source)
                text = r.recognize_google(audio_content)

            output = np.array([[text]], dtype='object')
        except:
            output = None

        return output
