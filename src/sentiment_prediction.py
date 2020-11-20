import os
import librosa
import numpy as np
import json
from python_speech_features import mfcc as psf_mfcc
from python_speech_features import delta
from python_speech_features import logfbank
from librosa.feature import mfcc as lb_mfcc
import speech_recognition as sr
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

# Package metadata
__authors__ = "Raha Sadeghi, Parima Mirshafiei, Jorge Cordero"
__email__ = "r.sadeghi@tue.nl; P.mirshafiei@tue.nl; j.a.cordero.cruz@tue.nl"
__copyright__ = "TU/e ST2019"
__version__ = "1.0"
__status__ = "Prototype"


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
        return np.squezee(self.model.predict(mfcc_features, lmfe_features))


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


class TextSentimentPredictor:
    """Feature extractor that computes text features required for emotion detection using text models.
    """

    def __init__(self, model, parameters):
        self.model = model
        self.max_tokens = parameters['text_model_max_features']
        self.output_sequence_length = parameters['text_model_sequence_length']
        self.batch_size = parameters['text_model_batch_size']

    def __compute_features(self, input_audio_filename):
        """Computes a text array from an audio signal stored in the filesystem.

        Args:
            input_audio_filename : Path of the wav file used to extract a sentence from the audio.

        Returns:
            Encoded text array and text array containing a sentence obtained from an audio file.
        """
        try:
            r = sr.Recognizer()
            with sr.AudioFile(input_audio_filename) as source:
                audio_content = r.record(source)
                sentence = r.recognize_google(audio_content)
                encoded_sentence = self.encode_sentence(sentence)
            print("Recognized text: {}".format(sentence))

        except:
            encoded_sentence = None
            sentence = None

        return encoded_sentence, sentence

    def encode_sentence(self, sentence):
        """[summary]

        Args:
            sentence ([type]): [description]

        Returns:
            [type]: [description]
        """
        sentence_array = np.array([[sentence]], dtype='object')
        vectorizer = TextVectorization(max_tokens=self.max_tokens,
                                       output_mode="int", output_sequence_length=self.output_sequence_length)
        text_vector = vectorizer(sentence_array)

        return text_vector

    def predict(self, input_audio_path):
        """Predicts the sentiment in the text obtained from an audio

        Args:
            input_audio_path : Path of the audio to be analyzed

        Returns:
            Probabilities of the emotions that can be predicted by a text based model
        """
        text_vector, sentence = self.__compute_features(input_audio_path)
        text_predictions = np.array([])
        if text_vector != None:
            text_predictions = np.squeeze(self.model.predict(
                text_vector, batch_size=self.batch_size))

        return text_predictions, sentence

