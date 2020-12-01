import os
import time
import joblib
import sys
import librosa
import numpy as np
from python_speech_features import logfbank
import yaml
from pathlib import Path
from pydub import AudioSegment
from pydub.utils import make_chunks


__authors__ = "Raha Sadeghi, Parima Mirshafiei",
__email__ = "r.sadeghi@tue.nl; P.mirshafiei@tue.nl;"
__copyright__ = "TU/e ST2019"

class FeatureExtractor:
    """
    This class contains all the required functions for extracting the most important features of audio files.
    The features can be saved in form of .joblib files to be used multiple times in training phase or prediction.
    """
    def __init__(self):
       pass

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
    
    
    def preprocessing(self):
        """
        This function traverses all the raw data directories and modify audio tracks with length
        less than chunk_length_in_milli_sec.
        """
        directories = [self.train_data_directory, self.validation_data_directory, self.test_data_directory]
        for directory in directories:
            for subdir, dirs, files in os.walk(directory):
                for file in files:
                    try:
                        self.audio_padding(str(os.path.join(subdir, file)))
                    except ValueError as err:
                        print(err)
                        


    
    def audio_padding(self, audio_file_path):
        """
        This function add silence to audio tracks with length less than chunk_length_in_milli_sec.
        the new files will be saved with the same name in the same path.
        :param audio_file_path: full path to the file (audio track)
        """

        audio_file = AudioSegment.from_file(audio_file_path, "wav")
        duration_in_sec = len(audio_file)  # Length of audio in milli-seconds
        if self.chunk_length_in_milli_sec > duration_in_sec:
            pad_ms = (self.chunk_length_in_milli_sec - duration_in_sec)  # milliseconds of silence needed
        else:
            pad_ms = 0

        silence = AudioSegment.silent(duration=pad_ms)

        audio = AudioSegment.from_wav(audio_file_path)

        # Adding silence after the audio
        padded = audio + silence
        padded.export(audio_file_path, format='wav')

    def __lmfe_feature_extractor(self, sig, sample_rate):
        """
        This function calculates log filter bank and transposes it.
        :param sig: audio time series
        :param sample_rate: sampling rate of sig
        """
        nfft = self.__calculate_nfft(sample_rate, winlen=0.025)
        fbank_feat = logfbank(sig, sample_rate, nfft=nfft)
        transposed_fbank_feat = fbank_feat.T
        return transposed_fbank_feat

    def __sequential_lmfe_feature_extractor(self, sig, sample_rate):
        """
        This function calculates log filter bank and transpose it.
        :param sig: audio time series
        :param sample_rate: sampling rate of sig
        """
        nfft = self.__calculate_nfft(sample_rate, winlen=0.025)
        fbank_feat = np.mean(logfbank(sig, sample_rate, nfft=nfft).T, axis=0)
        return fbank_feat
    
    def __mfcc_feature_extractor(self, sig, sample_rate):
        """
        This function calculates mel frequency cepstral coefficient feature _ mfcc.
        :param sig: audio time series
        :param sample_rate: sampling rate of sig
        """
        return librosa.feature.mfcc(y=sig, sr=sample_rate, n_mfcc=self.n_mfcc)

    def __sequential_mfcc_extractor(self, sig, sample_rate):
        """
        This function calculates the mean of transpose of mel frequency cepstral coefficient.
        :param sig: audio time series
        :param sample_rate: sampling rate of sig
        """
        mfccs = np.mean(librosa.feature.mfcc(y=sig, sr=sample_rate, n_mfcc=self.n_mfcc).T, axis=0)
        return mfccs

    def tone_features_creator(self, goal="train", feature_type= "mfcc_sequential"):
        """
        Capturing mfcc feature.
        :param goal: "train", "validation" or "test". using this paramter, one can define features of 
        which dataset should be extracted.
        :param feature_type: "mfcc_sequential", "lmfe_sequential",  "mfcc", or "lmfe". User can pass one of these three
        options to define which feature should be extracted. notice that mfcc_sequential is the mfcc 
        feature that should be extracted in the sequential model.
        """
        lst_X = []
        lst_y = []
        start_time = time.time()
        if goal == "test":
            file_path = self.test_data_directory
        elif goal == "validation":
            file_path = self.validation_data_directory
        else:
            file_path = self.train_data_directory

        for subdir, dirs, files in os.walk(file_path):
            for file in files:
                try:
                    # Load librosa array, 
                    # obtain required feature, 
                    # store the file and the feature information in a new array
                   
                    emotion_type = file[7:8]
                    # considered emotions: "happy, angray, sad, neutral, and fearful"
                    if emotion_type == "1" or emotion_type == "3" or emotion_type == "4" or emotion_type == "5" or \
                            emotion_type == "6":

                        X, sample_rate = librosa.load(os.path.join(subdir, file),
                                                      res_type='kaiser_fast')
                        if feature_type == "mfcc_sequential":
                            features = self.__sequential_mfcc_extractor(X, sample_rate)
                        elif feature_type == "mfcc":
                            features = self.__mfcc_feature_extractor(X, sample_rate)
                        elif feature_type == "lmfe":
                            features = self.__lmfe_feature_extractor(X, sample_rate)
                        elif feature_type == "lmfe_sequential":
                            features = self.__sequential_lmfe_feature_extractor(X, sample_rate)
                        else:
                            raise Exception("unsupported feature")
                        lst_X.append(features)

                        # The instruction below converts the labels (from 1 to 8) to a series from 0 to 7
                        # This is because our predictor needs to start from 0 otherwise it will try to predict also 0.
                        label = int(file[7:8]) - 1
                        if label == 0:
                            pass # neutral
                        elif label== 2:
                            label = 1 # happy
                        elif label == 3:
                            label = 2 # sad
                        elif label == 4:
                            label = 3 #angry
                        elif label == 5:
                            label = 4 # fearful
                        else:
                            raise Exception("unsupported emotion")
                        
                        lst_y.append(label)

                # If the file is not valid, skip it
                except ValueError as err:
                    print(err)
                    continue

        print("--- Data loaded. Loading time: %s seconds ---" % (time.time() - start_time))
        
        if feature_type == "mfcc_sequential":
            feature_type = "mfcc"
        if feature_type == "lmfe_sequential":
            feature_type = "lmfe"
        
        self.__saving_features( lst_X, lst_y, feature_type, goal)
        

    
    def __saving_features(self, lst_x, lst_y, type_, goal):
        """
        Saving features for later use. in case we want to retrain our model by only changing
        the hyper parameters, there is no need to extract features again.
        :param save_dir: the directory to save the features
        :param lst_y: the lables
        :param lst_x: the feature (mfcc or lmfe)
        :param type_ either: "mfcc" or "lmfe"
        :param goal: "train", "validation", or "test" to show whether we want to
        validate with the current dataset or a new dataset, which is located in other location
        """

        # Array conversion
        X, y = np.asarray(lst_x), np.asarray(lst_y, dtype=np.float32)

        # Preparing  features dump
        if goal == "train":
            x_name, y_name = type_ +'_' + goal + '.joblib', 'labels_' + goal + '.joblib'
            save_dir = self.feature_train_data_directory
        elif goal == "validation":
            x_name, y_name = type_ +'_' + goal + '.joblib', 'labels_'  + goal +  '.joblib'
            save_dir = self.feature_validation_data_directory
        elif goal == "test":
            x_name, y_name = type_ +'_' + goal + '.joblib', 'labels_' + goal +  '.joblib'
            save_dir = self.feature_test_data_directory

        # saving features as joblib files 
        joblib.dump(X, os.path.join(save_dir, x_name))
        joblib.dump(y, os.path.join(save_dir, y_name))

        return X, y

    def load_training_parameters(self, training_parameters_filename):
        """
        This function load training parameters from yaml file.
        :param training_parameters_filename the filename/file path to the training file 
        """
        # load training parameters from yaml file
        root_path = Path(os.getcwd())
        with open(os.path.join(root_path, "training", training_parameters_filename)) as f:
            training_parameters = yaml.load(f, Loader=yaml.FullLoader)

        model_type = training_parameters["model_type"]

        # preprocess training parameters
        parameters = training_parameters['parameters']

        self.train_data_directory = parameters['raw_train_data_directory']
        self.test_data_directory = parameters['raw_test_data_directory']
        self.validation_data_directory = parameters['raw_validation_data_directory']

        self.feature_train_data_directory = parameters['feature_train_data_directory']
        self.feature_test_data_directory = parameters['feature_test_data_directory']
        self.feature_validation_data_directory = parameters['feature_validation_data_directory']

        self.n_mfcc = parameters['n_mfcc']
        self.n_lmfe = parameters['n_lmfe']
        self.chunk_length_in_milli_sec = parameters['chunk_length_in_milli_sec']
        
        return model_type


if __name__ == '__main__':
    """
    In order to extract features, once you need to run this script.
    for running the sequential model, you need either mfcc_sequential or lmfe_sequential.
    for running the siamese model, you need to extract both mfcc and lmfe feature. 
    Therefore, please run this script twice, once with an argument mfcc, another time with lmfe argument.
    """
    if len(sys.argv) < 2:
        raise Exception("Please specify the feature to be extracted (mfcc, lmfe, mfcc_sequentiol, or lmfe_sequential) ")
    feature_type = sys.argv[1]
    if feature_type is None:
        raise Exception("Please specify the feature to be extracted (mfcc, lmfe, mfcc_sequentiol, or lmfe_sequential) ")
    else:
        print(feature_type, " will be extracted")

    feature_extractor = FeatureExtractor()
    training_parameters_filename = "training_parameters.yml"
    model_type = feature_extractor.load_training_parameters(training_parameters_filename)
   
    # preprocessing data - modify the audio tracks by padding silence to make them same length
    feature_extractor.preprocessing()
  
    # extracting required feature for different datasets
    feature_extractor.tone_features_creator("train", feature_type)
    feature_extractor.tone_features_creator("validation", feature_type)
    feature_extractor.tone_features_creator("test", feature_type)


    