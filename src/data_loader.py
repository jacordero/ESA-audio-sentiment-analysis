import os
import joblib
import numpy as np

"""
Copyright (c) 2020 TU/e - PDEng Software Technology C2019. All rights reserved. 
@ Authors: George Azis gazis@tue.nl; Niels Rood n.rood@tue.nl;
@ Contributors: Jorge Cordero j.a.cordero.cruz@tue.nl
Last modified: 01-12-2020
"""



class TextModelDataLoader():
    """ Data loader for text-based models that perform sentiment analysis.
    """

    def __list_audios(self, data_directory):
        """Lists the wav audios available in a directory.

        Args:
            data_directory : Path of directory containing test audios.

        Returns:
            List of wav filenames.
        """        
        audios = []
        audio_formats = ['wav']

        for filename in os.listdir(data_directory):
            print(filename)
            audio_format = filename.split('.')[1]
            if audio_format in audio_formats:
                audios.append(data_directory + "/" + filename)

        return audios

    def load_test_data(self, filename):
        """Loads tests sentences used to evaluate the performance of text-based sentiment detection models.

        Args:
            data_directory : Path of the directory containing audios used to generate text sentences.

        Returns:
            A list of sentences corresponding to a set of test audios.
        X
        audios_to_load = self.__list_audios(data_directory)
        feature_extractor = TextModelFeatureExtractor()

        sentences = []
        for audio_filename in audios_to_load:
            sentence = feature_extractor.compute_features(audio_filename)
            sentences.append(sentence)
        """
        sentences = np.load(filename)['arr_0']
        return sentences


class SequentialDataLoader():
    """ Data loader for sequential-based CNN tone models that perform sentiment analysis.
    """

    def load_test_data(self, data_directory):
        """Loads preprocessed testing audios and their correspondig labels.

        Args:
            data_directory : Path of the directory containing audios.

        Returns:
            Xmfcc features and their corresponding labels.
        """
        mfcc_path = os.path.join(data_directory, "sequential", "mfcc_test.joblib")
        labels_path = os.path.join(data_directory, "sequential", "labels_test.joblib")

        mfcc = joblib.load(mfcc_path)
        labels = joblib.load(labels_path)

        # Expanding dimension
        x = np.expand_dims(mfcc, axis=2)


        return x, labels


class SiameseDataLoader():
    """ Data loader for siamese-based CNN tone models that perform sentiment analysis.
    """

    def load_test_data(self, data_directory):
        """Loads preprocessed testing audios and their correspondig labels.

        Args:
            data_directory : Path of the directory containing audios.

        Returns:
            Xmfcc features, xlmfe features, and their corresponding labels.
        """
        mfcc_path = os.path.join(data_directory, "siamese", "mfcc_test.joblib")
        lmfe_path = os.path.join(data_directory, "siamese", "lmfe_test.joblib")
        labels_path = os.path.join(
            data_directory, "siamese", "labels_test.joblib")

        mfcc = joblib.load(mfcc_path)
        lmfe = joblib.load(lmfe_path)
        labels = joblib.load(labels_path)

        return mfcc, lmfe, labels
