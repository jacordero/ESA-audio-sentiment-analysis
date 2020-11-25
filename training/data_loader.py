import os
import joblib
import numpy as np

__authors__ = "Raha Sadeghi, Jorge Cordero, Parima Mirshafiei",
__email__ = "r.sadeghi@tue.nl; j.a.cordero.cruz@tue.nl; p.mirshafiei@tue.nl;"
__copyright__ = "TU/e ST2019"


class SequentialToneModelDataLoader():
    """ 
    Data loader for sequential-based CNN tone models that perform sentiment analysis.
    """


    def load_train_data(self, data_directory):
        """
        Loads preprocessed testing audios and their correspondig labels.

        Param: data_directory : Path of the directory containing audios.

        Returns: mfcc features and their corresponding labels.
        """

        features_path = os.path.join(data_directory, "mfcc_train.joblib")
        labels_path = os.path.join(data_directory, "labels_train.joblib")
        if os.path.exists(features_path):
            features_path = os.path.join(data_directory, "lmfe_train.joblib")
            features = joblib.load(features_path)
            features = np.expand_dims(features, axis=2)
            print(features.shape)
        else :
            features_path = os.path.join(data_directory, "lmfe_train.joblib")
            if os.path.exists(features_path):
                features = joblib.load(features_path)
                features = np.expand_dims(features, axis=2)
            else:
                raise Exception("can't find the training feature file")
        if os.path.exists(labels_path):
            labels = joblib.load(labels_path)
        else:
            raise Exception("can't find the label file")

        return features, labels


    def load_validation_data(self, data_directory):
        """
        Loads preprocessed testing audios and their correspondig labels.

        Param: data_directory : Path of the directory containing audios.

        Returns: mfcc features and their corresponding labels.
        """

        features_path = os.path.join(data_directory, "mfcc_validation.joblib")
        labels_path = os.path.join(data_directory, "labels_validation.joblib")
        if os.path.exists(features_path):
            features_path = os.path.join(data_directory, "lmfe_validation.joblib")
            features = joblib.load(features_path)
            features = np.expand_dims(features, axis=2)
            print(features.shape)
        else :
            features_path = os.path.join(data_directory, "lmfe_validation.joblib")
            if os.path.exists(features_path):
                features = joblib.load(features_path)
                features = np.expand_dims(features, axis=2)
            else:
                raise Exception("can't find the validation feature file")
        if os.path.exists(labels_path):
            labels = joblib.load(labels_path)
        else:
            raise Exception("can't find the label file")

        return features, labels


    def load_test_data(self, data_directory):
        """
        Loads preprocessed testing audios and their correspondig labels.

        Param: data_directory : Path of the directory containing audios.

        Returns: mfcc features and their corresponding labels.
        """
        features_path = os.path.join(data_directory, "mfcc_test.joblib")
        labels_path = os.path.join(data_directory, "labels_test.joblib")
        if os.path.exists(features_path):
            features_path = os.path.join(data_directory, "lmfe_test.joblib")
            features = joblib.load(features_path)
            features = np.expand_dims(features, axis=2)
            print(features.shape)
        else :
            features_path = os.path.join(data_directory, "lmfe_test.joblib")
            if os.path.exists(features_path):
                features = joblib.load(features_path)
                features = np.expand_dims(features, axis=2)
            else:
                raise Exception("can't find the test feature file")
        if os.path.exists(labels_path):
            labels = joblib.load(labels_path)
        else:
            raise Exception("can't find the label file")

        return features, labels


class SiameseToneModelDataLoader():
    """ 
    Data loader for Siamese-based CNN tone models that perform sentiment analysis.
    """
    def load_train_data(self, data_directory):
        """
        Loads preprocessed validation audios and their correspondig labels.

        :Param: data_directory: Path of the directory containing processed audio features.

        Returns: mfcc/lmfe features and their corresponding labels.
        """

        mfcc_feature_path = os.path.join(data_directory, "mfcc_train.joblib")
        lmfe_feature_path = os.path.join(data_directory, "lmfe_train.joblib")
        labels_path = os.path.join(data_directory, "labels_train.joblib")

        mfcc_feature = joblib.load(mfcc_feature_path)
        mfcc_feature = np.expand_dims(mfcc_feature, axis=3)

        lmfe_feature = joblib.load(lmfe_feature_path)
        lmfe_feature = np.expand_dims(lmfe_feature, axis=3)

        labels = joblib.load(labels_path)

        return mfcc_feature, lmfe_feature, labels


    def load_validation_data(self, data_directory):
        """
        Loads preprocessed validation audios and their correspondig labels.

        :Param: data_directory: Path of the directory containing processed audio features.

        Returns: mfcc/lmfe features and their corresponding labels.
        """

        mfcc_feature_path = os.path.join(data_directory, "mfcc_validation.joblib")
        lmfe_feature_path = os.path.join(data_directory, "lmfe_validation.joblib")
        labels_path = os.path.join(data_directory, "labels_validation.joblib")

        mfcc_feature = joblib.load(mfcc_feature_path)
        mfcc_feature = np.expand_dims(mfcc_feature, axis=3)

        lmfe_feature = joblib.load(lmfe_feature_path)
        lmfe_feature = np.expand_dims(lmfe_feature, axis=3)

        labels = joblib.load(labels_path)

        return mfcc_feature, lmfe_feature, labels



    def load_test_data(self, data_directory):
        """
        Loads preprocessed testing audios and their correspondig labels.

        :Param :data_directory : Path of the directory containing processed audio features.

        Returns: mfcc/lmfe features and their corresponding labels.
        """

        mfcc_feature_path = os.path.join(data_directory, "mfcc_test.joblib")
        lmfe_feature_path = os.path.join(data_directory, "lmfe_test.joblib")
        labels_path = os.path.join(data_directory, "labels_test.joblib")

        mfcc_feature = joblib.load(mfcc_feature_path)
        mfcc_feature = np.expand_dims(mfcc_feature, axis=3)

        lmfe_feature = joblib.load(lmfe_feature_path)
        lmfe_feature = np.expand_dims(lmfe_feature, axis=3)

        labels = joblib.load(labels_path)

        return mfcc_feature, lmfe_feature, labels