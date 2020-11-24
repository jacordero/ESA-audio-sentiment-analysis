import os
import joblib
import numpy as np

class SequentialToneModelDataLoader():
    """ Data loader for sequential-based CNN tone models that perform sentiment analysis.
    """


    def load_train_data(self, data_directory):

        features_path = os.path.join(data_directory, "mfcc_train.joblib")
        labels_path = os.path.join(data_directory, "labels_train.joblib")

        features = joblib.load(features_path)
        features = np.expand_dims(features, axis=2)
        print(features.shape)
        labels = joblib.load(labels_path)

        return features, labels


    def load_validation_data(self, data_directory):

        features_path = os.path.join(data_directory, "mfcc_validation.joblib")
        labels_path = os.path.join(data_directory, "labels_validation.joblib")

        features = joblib.load(features_path)
        features = np.expand_dims(features, axis=2)
        print(features.shape)
        labels = joblib.load(labels_path)

        return features, labels


    def load_test_data(self, data_directory):
        """Loads preprocessed testing audios and their correspondig labels.

        Args:
            data_directory : Path of the directory containing audios.

        Returns:
            mfcc features and their corresponding labels.
        """
        features_path = os.path.join(data_directory, "mfcc_test.joblib")
        labels_path = os.path.join(data_directory, "labels_test.joblib")

        features = joblib.load(features_path)
        features = np.expand_dims(features, axis=2)
        print(features.shape)        
        labels = joblib.load(labels_path)

        return features, labels