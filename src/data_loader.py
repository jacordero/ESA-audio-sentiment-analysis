import os
import joblib

from src.feature_extractor import TextModelFeatureExtractor


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
            audio_format = filename.split('.')[1]
            if audio_format in audio_formats:
                audios.append(data_directory + "/" + filename)

        return audios

    def load_test_data(self, data_directory):
        """Loads tests sentences used to evaluate the performance of text-based sentiment detection models.

        Args:
            data_directory : Path of the directory containing audios used to generate text sentences.

        Returns:
            A list of sentences corresponding to a set of test audios.
        """
        audios_to_load = self.__list_audios(data_directory)
        feature_extractor = TextModelFeatureExtractor()

        sentences = []
        for audio_filename in audios_to_load:
            sentence = feature_extractor.compute_features(audio_filename)
            sentences.append(sentence)

        return sentences


class SequentialToneModelDataLoader():
    """ Data loader for sequential-based CNN tone models that perform sentiment analysis.
    """

    def load_test_data(self, data_directory):
        """Loads preprocessed testing audios and their correspondig labels.

        Args:
            data_directory : Path of the directory containing audios.

        Returns:
            Xmfcc features and their corresponding labels.
        """
        xmfcc_path = os.path.join(data_directory, "test", "xmfcc_seq.joblib")
        labels_path = os.path.join(data_directory, "test", "xmfcc_seq_labels.joblib")

        xmfcc = joblib.load(xmfcc_path)
        labels = joblib.load(labels_path)

        return xmfcc, labels


class SiameseToneModelDataLoader():
    """ Data loader for siamese-based CNN tone models that perform sentiment analysis.
    """

    def load_test_data(self, data_directory):
        """Loads preprocessed testing audios and their correspondig labels.

        Args:
            data_directory : Path of the directory containing audios.

        Returns:
            Xmfcc features, xlmfe features, and their corresponding labels.
        """
        xmfcc_path = os.path.join(data_directory, "test", "xmfcc_siam.joblib")
        xlmfe_path = os.path.join(data_directory, "test", "xlmfe_siam.joblib")
        xmfcc_labels_path = os.path.join(
            data_directory, "test", "xmfcc_siam_labels.joblib")
        xlmfe_labels_path = os.path.join(
            data_directory, "test", "xlme_siam_labels.joblib")

        xmfcc = joblib.load(xmfcc_path)
        xlmfe = joblib.load(xlmfe_path)
        xmfcc_labels = joblib.load(xmfcc_labels_path)
        xlmfe_labels = joblib.load(xlmfe_labels_path)

        return xmfcc, xlmfe, xmfcc_labels, xlmfe_labels
