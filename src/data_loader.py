import os
import joblib


class TextModelDataLoader():
    """[summary]
    """
    def load_test_data(self, data_directory):
        """[summary]

        Args:
            data_directory ([type]): [description]
        """        
        pass


class SequentialToneModelDataLoader():
    """[summary]
    """

    def load_test_data(self, data_directory):
        """[summary]

        Args:
            data_directory ([type]): [description]

        Returns:
            [type]: [description]
        """
        audios_path = os.path.join(data_directory, "test", "audios.joblib")
        labels_path = os.path.join(data_directory, "test", "labels.joblib")

        audios = joblib.load(audios_path)
        labels = joblib.load(labels_path)

        return audios, labels


class SiameseToneModelDataLoader():
    """[summary]
    """

    def load_test_data(self, data_directory):
        """[summary]

        Args:
            data_directory ([type]): [description]

        Returns:
            [type]: [description]
        """
        xmfcc_path = os.path.join(data_directory, "test", "xmfcc.joblib")
        xlmfe_path = os.path.join(data_directory, "test", "xlmfe.joblib")
        labels_path = os.path.join(data_directory, "test", "labels.joblib")

        xmfcc = joblib.load(xmfcc_path)
        xlmfe = joblib.load(xlmfe_path)
        labels = joblib.load(labels_path)

        return xmfcc, xlmfe, labels
