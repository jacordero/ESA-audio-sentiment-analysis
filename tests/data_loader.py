"""
Copyright (c) 2020 TU/e - PDEng Software Technology C2019. All rights reserved.
@ Authors: George Azis gazis@tue.nl; Niels Rood n.rood@tue.nl;
@ Contributors: Jorge Cordero j.a.cordero.cruz@tue.nl
Last modified: 01-12-2020
"""

import os
import joblib
import numpy as np


class SequentialDataLoader():
    """ Data loader for sequential-based CNN tone models that perform sentiment
        analysis.
    """

    def load_test_data(self, data_directory):
        """Loads preprocessed testing audios and their correspondig labels.

        Args:
            data_directory : Path of the directory containing audios.

        Returns:
            Xmfcc features and their corresponding labels.
        """
        mfcc_path = os.path.join(data_directory, "sequential",
                                 "mfcc_test.joblib")
        labels_path = os.path.join(data_directory, "sequential",
                                   "labels_test.joblib")

        mfcc = joblib.load(mfcc_path)
        labels = joblib.load(labels_path)

        # Expanding dimension
        x = np.expand_dims(mfcc, axis=2)

        return x, labels


class SiameseDataLoader():
    """ Data loader for siamese-based CNN tone models that perform sentiment
        analysis.
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
