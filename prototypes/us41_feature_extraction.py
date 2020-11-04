import os
import time
import joblib
import librosa
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import json
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

class feature_extractor:

    def __init__(self):
        crr_path = Path(os.getcwd())
        self.parent_path = crr_path.parent
        self.path = str(self.parent_path) + '\\data\\tone_cnn_8_emotions_dataset\\'


    def tone_features_creator(self, save_dir="."):
        lst = []
        save_dir = str(self.parent_path) + '\\data\\features\\tone';
        start_time = time.time()

        for subdir, dirs, files in os.walk(self.path):
            for file in files:
                try:
                    # Load librosa array, obtain mfcss, store the file and the mcss information in a new array
                    X, sample_rate = librosa.load(os.path.join(subdir, file),
                                                  res_type='kaiser_fast')
                    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate,
                                                         n_mfcc=50).T, axis=0)
                    # The instruction below converts the labels (from 1 to 8) to a series from 0 to 7
                    # This is because our predictor needs to start from 0 otherwise it will try to predict also 0.
                    file = int(file[7:8]) - 1
                    arr = mfccs, file
                    lst.append(arr)
                # If the file is not valid, skip it
                except ValueError as err:
                    print(err)
                    continue

        print("--- Data loaded. Loading time: %s seconds ---" % (time.time() - start_time))

        # Creating X and y: zip makes a list of all the first elements, and a list of all the second elements.
        X, y = zip(*lst)

        # Array conversion
        X, y = np.asarray(X), np.asarray(y)

        # Array shape check
        print(X.shape, y.shape)

        # Preparing  features dump
        X_name, y_name = 'X.joblib', 'y.joblib'

        joblib.dump(X, os.path.join(save_dir, X_name))
        joblib.dump(y, os.path.join(save_dir, y_name))

        return X, y

