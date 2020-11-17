import os
import librosa
import numpy as np
import time
import joblib
import librosa
from pathlib import Path
import matplotlib.pyplot as plt
import json
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from pydub import AudioSegment
from pydub.utils import make_chunks
import math

__authors__ = "Raha Sadeghi, Parima Mirshafiei, Jorge Cordero"
__email__ = "r.sadeghi@tue.nl; P.mirshafiei@tue.nl; j.a.cordero.cruz@tue.nl"
__copyright__ = "TU/e ST2019"

tone_emotions = {0: 'neutral',
                 1: 'calm',
                 2: 'happy',
                 3: 'sad',
                 4: 'angry',
                 5: 'fearful',
                 6: 'disgust',
                 7: 'surprised'}

text_emotions = {0: 'neutral',
                 7: 'calm',
                 1: 'happy',
                 2: 'sad',
                 3: 'angry',
                 4: 'fearful',
                 5: 'disgust',
                 6: 'surprised'}

class Utils:
    """
    this class provides some functions that could be used for preprocessing, saving or sketching the model result
    """

    def __init__(self):
        pass

    @staticmethod
    def save_model(trained_model):
        """
        saving model as h5 file
        :param trained_model the trained model to be saved
        """

        model_name = 'Emotion_Voice_Detection_Model.h5'
        save_dir = os.path.join(os.getcwd(), 'saved_models')
        # Save model and weights
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        model_path = os.path.join(save_dir, model_name)
        trained_model.save(model_path)
        print('Saved trained model at %s ' % model_path)

        model_json = trained_model.to_json()
        with open("model.json", "w") as json_file:
            json_file.write(model_json)

    @staticmethod
    def plot_trained_model(model_history):
        """
        plotting the model history, depicting its accuracy, useful in finding overfitting
        :param model_history the history of the model (its accuracy in each epoch)
        """
        # Loss plotting
        plt.plot(model_history.history['loss'])
        plt.plot(model_history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig('loss.png')
        plt.close()

        # Accuracy plotting
        plt.plot(model_history.history['accuracy'])
        plt.plot(model_history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('acc')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig('accuracy.png')

    @staticmethod
    def model_summary(model, x_testcnn, y_test):
        if len(x_testcnn) >1:
            predict_vector = model.predict(x_testcnn)
            predictions = np.argmax(predict_vector,axis=1)
            new_y_test = y_test.astype(int)
            matrix = confusion_matrix(new_y_test, predictions)
            print(classification_report(new_y_test, predictions))
            print(matrix)

    @staticmethod
    def file_name_extractor(audio_file_path):
        """
        getting the file name, useful in case we want to replace a file.
        :param audio_file_path: full path to the file (audio track)
        """
        name = (audio_file_path.split(".wav")[0]).split("/")[-1]
        return name