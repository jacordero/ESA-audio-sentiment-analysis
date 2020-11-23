"""
Copyright (C) Tu/e.

======================================================

This is the util library.

"""

import os
import librosa
import numpy as np
import time
import json
import time
import pytz
import subprocess
from datetime import datetime, date

import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import json
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from pydub import AudioSegment
from pydub.utils import make_chunks
from scipy.io import wavfile

import math

__authors__ = "Raha Sadeghi, Parima Mirshafiei, Jorge Cordero", "Niels Rood"
__email__ = "r.sadeghi@tue.nl; P.mirshafiei@tue.nl; j.a.cordero.cruz@tue.nl", "n.rood@tue.nl"
__copyright__ = "TU/e ST2019"


class Utils:
    """
    this class provides some functions that could be used for preprocessing, saving or sketching the model result
    """

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

    @staticmethod
    def save_wav_pcm(audio_array, output_audio_filename, audio_frequency):
        """Stores an audio signal as a pcm wav file

        Args:
            audio_array ([type]): Audio signal to be stored.
            output_audio_filename ([type]): Path where the audio signal is stored.
            audio_frequency ([type]): Frequency required to save the signal.
        """
        tmp_audio = np.copy(audio_array)
        tmp_audio /= 1.414
        tmp_audio *= 32767
        int16_data = tmp_audio.astype(np.int16)
        wavfile.write(output_audio_filename, audio_frequency, int16_data)

    @staticmethod
    def get_path(dir_name):
        """Find path of requested directory such as 'models' and 'data' and 'docs'.

        Args:
          dir_name: A name of directory.

        Returns:
          The full path of requested directory.

        """
        path = os.path.join(os.path.dirname(
            os.path.normpath(os.getcwd())), dir_name)
        if path:
            return path
        else:
            raise Exception("NO DIRECTORY FOUND")

    @staticmethod
    def get_time():
        """Get the current local time in the Netherlands.

        Returns:
          The current local time.

        """
        tz_nl = pytz.timezone('Europe/Amsterdam')
        time_nl = datetime.now(tz_nl)
        return time_nl.strftime("%H")

    @staticmethod
    def logging(data, logging_file_name):
        """Write the data to logging file.

        Args:
          data: list parameters for example [["13-11-20 09:53:12"], ["emotion"], ["emotion distribution"], ["text"]].
          logging_file_name: A name of logging file.

        """
        folder_path = os.path.join(os.getcwd(), 'src/logs')
        folder_name = os.path.join(
            folder_path, logging_file_name + '_' + str(date.today()))
        # if directory does not exist , create
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        #print(data)
        log_input = {
            'time': datetime.now().strftime('%d-%m-%y %H:%M:%S'),
            'tone_emotion': data[0]
        }

        log_file_name = os.path.join(
            folder_name, 'log_' + Utils.get_time() + '.json')
        if os.path.exists(log_file_name):
            with open(log_file_name) as logging_file:
                temp = json.load(logging_file)
                temp.append(log_input)
                with open(log_file_name, 'w') as logging_file:
                    json.dump(temp, logging_file, indent=2)
        else:
            # create the file if not exist and write data to file.
            with open(log_file_name, 'w+') as logging_file:
                json.dump([log_input], logging_file, indent=2)

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
        if len(x_testcnn) > 1:
            predict_vector = model.predict(x_testcnn)
            predictions = np.argmax(predict_vector, axis=1)
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
