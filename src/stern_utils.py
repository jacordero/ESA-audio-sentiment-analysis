"""
Copyright (c) 2020 TU/e - PDEng Software Technology C2019. All rights reserved. 
@ Authors: Raha Sadeghi r.sadeghi@tue.nl; Parima Mirshafiei P.mirshafiei@tue.nl; Tuvi Purevsuren t.purevsuren@tue.nl
@ Contributors:  Jorge Cordero j.a.cordero.cruz@tue.nl; Niels Rood n.rood@tue.nl
Last modified: 01-12-2020
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

class Utils:
    """
    this class provides some functions that could be used for preprocessing, saving or sketching the model result
    """

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
          data: list parameters.
          logging_file_name: A name of logging file.

        """
        if data is None or len(data) == 0 :
            return

        root_folder = os.path.dirname(os.path.normpath(os.getcwd()))
        folder_name = os.path.join(root_folder, logging_file_name + '_' + str(date.today()))
        
        # if directory does not exist , create
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

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
        """ Saving model as h5 file

        Args:
            trained_model: the trained model to be saved
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
        """plotting the model history, depicting its accuracy, useful in finding overfitting

        Args:
            model_history: the history of the model (its accuracy in each epoch)
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
        """Generates and prints a summary of the models accuracy on a given dataset

        Args:
            model: tone prediction model
            x_testcnn: array of features
            y_test: array of labels
        """ 
        if len(x_testcnn) > 1:
            predict_vector = model.predict(x_testcnn)
            predictions = np.argmax(predict_vector, axis=1)
            new_y_test = y_test.astype(int)
            matrix = confusion_matrix(new_y_test, predictions)
            print(classification_report(new_y_test, predictions))
            print(matrix)

    @staticmethod
    def file_name_extractor(audio_file_path):
        """ extracts the name of an audio file from an file path

        Args:
            audio_file_path: file path

        Returns:
            name of the audio file
        """ 
        name = (audio_file_path.split(".wav")[0]).split("/")[-1]
        return name
