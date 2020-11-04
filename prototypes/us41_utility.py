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


class UtilityClass:

    def __init__(self):
        pass

    @staticmethod
    def save_model(trained_model):

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
        predictions = model.predict_classes(x_testcnn)
        new_y_test = y_test.astype(int)
        matrix = confusion_matrix(new_y_test, predictions)

        print(classification_report(new_y_test, predictions))
        print(matrix)
