"""
This file can be used to predict audio emotion based on tone
"""

import keras
import librosa
import numpy as np


class ToneBasedPredictions:
    """
    Main class of the application.
    """

    def __init__(self, file_path=None, file_name=None, model_path=None):
        """
        Init method is used to initialize the main parameters and load the model.
        """

        if file_name is None:
            raise Exception("file name should be specified")

        if file_path is None:
            self.file = 'data/' + file_name
        else:
            self.file = 'data/' + file_path + '/' + file_name

        if model_path is not None:
            self.path = 'models/' + model_path + '/saved_models/Emotion_Voice_Detection_Model.h5'
        else:
            self.path = 'models/' + 'saved_models/Emotion_Voice_Detection_Model.h5'

        self.loaded_model = keras.models.load_model(self.path)

    def make_predictions(self):
        """
        Method to process the files and create your features.
        """
        data, sampling_rate = librosa.load(self.file)
        mfccs = np.mean(librosa.feature.mfcc(y=data, sr=sampling_rate, n_mfcc=40).T, axis=0)
        x = np.expand_dims(mfccs, axis=0)
        x = np.expand_dims(x, axis=2)
        predictions = np.argmax(self.loaded_model.predict(x), axis=-1)
        print("Prediction is", " ", self.convert_class_to_emotion(predictions))

    @staticmethod
    def convert_class_to_emotion(pred):
        """
        Method to convert the predictions (int) into human readable strings.
        """

        label_conversion = {'0': 'neutral',
                            '1': 'calm',
                            '2': 'happy',
                            '3': 'sad',
                            '4': 'angry',
                            '5': 'fearful',
                            '6': 'disgust',
                            '7': 'surprised'}

        for key, value in label_conversion.items():
            if int(key) == pred:
                label = value
        return label


if __name__ == '__main__':
    tone_prediction = ToneBasedPredictions(file_path='tone_cnn_happy_angry_dataset_sample_prediction_tracks',
                                           file_name='output10.wav',
                                           model_path='tone_cnn_8_emotions')
    tone_prediction.make_predictions()
    tone_prediction = ToneBasedPredictions(file_path='tone_cnn_happy_angry_dataset_sample_prediction_tracks',
                                           file_name='Recording.wav',
                                           model_path='tone_cnn_8_emotions')
    tone_prediction.make_predictions() 
