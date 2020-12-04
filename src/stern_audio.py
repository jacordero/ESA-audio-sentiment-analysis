"""
Copyright (c) 2020 TU/e - PDEng Software Technology C2019. All rights reserved. 
@ Authors: Jorge Cordero j.a.cordero.cruz@tue.nl;
@ Contributors: Parima Mirshafiei P.mirshafiei@tue.nl; Tuvi Purevsuren t.purevsuren@tue.nl; Niels Rood n.rood@tue.nl; 
Description: Module that performs sentiment analysis using tone models previously trained with the scripts available 
             in the training module
Last modified: 01-12-2020
"""

import time
import sys
import os
from pathlib import Path

import sounddevice as sd
import numpy as np
import pandas as pd
import yaml

from scipy.io.wavfile import write
from librosa.feature import mfcc

import tensorflow as tf
import keras

import librosa

from sentiment_prediction import SiameseToneSentimentPredictor
from sentiment_prediction import SequentialToneSentimentPredictor
from stern_utils import Utils
from sentiment_analyzer import AudioAnalyzer


class SternAudio:
    """ Class that records and analyses audio using tone sentiment analysis modules.
	"""

    def __init__(self, audio_analyzer, parameters):
        self.audio_analyzer = audio_analyzer
        self.audio_length = parameters['audio_length']
        self.audio_channels = parameters['audio_channels']
        self.audio_frequency = parameters['audio_frequency']
        self.before_recording_pause = parameters['before_recording_pause']
        self.after_audio_analysis_pause = parameters['after_audio_analysis_pause']
        self.iterations = parameters['iterations']
        self.input_type = parameters['input_type']
        self.input_directory = parameters['input_directory']

    def print_emotions(self, emotion_probabilities):
        """ Prints a formatted message showing the top 3 emotions detected in an audio frame.

		Args:
			title: Title of the message to be displayed.
			emotion_probabilities: Tuple containing emotions and probabilities to be displayed.
		"""
        sorted_probabilities = sorted(emotion_probabilities, key=lambda x: x[1], reverse=True)

        print("\t***************************")
        for e, p in sorted_probabilities[:3]:
            print("\t{}: {:.2f}".format(e, p))
        print("\t****************************")

    def print_welcome_message(self):
        """ Supporting function that prints a welcome message.
		"""
        print("\n\n\n\n\n\n\n\n** Audio analysis started!**\n")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(script_dir, "ascii_astronaut.txt")) as f:
            print(f.read())

    def run(self):
        """ Main function that peforms continuous emotion detection from audios.
		"""
        self.print_welcome_message()
        if self.input_type == "mic":
            self.mic_predict()

        elif self.input_type == "recorded":
            self.recorded_predict()

        print("\n** Audio analysis finished!**")

    def mic_predict(self):
        try:
            keep_running = True
            recorded_audio_counter = 0
            while keep_running:

                print("\n\n==================================================================\n")
                print("1) Recording. Speak for the next {} seconds".format(self.audio_length))
                time.sleep(self.before_recording_pause)

                recorded_audio = sd.rec(int(self.audio_length * self.audio_frequency),
                                        samplerate=self.audio_frequency, channels=self.audio_channels)
                sd.wait()

                predicted_emotion_probs = audio_analyzer.analyze(recorded_audio)

                ordinal = lambda n: "%d%s" % (n, "tsnrhtdd"[(n // 10 % 10 != 1) * (n % 10 < 4) * n % 10::4])

                print("\n- Predictions for the {} record: ".format(str(ordinal(recorded_audio_counter + 1))))
                self.print_emotions(predicted_emotion_probs)

                recorded_audio_counter += 1
                if -1 < self.iterations <= recorded_audio_counter:
                    keep_running = False

                print("\n3) Pause for {} second(s)".format(self.after_audio_analysis_pause))
                time.sleep(self.after_audio_analysis_pause)
        except ValueError as err:
            print(err)

    def recorded_predict(self):
        try:
            for subdir, dirs, files in os.walk(self.input_directory):
                for file in files:
                    try:
                        # Load librosa array, obtain lmfe, store the file and the lmfe information in a new array
                        audio, sample_rate = librosa.load(os.path.join(subdir, file),
                                                          res_type='kaiser_fast', sr=self.audio_frequency)
                        predicted_emotion_probs = audio_analyzer.analyze(audio)
                        print("\n- Predictions for: " + file)
                        self.print_emotions(predicted_emotion_probs)

                    # If the file is not valid, skip it
                    except ValueError as err:
                        print(err)
                        continue
        # If the directory is not valid, raise the error
        except ValueError as err:
            print(err)


def load_tone_model(prod_models_dir, tone_model_dir, tone_model_name):
    """ Supporting function that loads tone models

	Args:
		prod_models_dir : relative path of the production models directory
		tone_model_dir : name of the directory containing the tone model to be loaded
		tone_model_name : name of the tone model

	Returns:
		A tone model
	"""
    root_path = Path(os.getcwd())
    tone_model_dir_path = os.path.normpath(os.path.join(root_path, prod_models_dir, tone_model_dir, tone_model_name))
    return tf.keras.models.load_model(tone_model_dir_path)


def create_tone_predictor(parameters):
    """ Factory function that creates a tone predictor.

	Args:
		parameters : Dictionary containing the parameters required to initialize the
		appropriate tone predictor.

	Raises:
		ValueError: model type is not yet supported

	Returns:
		Tone predictor.
	"""
    tone_model = load_tone_model(parameters['prod_models_dir'], parameters['model_dir'], parameters['model_name'])
    print(tone_model.summary())
    if parameters['model_type'] == "Sequential":
        tone_predictor = SequentialToneSentimentPredictor(tone_model, parameters)
    elif parameters['model_type'] == "Siamese":
        tone_predictor = SiameseToneSentimentPredictor(tone_model, parameters)
    else:
        raise ValueError("Invalid tone model type: {}".format(parameters['model_type']))

    return tone_predictor


def parse_parameters(parameters):
    """Supporting function that parses the configuration parameters loaded from the configuration file.

	Args:
		parameters : Dictionary containing the parameters of the configuration file

	Returns:
		Four dictionaries containing parameters to initialize model predictors, the audio analyzer class, the stern audio class, and tone emotion mapping.
	"""

    tone_emotions = {}
    for key, value in parameters['emotions'].items():
        tone_emotions[key] = value

    predictor_parameters = {
        'prod_models_dir': parameters['prod_models_dir'],
        'model_name': parameters['model']['file'],
        'model_dir': parameters['model']['dir'],
        'model_type': parameters['model']['type'],
        'audio_frequency': parameters['audio_frequency'],
        'n_mfcc': parameters['model']['n_mfcc']
    }

    analyzer_parameters = {
        'logging_file_prefix': parameters['logging_file_prefix'],
        'logging_directory': parameters['logging_directory'],
        'audio_frequency': parameters['audio_frequency'],
        'emotions': tone_emotions
    }

    stern_audio_parameters = {
        'audio_length': int(parameters['audio_length']),
        'audio_frequency': int(parameters['audio_frequency']),
        'audio_channels': int(parameters['audio_channels']),
        'before_recording_pause': int(parameters['before_recording_pause']),
        'after_audio_analysis_pause': int(parameters['after_audio_analysis_pause']),
        'iterations': int(parameters['iterations']),
        'input_type': parameters['input_type'],
        'input_directory': parameters['input_directory']
    }

    return (predictor_parameters, analyzer_parameters, stern_audio_parameters)


if __name__ == "__main__":
    """Configures and executes the program that performs sentiment analysis on audio recorded from a microphone.
	"""

    usage = "\n\nError: Configuration file for stern_audio not found.\n\nUsage:\n> python src/stern_audio.py configuration_file.yml\n"

    if len(sys.argv) > 1:
        configuration_file = sys.argv[1]
    else:
        print(usage)
        exit(1)

    root_path = Path(os.getcwd())
    with open(os.path.join(root_path, configuration_file)) as input_file:
        config_parameters = yaml.load(input_file, Loader=yaml.FullLoader)

    parsed_parameters = parse_parameters(config_parameters)
    tone_predictor = create_tone_predictor(parsed_parameters[0])
    audio_analyzer = AudioAnalyzer(tone_predictor, parsed_parameters[1])
    stern_audio = SternAudio(audio_analyzer, parsed_parameters[2])

    stern_audio.run()
