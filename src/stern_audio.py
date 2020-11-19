#!/usr/bin/env python

"""
Copyright (C) Tu/e.

@author  Jorge Cordero Cruzue  j.a.cordero.cruz@tue.nl

@maintainers: Engineers

======================================================

This is the STERN-AUDIO.

"""

import time
import sys
import os
from pathlib import Path

import sounddevice as sd
import numpy as np
import pandas as pd

from scipy.io.wavfile import write
from librosa.feature import mfcc

import tensorflow as tf
import keras

from sentiment_prediction import SiameseToneSentimentPredictor
from sentiment_prediction import SequentialToneSentimentPredictor
from sentiment_prediction import TextSentimentPredictor
from stern_utils import Utils


class AudioAnalyzer:

	def __init__(self, tone_model, parameters):
		self.frame_rate = parameters['audio_frequency']

		# TODO: remove hardcoded dependency by passing the appropriate predictors according
		self.tone_predictor = SequentialToneSentimentPredictor(
		    tone_model, parameters)

		# Logging functionality
		self.logging_file_prefix = parameters['logging_file_prefix']
		script_dir = os.path.dirname(os.path.abspath(__file__))
		self.logging_directory = os.path.normpath(
		    os.path.join(script_dir, parameters['logging_directory']))
		self.__create_log_directory()

	def __create_log_directory(self):
		"""Create a logging directory if it does not exists yet.
		"""
		if not os.path.exists(self.logging_directory):
			os.makedirs(self.logging_directory)

	def emotion_logbook(self, audio_emotion_probabilities):
		"""Write utterances of the audio and detected emotions in logbook.

		Args:
			audio_emotion_probabilities: Predicted tone model emotions.
		"""
		data = []
		tone_emotion_dist = []
		for e, p in audio_emotion_probabilities:
			tone_emotion_dist.append('{}:{:.2f}'.format(e, p))
		data.append(tone_emotion_dist)

		logging_file_path = os.path.join(
		    self.logging_directory, self.logging_file_prefix)
		Utils.logging(data, logging_file_path)

	def analyze(self, audio):

		tone_predictions = self.tone_predictor.predict(np.squeeze(audio))
		tone_emotion_probabilities = [
			(Utils.tone_emotions[i], tone_predictions[i]) for i in range(len(tone_predictions))]

		self.emotion_logbook(tone_emotion_probabilities)
		return tone_emotion_probabilities


def print_emotions(title, emotion_probabilities):

	sorted_probabilities = sorted(emotion_probabilities, key=lambda x: x[1], reverse=True)

	print("\n\t***************************")
	print("\t" + title)
	print("\t***************************")
	for e, p in sorted_probabilities[:3]:
		print("\t{}: {:.2f}".format(e, p))
	print("\t****************************")


def print_welcome_message():
    print("\n\n\n\n\n\n\n\n** Starting audio demo!**\n")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(script_dir, "ascii_astronaut.txt")) as f:
        print(f.read())


def load_tone_model(tone_model_name):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    tone_model_path = os.path.normpath(os.path.join(
        script_dir, "../prod_models/deployed", tone_model_name))
    return keras.models.load_model(tone_model_path)


def load_text_model(text_model_path):
    return tf.keras.models.load_model(text_model_path)


def run(tone_model_name, text_model_path, parameters):

	tone_model = load_tone_model(tone_model_name)
	audio_analyzer = AudioAnalyzer(tone_model, parameters)
	print_welcome_message()

	keep_running = True
	recorded_audio_counter = 0
	while keep_running:

		print("\n\n==================================================================\n")
		print("1) Recording. Speak for the next five seconds")
		time.sleep(parameters['short_pause'])

		recorded_audio = sd.rec(int(parameters['audio_length'] * parameters['audio_frequency']),
                                samplerate=parameters['audio_frequency'], channels=parameters['audio_channels'])
		sd.wait()

		predicted_emotion_probs = audio_analyzer.analyze(recorded_audio)
		print("\n2) Predictions")
		print_emotions("Audio prediction", predicted_emotion_probs)

		recorded_audio_counter += 1
		if recorded_audio_counter > parameters['live_audio_iterations']:
			keep_running = False

		print("\n3) Pause for {} seconds".format(parameters['long_pause']))
		time.sleep(parameters['long_pause'])

	print("\n** Demo finished! **")


if __name__ == "__main__":

	usage_message = """"
Usage of demo script.
> python demo.py [audio model path] [text model path] [mode] [pause duration]
"""
	if len(sys.argv) < 5:
		print(usage_message)
		exit(0)

	audio_model_path = sys.argv[1]
	text_model_path = sys.argv[2]

	parameters = {
		'mode': sys.argv[3],
		'live_audio_iterations': int(sys.argv[4]),
		'recorded_audio_counter': 1,
		'audio_channels': 1,
		'audio_frequency': 44100,
		'audio_length': 5,
		'n_mfcc': 40,
		'long_pause': int(sys.argv[5]),
		'short_pause': 1,
		'test_data_dir': '../data/test',
		'test_data_filenames': [],
		'logging_directory': '../logs',
		'logging_file_prefix': 'test_logging_file'
	}

	run(audio_model_path, text_model_path, parameters)
