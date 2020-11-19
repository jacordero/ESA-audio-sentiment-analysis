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
import yaml

from scipy.io.wavfile import write
from librosa.feature import mfcc

import tensorflow as tf
import keras

from sentiment_prediction import SiameseToneSentimentPredictor
from sentiment_prediction import SequentialToneSentimentPredictor
from sentiment_prediction import TextSentimentPredictor
from stern_utils import Utils


class AudioAnalyzer:

	def __init__(self, tone_predictor, parameters):
		self.frame_rate = parameters['audio_frequency']

		# TODO: remove hardcoded dependency by passing the appropriate predictors according
		self.tone_predictor = tone_predictor
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


def load_tone_model(tone_model_dir, tone_model_name):
	script_dir = os.path.dirname(os.path.abspath(__file__))
	tone_model_path = os.path.normpath(os.path.join(script_dir, "../prod_models/deployed", tone_model_dir, tone_model_name))
	print(tone_model_path)
	return keras.models.load_model(tone_model_path)


def create_tone_predictor(parameters):

	tone_model = load_tone_model(parameters['model_dir'], parameters['model_name'])
	if parameters['model_type'] == "sequential":
		tone_predictor = SequentialToneSentimentPredictor(tone_model, parameters)
	elif parameters['model_type'] == "siamase":
		tone_predictor = SiameseToneSentimentPredictor(tone_model, parameters)
	else:
		raise ValueError("Invalid tone model type: {}".format(parameters['model_type']))

	return tone_predictor


def run(parameters):

	predictor_parameters = {
		'model_name': parameters['models']['tone_model']['file'],
		'model_dir': parameters['models']['tone_model']['dir'],
		'model_type': parameters['models']['tone_model']['type'],
		'audio_frequency': parameters['audio_frequency'],
		'n_mfcc': parameters['models']['tone_model']['n_mfcc']
	}

	audio_analyzer_parameters = {
		'logging_file_prefix': parameters['logging_file_prefix'],
		'logging_directory': parameters['logging_directory'],
		'audio_frequency': parameters['audio_frequency'],
	}

	tone_predictor = create_tone_predictor(predictor_parameters)
	audio_analyzer = AudioAnalyzer(tone_predictor, audio_analyzer_parameters)
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

	prod_config_file = "raspi_deployment_config.yml"
	with open(prod_config_file) as input_file:
		config_parameters = yaml.load(input_file, Loader=yaml.FullLoader)

	print(config_parameters)
	run(config_parameters)
