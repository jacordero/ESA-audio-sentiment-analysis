#!/usr/bin/env python

import time
import sys
import os
import keyboard
import keras
from pathlib import Path

import sounddevice as sd
import numpy as np
import pandas as pd

import speech_recognition as sr
from scipy.io.wavfile import write
from librosa.feature import mfcc

import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

from feature_extractor import SiameseToneModelFeatureExtractor
from feature_extractor import SequentialToneModelFeatureExtractor
from feature_extractor import TextModelFeatureExtractor
from stern_utils import tone_emotions
from stern_utils import text_emotions
from stern_utils import save_wav_pcm


class AudioAnalyzer:

	def __init__(self, tone_model, text_model, parameters):
		self.parameters = parameters.copy()
		self.tone_model = tone_model
		self.text_model = text_model
		self.tone_feature_extractor = SequentialToneModelFeatureExtractor()
		self.text_feature_extractor = TextModelFeatureExtractor()

	def get_new_recorded_audio_path(self):
		recorded_audio_filename = self.parameters['recorded_audio_filename_template'].format(self.parameters['recorded_audio_counter'])
		recorded_audio_path = os.path.join(self.parameters['script_dir'], recorded_audio_filename)
		self.parameters['recorded_audio_counter'] += 1
		return recorded_audio_path

	def analyze(self, audio):

		recorded_audio_path = self.get_new_recorded_audio_path()
		save_wav_pcm(audio, recorded_audio_path, self.parameters['audio_frequency'])

		preprocessed_audio = self.tone_feature_extractor.compute_features(np.squeeze(audio), self.parameters['audio_frequency'])

		# predictions happen here
		audio_predictions = np.squeeze(self.tone_model.predict(preprocessed_audio))
		audio_emotion_probabilities = [
			(tone_emotions[i], audio_predictions[i]) for i in range(len(audio_predictions))]

		#print("Preprocessing audio for text-based model")
		sentence = self.text_feature_extractor.compute_features(
			os.path.normpath(recorded_audio_path))

		if sentence != None:
			vectorizer = TextVectorization(max_tokens=self.parameters['text_model_max_features'],
										output_mode="int",
										output_sequence_length=self.parameters['text_model_sequence_length'])
			text_vector = vectorizer(sentence)
			print("Recognized text: {}".format(sentence))
			# print(text_vector)

			text_predictions = np.squeeze(self.text_model.predict(
				text_vector, batch_size=self.parameters['text_model_batch_size']))
			text_emotion_probabilities = [
				(text_emotions[i], text_predictions[i]) for i in range(len(text_predictions))]

		else:
			text_emotion_probabilities = []

		return audio_emotion_probabilities, text_emotion_probabilities


def print_emotions(title, emotion_probabilities):

	sorted_probabilities = sorted(
		emotion_probabilities, key=lambda x: x[1], reverse=True)

	print("\n\t***************************")
	print("\t" + title)
	print("\t***************************")
	for e, p in sorted_probabilities[:3]:
		print("\t{}: {:.2f}".format(e, p))
	print("\t****************************")


def print_welcome_message(script_dir):
	print("\n\n\n\n\n\n\n\n** Starting audio demo!**\n")
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
	text_model = load_text_model(text_model_path)
	audio_analyzer = AudioAnalyzer(tone_model, text_model, parameters)

	print_welcome_message(parameters['script_dir'])

	keep_running = True
	while keep_running:

		print("\n\n==================================================================\n")
		print("1) Recording. Speak for the next five seconds")
		time.sleep(parameters['short_pause'])

		recorded_audio = sd.rec(int(parameters['audio_length'] * parameters['audio_frequency']),
								samplerate=parameters['audio_frequency'],
								channels=parameters['audio_channels'])
		sd.wait()

		predicted_emotion_probs = audio_analyzer.analyze(recorded_audio)
		print("\n2) Predictions")
		print_emotions("Audio prediction", predicted_emotion_probs[0])
		print_emotions("Text prediction", predicted_emotion_probs[1])

		parameters['recorded_audio_counter'] += 1
		if parameters['recorded_audio_counter'] > parameters['live_audio_iterations']:
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
		'recorded_audio_filename_template': "../data/output_{:0>4}.wav",
		'long_pause': int(sys.argv[5]),
		'short_pause': 1,
		'text_model_max_features': 20000,
		'text_model_embedding_dim': 128,
		'text_model_sequence_length': 50,
		'text_model_batch_size': 32,
		'test_data_dir': '../data/test',
		'test_data_filenames': [],
		'script_dir': os.path.dirname(os.path.abspath(__file__))
	}

	run(audio_model_path, text_model_path, parameters)
