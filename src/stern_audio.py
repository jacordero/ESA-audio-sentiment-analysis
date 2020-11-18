"""
Copyright (C) Tu/e.

@author  Jorge Cordero Cruzue  j.a.cordero.cruz@tue.nl

@maintainers: Engineers

======================================================

This is the STERN-AUDIO.

"""
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

from stern_utils import logging, get_path

def prepare_audio_for_tone_model(audio_array, sample_rate,_n_mfcc):
	"""Extract the mfcc feature from the audio.
	
	Args:
		audio_array: An array of the audio file.
		sample_rate: An audio frequency (rate).
		_n_mfcc: Mel frequency cepstral coefficients.
	
	Returns:
		The arry of mfcc features of the audio array.

	"""
	mfccs = np.mean(mfcc(y=audio_array, sr=np.array(sample_rate), n_mfcc=_n_mfcc).T, axis=0)
	x = np.expand_dims(mfccs, axis=0)
	x = np.expand_dims(x, axis=2)
	return x

def save_wav_pcm(audio_array, output_audio_filename, audio_frequency):
	"""Save the audio array with .wav format.
	
	Args:
		audio_array: An array of the audio file.
		output_audio_filename: A name of the output file.
		audio_frequency: An audio frequency.
		
	"""
	tmp_audio = np.copy(audio_array)
	tmp_audio /=1.414	
	tmp_audio *= 32767
	int16_data = tmp_audio.astype(np.int16)
	write(output_audio_filename, audio_frequency, int16_data)

def prepare_sentence_for_text_model(input_audio_filename):
	"""Convert speech to text.

	Args:
		input_audio_filename: A name of the audio file.

	Returns:
		The array of texts.

	"""
	try:
		r = sr.Recognizer()
		with sr.AudioFile(input_audio_filename) as source:
			audio_content = r.record(source)
			text = r.recognize_google(audio_content)

		output = np.array([[text]], dtype='object')
	except: 
		output = None

	return output

emotions_dict_audio = {0: 'neutral',
    			 	1: 'calm',
    			 	2: 'happy',
    			 	3: 'sad',
    			 	4: 'angry',
    			 	5: 'fearful',
    			 	6: 'disgust',
			     	7: 'surprised'}

emotions_dict_text = {0: 'neutral',
    			 	7: 'calm',
    			 	1: 'happy',
    			 	2: 'sad',
    			 	3: 'angry',
    			 	4: 'fearful',
    			 	5: 'disgust',
			     	6: 'surprised'}

def print_emotions(title, emotion_probabilities):
	"""Display predicted emotions human readable format.

	Args:
		title: A name of models such as tone and text.
		emotion_probabilities: An array of emotion distributions.
		
	"""
	sorted_probabilities = sorted(emotion_probabilities, key=lambda x: x[1], reverse=True)
	
	print("\n\t***************************")
	print("\t" + title)
	print("\t***************************")
	for e, p in sorted_probabilities[:3]:
		print("\t{}: {:.2f}".format(e, p))
	print("\t****************************")

def live_audio_analysis(audio_model, text_model, parameters):
	"""Load data stream from microphone.

	Args:
		audio_model: A path of the audio model .
		text_model: A path of the text model.
		parameters: A dictionary of parameters.
	
	Returns:
		audio emotion probabilities and text emotion probabilities.
		
	"""
	myrecording = sd.rec(int(parameters['audio_length'] * parameters['audio_frequency']), 
		                 samplerate=parameters['audio_frequency'], 
		                 channels=parameters['audio_channels'])
	sd.wait()

	recorded_audio_filename = parameters['recorded_audio_filename_template'].format(parameters['recorded_audio_counter'])
	recorded_audio_path = os.path.join(parameters['script_dir'], recorded_audio_filename)
	save_wav_pcm(myrecording, recorded_audio_path, parameters['audio_frequency'])

	print("\n2) Finished recording. Starting analysis.")
	#sd.play(myrecording, parameters['audio_frequency'])

	#print("Preprocessing audio for tone model")
	preprocessed_audio = prepare_audio_for_tone_model(np.squeeze(myrecording), 
		 											  parameters['audio_frequency'],
		 											  parameters['n_mfcc'])

	## predictions happen here
	audio_predictions = np.squeeze(audio_model.predict(preprocessed_audio))
	audio_emotion_probabilities = [(emotions_dict_audio[i], audio_predictions[i]) for i in range(len(audio_predictions))]

	argmax_audio_prediction = np.argmax(np.squeeze(audio_predictions), axis=-1)
	predicted_emotion = emotions_dict_audio.get(argmax_audio_prediction, "Unknown")		


	#print("Preprocessing audio for text-based model")
	sentence = prepare_sentence_for_text_model(os.path.normpath(recorded_audio_path))	


	if sentence != None:
		vectorizer = TextVectorization(max_tokens=parameters['text_model_max_features'], 
									   output_mode="int", 
									   output_sequence_length=parameters['text_model_sequence_length'])
		text_vector = vectorizer(sentence)
		print("Recognized text: {}".format(sentence))
		#print(text_vector)

		text_predictions = np.squeeze(text_model.predict(text_vector, batch_size=parameters['text_model_batch_size']))
		text_emotion_probabilities = [(emotions_dict_text[i], text_predictions[i]) for i in range(len(text_predictions))]

	else:
		text_emotion_probabilities = []

	# To write the prediction output to logging file
	emotion_logbook(sentence, audio_emotion_probabilities, text_emotion_probabilities )
	
	return audio_emotion_probabilities, text_emotion_probabilities

def recorded_audio_analysis():
	"""Keep this function for future use.

	Returns:
		None.
		
	"""
	return None

def print_welcome_message(script_dir):
	"""Print welcome message and astronaut's image.

	Args:
		script_dir: A path of the script.
		
	"""
	print("\n\n\n\n\n\n\n\n** Starting audio demo!**\n")
	with open(os.path.join(script_dir, "ascii_astronaut.txt")) as f:
		print(f.read())

def run(audio_model_path, text_model_path, parameters):
	"""Run the models.

	Args:
		audio_model_path: A path of the audio model.
		text_model_path: A path of the text model.
		parameters: A dictionary of parameters.
		
	"""
	audio_model = keras.models.load_model(audio_model_path)

	text_model =  tf.keras.models.load_model(text_model_path)

	print_welcome_message(parameters['script_dir'])

	keep_running = True
	while keep_running:

		print("\n\n==================================================================\n")
		print("1) Recording. Speak for the next five seconds")
		time.sleep(parameters['short_pause'])

		if parameters['mode'] == "live":
			predicted_emotion_probs = live_audio_analysis(audio_model, 
														  text_model, 
														  parameters)
			parameters['recorded_audio_counter'] += 1
			if parameters['recorded_audio_counter'] > parameters['live_audio_iterations']:
				keep_running = False 

		elif parameters['mode'] == "recorded":
			predicted_emotion_probs = recorded_audio_analysis(audio_model, 
															  text_model, 
															  parameters)			

		else:
			print("Uknown mode. Valid modes are 'live' and 'recorded'")
			exit(0)

		print("\n3) Predictions")
		print_emotions("Audio prediction", predicted_emotion_probs[0])
		print_emotions("Text prediction", predicted_emotion_probs[1])

		print("\n4) Pause for {} seconds".format(parameters['long_pause']))
		time.sleep(parameters['long_pause'])

	print("\n** Demo finished! **")

def emotion_logbook(sentence, audio_emotion_probabilities, text_emotion_probabilities):
	"""Write utterances of the audio and detected emotions in logbook.

	Args:
		sentence: An utterances of audio.
		audio_emotion_probabilities: Predicted tone model emotions.
		text_emotion_probabilities: Predicted text model emotions.
		
	"""
	if sentence is None:
		sentence = ['Empty']
	data = []
	data.append(sentence)
	tone_emotion_dist = []
	text_emotion_dist = []
	for e, p in audio_emotion_probabilities:
		tone_emotion_dist.append('{}:{:.2f}'.format(e, p))
	for e, p in text_emotion_probabilities:
		text_emotion_dist.append('{}:{:.2f}'.format(e, p))
	data.append(tone_emotion_dist)
	data.append(text_emotion_dist)
	logging(data, parameters['logging_file_name'])

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
	#long_pause = 

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
		'script_dir': os.path.dirname(os.path.abspath(__file__)), 
		'logging_file_name': 'test_logging_file'
	}
	
	run(audio_model_path, text_model_path, parameters) 
