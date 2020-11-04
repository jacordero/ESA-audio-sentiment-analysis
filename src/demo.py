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

def prepare_audio_for_tone_model(audio_array, sample_rate, _n_mfcc):
	
	mfccs = np.mean(mfcc(y=audio_array, sr=np.array(sample_rate), n_mfcc=_n_mfcc).T, axis=0)
	x = np.expand_dims(mfccs, axis=0)
	x = np.expand_dims(x, axis=2)
	return x

def save_wav_pcm(audio_array, output_audio_filename, audio_frequency):
	
	tmp_audio = np.copy(audio_array)
	tmp_audio /=1.414
	tmp_audio *= 32767
	int16_data = tmp_audio.astype(np.int16)
	write(output_audio_filename, audio_frequency, int16_data)


def prepare_sentence_for_text_model(input_audio_filename):

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
	
	sorted_probabilities = sorted(emotion_probabilities, key=lambda x: x[1], reverse=True)
	
	print("\n***************************")
	print(title)
	print("***************************")
	for e, p in sorted_probabilities[:3]:
		print("{}: {:.2f}".format(e, p))
	print("****************************")


def live_audio_analysis(audio_model, text_model, parameters):

	myrecording = sd.rec(int(parameters['audio_length'] * parameters['audio_frequency']), 
		                 samplerate=parameters['audio_frequency'], 
		                 channels=parameters['audio_channels'])
	sd.wait()

	recorded_audio_filename = parameters['recorded_audio_filename_template'].format(parameters['recorded_audio_counter'])
	recorded_audio_path = os.path.join(parameters['script_dir'], recorded_audio_filename)
	save_wav_pcm(myrecording, recorded_audio_path, parameters['audio_frequency'])

	print("<- Finished recording.")	
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

	return audio_emotion_probabilities, text_emotion_probabilities



def recorded_audio_analysis():
	return ([], [])



def run(audio_model_path, text_model_path, parameters):

	audio_model = keras.models.load_model(audio_model_path)
	text_model =  tf.keras.models.load_model(text_model_path)

	script_dir = os.path.dirname(os.path.abspath(__file__))
	parameters['script_dir'] = script_dir
	print("\n** Starting audio demo!\n **")
	with open(os.path.join(script_dir, "ascii_astronaut.txt")) as f:
		print(f.read())

	keep_running = True
	while keep_running:

		print("\n\n\n-> Speak for the next five seconds")
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

		print_emotions("Audio prediction", predicted_emotion_probs[0])
		print_emotions("Text prediction", predicted_emotion_probs[1])

		time.sleep(parameters['long_pause'])

	print("\n** Demo finished! **")


if __name__ == "__main__":

	usage_message = """"
Usage of demo script.
> python demo.py [audio model path] [text model path] [mode]
"""

	if len(sys.argv) < 4:
		print(usage_message)
		exit(0)
		
	audio_model_path = sys.argv[1]
	text_model_path = sys.argv[2]

	parameters = {
		'mode': sys.argv[3],
		'live_audio_iterations': 4,
		'recorded_audio_counter': 1,
		'audio_channels': 1,
		'audio_frequency': 44100,
		'audio_length': 5,
		'n_mfcc': 40,
		'recorded_audio_filename_template': "../data/output_{:0>4}.wav",
		'long_pause': 10,
		'short_pause': 1,
		'text_model_max_features': 20000,
		'text_model_embedding_dim': 128,
		'text_model_sequence_length': 50,
		'text_model_batch_size': 32,
		'test_data_dir': '../data/test',
		'test_data_filenames': []
	}

	run(audio_model_path, text_model_path, parameters) 
