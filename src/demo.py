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

channels = 1 # recording monophonic audio
fs = 44100
seconds = 5
output_audio_filename_template = "../data/output_{:0>4}.wav"


def load_text_model():
	return None

def prepare_audio_for_tone_model(audio_array, sample_rate):
	
	mfccs = np.mean(mfcc(y=audio_array, sr=np.array(sample_rate), n_mfcc=40).T, axis=0)
	x = np.expand_dims(mfccs, axis=0)
	x = np.expand_dims(x, axis=2)
	return x


def save_wav_pcm(audio_array, output_audio_filename):
	
	tmp_audio = np.copy(audio_array)
	tmp_audio /=1.414
	tmp_audio *= 32767
	int16_data = tmp_audio.astype(np.int16)
	write(output_audio_filename, fs, int16_data)


def prepare_sentence_for_text_model(input_audio_filename):

	try:
		r = sr.Recognizer()
		with sr.AudioFile(input_audio_filename) as source:
			audio_content = r.record(source)
			text = r.recognize_google(audio_content)

		output = np.array([[text]], dtype='object')
	except Exception as e:
		print(e) 
		output = None

	return output


emotions_dict = {0: 'neutral',
    			 1: 'calm',
    			 2: 'happy',
    			 3: 'sad',
    			 4: 'angry',
    			 5: 'fearful',
    			 6: 'disgust',
			     7: 'surprised'}


def print_emotions(title, emotion_probabilities):
	
	sorted_probabilities = sorted(emotion_probabilities, key=lambda x: x[1])
	
	print("\n***************************************")
	print(title)
	for e, p in sorted_probabilities:
		print("{}: {:.2f}".format(e, p))
	print("***************************************")

def run(audio_model_path, text_model_path):

	pause = 10
	counter = 1
	max_features = 20000
	embedding_dim = 128
	sequence_length = 50

	sr_recognizer = sr.Recognizer()

	parent_dir = os.path.dirname(os.path.abspath(__file__))
	print(parent_dir)

	with open(os.path.join(parent_dir, "ascii_astronaut.txt")) as f:
		print(f.read())

	print("\n\n\nLoading models")

	audio_model = keras.models.load_model(audio_model_path)
	text_model =  tf.keras.models.load_model(text_model_path)

	print("Starting audio demo!")

	while True:

	#print("To record for five seconds, press Enter.")
	#print("To stop program, press q.")
	#key_pressed = keyboard.read_key()
	#if (key_pressed == "q"):
	#	exit(0)


		print("Speak for the next five seconds")
		time.sleep(1)

		#recording message
		myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=channels)
		sd.wait()

		output_audio_filename = os.path.join(parent_dir, output_audio_filename_template.format(counter))
		save_wav_pcm(myrecording, output_audio_filename)

		print("Finished recording. Start playing message")	
#		sd.play(myrecording, fs)

		print("Preprocessing audio for tone model")
		preprocessed_audio = prepare_audio_for_tone_model(np.squeeze(myrecording), fs)

		## predictions happen here
		audio_predictions = np.squeeze(audio_model.predict(preprocessed_audio))
		audio_emotion_probabilities = [(emotions_dict[i], audio_predictions[i]) for i in range(len(audio_predictions))]

		argmax_audio_prediction = np.argmax(np.squeeze(audio_predictions), axis=-1)
		predicted_emotion = emotions_dict.get(argmax_audio_prediction, "Unknown")		
		print_emotions("Audio prediction", audio_emotion_probabilities)


		print("Preprocessing audio for text-based model")
		sentence = prepare_sentence_for_text_model(os.path.normpath(output_audio_filename))		

		if sentence != None:
			vectorizer = TextVectorization(max_tokens=max_features, output_mode="int", output_sequence_length=sequence_length)
			text_vector = vectorizer(sentence)
			print(sentence)
			print(text_vector)

			text_predictions = np.squeeze(text_model.predict(text_vector, batch_size=32))
			text_emotion_probabilities = [(emotions_dict[i], text_predictions[i]) for i in range(len(text_predictions))]

			print_emotions("Text prediction", text_emotion_probabilities)

		counter += 1
		time.sleep(seconds)



if __name__ == "__main__":

	usage_message = """"
Usage of demo script.
> python demo.py [audio model path] [text model path]
"""

	if len(sys.argv) != 3:
		print(usage_message)
		exit(0)
		
	audio_model_path = sys.argv[1]
	text_model_path = sys.argv[2]
	run(audio_model_path, text_model_path) 
