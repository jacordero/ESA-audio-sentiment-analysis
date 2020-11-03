#!/usr/bin/env python

import time
import sys
import os
import keyboard 
import keras

import sounddevice as sd
import numpy as np
import pandas as pd

from scipy.io.wavfile import write
from librosa.feature import mfcc

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


def prepare_sentences_for_text_model(audio_array, fs):
	return None


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
	print("***************************************\n")

def run(audio_model_path, text_model_path):

	pause = 10
	counter = 1

	parent_dir = os.path.dirname(os.path.abspath(__file__))
	print(parent_dir)

	with open(os.path.join(parent_dir, "ascii_astronaut.txt")) as f:
		print(f.read())

	print("\n\n\nLoading models")

	audio_model = keras.models.load_model(audio_model_path)
	text_model = load_text_model()

	print("Starting audio demo!")

	while True:


		print("To record for five seconds, press Enter.")
		print("To stop program, press q.")
		key_pressed = keyboard.read_key()
		if (key_pressed == "q"):
			exit(0)

		#recording message
		myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=channels)
		sd.wait()

		output_audio_filename = os.path.join(parent_dir, output_audio_filename_template.format(counter))
		write(output_audio_filename, fs, myrecording)

		print("Finished recording. Start playing message")	
		sd.play(myrecording, fs)

		print("Preprocessing audio for tone model")
		preprocessed_audio = prepare_audio_for_tone_model(np.squeeze(myrecording), fs)

		print("Preprocessing audio for text-based model")
		sentences = prepare_sentences_for_text_model(myrecording, fs) 

		## predictions happen here
		audio_predictions = np.squeeze(audio_model.predict(preprocessed_audio))
		audio_emotion_probabilities = [(emotions_dict[i], audio_predictions[i]) for i in range(len(audio_predictions))]

		argmax_audio_prediction = np.argmax(np.squeeze(audio_predictions), axis=-1)
		predicted_emotion = emotions_dict.get(argmax_audio_prediction, "Unknown")
		text_prediction = ""

		print_emotions("Audio prediction", audio_emotion_probabilities)
		print("Text prediction: {}".format(text_prediction))


		#print(myrecording)
		counter += 1

		print("Wait 10 seconds before recording new message")
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
