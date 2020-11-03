#!/usr/bin/env python

import time
import keyboard 
import keras

import sounddevice as sd
import numpy as np
import pandas as pd

from scipy.io.wavfile import write
from librosa.feature import mfcc


ascii_art = """
        _..._
      .'     '.      _
     /    .-""-\\   _/ \
   .-|   /:.   |  |   |
   |  \\  |:.   /.-'-./
   | .-'-;:__.'    =/
   .'=  *=|ESA  _.='
  /   _.  |    ;
 ;-.-'|    \\   |
/   | \\    _\\  _\
\\__/'._;.  ==' ==\
         \\    \\   |
         /    /   /
         /-._/-._/
         \\   `\\  \
          `-._/._/

"""

channels = 1 # recording monophonic audio
fs = 44100
seconds = 5
output_filename = "../data/output_{:0>4}.wav"


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

def run(audio_model_filename, text_model_filename):

	pause = 10
	counter = 1

	with open("ascii_astronaut.txt") as f:
		print(f.read())

	print("\n\n\nLoading models")

	audio_model = keras.models.load_model(audio_model_filename)
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
		write(output_filename.format(counter), fs, myrecording)

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
	audio_model_filename = "../models/tone_cnn_8_emotions/saved_models/Emotion_Voice_Detection_Model.h5"
	run(audio_model_filename, "hoale")
