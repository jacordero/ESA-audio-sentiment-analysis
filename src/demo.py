#!/usr/bin/env python

import time
import keyboard 

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



def load_audio_model():
	return None


def load_text_model():
	return None

def prepare_audio_for_tone_model(audio_array, fs):
	
	sample_rate = np.array(fs)
	mfccs = np.mean(mfcc(y=audio_array, sr=np.array(sample_rate), n_mfcc=13), axis=0)

	featurelive = mfccs
	livedf2 = featurelive
	livedf2= pd.DataFrame(data=livedf2)
	livedf2 = livedf2.stack().to_frame().T
	twodim= np.expand_dims(livedf2, axis=2)

	return twodim

def prepare_sentences_for_text_model(audio_array, fs):
	return None


def run():

	pause = 10
	counter = 1

	with open("ascii_astronaut.txt") as f:
		print(f.read())

	print("Loading models")

	audio_model = load_audio_model()
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
		audio_for_tone_model = prepare_audio_for_tone_model(np.squeeze(myrecording), fs)

		print("Preprocessing audio for text-based model")
		sentences = prepare_sentences_for_text_model(myrecording, fs) 

		## predictions happen here
		audio_prediction = ""
		text_prediction = ""


		print("Audio prediction: {}".format(audio_prediction))
		print("Text prediction: {}".format(text_prediction))


		print(myrecording)
		counter += 1

		print("Wait 10 seconds before recording new message")
		time.sleep(seconds)



if __name__ == "__main__":
	run()
