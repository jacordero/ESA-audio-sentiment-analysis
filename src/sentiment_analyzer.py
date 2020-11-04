import keras
from keras.models import model_from_json
import json
import librosa
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

import sys
import os


def load_model(model_weights_filename, model_architecture_filename):
	# load model architecture
	json_file = open(model_architecture_filename, 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)

	# load weights into new model
	loaded_model.load_weights(model_weights_filename)
	print("Loaded model from disk")
	print(loaded_model.summary())
	return loaded_model

def list_audios(data_directory):
	audios = []
	audio_formats = ['wav']

	for filename in os.listdir(data_directory):
		audio_format = filename.split('.')[1]
		if audio_format in audio_formats:
			audios.append(data_directory + "/" + filename)

	return audios


def load_audio(filename):

	X, sample_rate = librosa.load(filename, 
		res_type='kaiser_fast',
		duration=2.5,
		sr=22050*2,
		offset=0.5)

	sample_rate = np.array(sample_rate)
	mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13), axis=0)
	featurelive = mfccs
	livedf2 = featurelive
	livedf2= pd.DataFrame(data=livedf2)
	livedf2 = livedf2.stack().to_frame().T
	twodim= np.expand_dims(livedf2, axis=2)

	return twodim


def run(model_directory, data_directory):

	model_weights_filename = model_directory + "/weights.h5"
	model_architecture_filename = model_directory + "/model_architecture.json"
	audio_encoder_filename =  model_directory + "/encoder.npy"

	model = load_model(model_weights_filename, model_architecture_filename)
	audio_filenames = list_audios(data_directory)
	print (audio_filenames)

	for audio_filename in audio_filenames:
		audio = load_audio(audio_filename)

		lb = LabelEncoder()
		lb.classes_ = np.load(audio_encoder_filename, allow_pickle=True)

		livepreds = model.predict(audio, batch_size=32, verbose=1)
		livepreds1 = livepreds.argmax(axis=1)
		liveabc = livepreds1.astype(int).flatten()

		livepredictions = (lb.inverse_transform((liveabc)))
		print(livepredictions)


if __name__ == "__main__":

	usage_message = """"
Usage of sentiment_analyzer script.
> python sentiment_analyzer.py [model directory] [data directory]
"""

	if len(sys.argv) != 3:
		print(usage_message)
		exit(0)
		
	model_directory = sys.argv[1]
	data_directory = sys.argv[2]
	run(model_directory, data_directory) 
