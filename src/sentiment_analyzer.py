import os
import numpy as np
from stern_utils import Utils

class AudioAnalyzer:
	""" This class is used to analyze audio recordings using tone model predictors.
	"""
	def __init__(self, tone_predictor, parameters):
		self.frame_rate = parameters['audio_frequency']
		self.tone_predictor = tone_predictor
		# Logging functionality
		self.logging_file_prefix = parameters['logging_file_prefix']
		script_dir = os.path.dirname(os.path.abspath(__file__))
		self.logging_directory = os.path.normpath(
		    os.path.join(script_dir, parameters['logging_directory']))
		self.__create_log_directory()
		self.emotions=parameters['emotions']

	def __create_log_directory(self):
		"""Creates a logging directory if it does not exists yet.
		"""
		if not os.path.exists(self.logging_directory):
			os.makedirs(self.logging_directory)

	def emotion_logbook(self, audio_emotion_probabilities):
		"""Writes utterances of the audio and detected emotions in logbook.

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
		"""Predicts emotions in an audio using tone based sentiment prediction models.

		Args:
			audio : Audio to be analyzed.

		Returns:
			List containing pairs of sentiments and their corresponding predictions.
		"""		
		tone_predictions = self.tone_predictor.predict(np.squeeze(audio))
		print(tone_predictions)
		tone_emotion_probabilities = []
			# (self.emotions[0], tone_predictions[0]) for i in range(len(tone_predictions))]
		i=0
		for key in self.emotions:
			tone_emotion_probabilities.append((self.emotions[key],tone_predictions[i]))
			i=i+1
		print(tone_emotion_probabilities)
		self.emotion_logbook(tone_emotion_probabilities)
		return tone_emotion_probabilities

