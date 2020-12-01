"""
Copyright (c) 2020 TU/e - PDEng Software Technology C2019. All rights reserved. 
@ Authors: Niels Rood n.rood@tue.nl; Jorge Cordero j.a.cordero.cruz@tue.nl;
@ Contributors: Jobaer Khan j.i.khan@tue.nl; 
Description: Class that performs audio analysis using different sentiment analyzers.
Last modified: 01-12-2020
"""

import os
import numpy as np
from pathlib import Path
from stern_utils import Utils
from pydub import AudioSegment
from scipy.io.wavfile import write


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
        self.emotions = parameters['emotions']

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
        tone_emotion_probabilities = []
        i = 0
        for key in self.emotions:
            tone_emotion_probabilities.append((self.emotions[key], tone_predictions[i]))
            i = i + 1
        self.emotion_logbook(tone_emotion_probabilities)
        return tone_emotion_probabilities
