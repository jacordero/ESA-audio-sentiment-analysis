import librosa
import numpy as np
import os
import pandas as pd

tone_emotions = {0: 'neutral',
                 1: 'calm',
                 2: 'happy',
                 3: 'sad',
                 4: 'angry',
                 5: 'fearful',
                 6: 'disgust',
                 7: 'surprised'}

text_emotions = {0: 'neutral',
                 7: 'calm',
                 1: 'happy',
                 2: 'sad',
                 3: 'angry',
                 4: 'fearful',
                 5: 'disgust',
                 6: 'surprised'}


def list_audios(data_directory):
    audios = []
    audio_formats = ['wav']

    for filename in os.listdir(data_directory):
        audio_format = filename.split('.')[1]
        if audio_format in audio_formats:
            audios.append(data_directory + "/" + filename)

    return audios


def load_audio(filename, audio_length=5, sample_rate=44010, _n_mfcc=40):
    #raise Exception("TODO")

    X, sample_rate = librosa.load(filename,
                                  res_type='kaiser_fast',
                                  duration=audio_length,
                                  sr=sample_rate,
                                  offset=0.5)
    
    sample_rate = np.array(sample_rate)
    mfccs = np.mean(librosa.feature.mfcc(
        y=X, sr=sample_rate, n_mfcc=_n_mfcc), axis=0)
    
    featurelive = mfccs
    livedf2 = featurelive
    livedf2 = pd.DataFrame(data=livedf2)
    livedf2 = livedf2.stack().to_frame().T
    twodim = np.expand_dims(livedf2, axis=2)

    return twodim


def load_audios(audio_directory):
    """[summary]

    Args:
        audio_directory ([type]): [description]
    """

    audio_filenames = list_audios(audio_directory)
    print(audio_filenames)

    audios = []
    for audio_filename in audio_filenames:
        audios.append(load_audio(audio_filename))

    return audios
