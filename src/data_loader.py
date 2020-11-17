import librosa
import numpy as np
import os
import pandas as pd

def list_audios(data_directory):
    audios = []
    audio_formats = ['wav']

    for filename in os.listdir(data_directory):
        audio_format = filename.split('.')[1]
        if audio_format in audio_formats:
            audios.append(data_directory + "/" + filename)

    return audios


def load_audio(filename, audio_length=5, sample_rate=44010, _n_mfcc=40):
    """[summary]

    Args:
        filename ([type]): [description]
        audio_length (int, optional): [description]. Defaults to 5.
        sample_rate (int, optional): [description]. Defaults to 44010.
        _n_mfcc (int, optional): [description]. Defaults to 40.

    Raises:
        Exception: [description]

    Returns:
        [type]: [description]
    """

    #TODO: also load the corresponding labels    

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


def load_tone_data(audio_directory):
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
