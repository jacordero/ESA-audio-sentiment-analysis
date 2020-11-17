import numpy as np
from librosa.feature import mfcc

def tone_model_preprocessing(audio_array, sample_rate, _n_mfcc):

    mfccs = np.mean(mfcc(y=audio_array, sr=np.array(
        sample_rate), n_mfcc=_n_mfcc).T, axis=0)
    x = np.expand_dims(mfccs, axis=0)
    x = np.expand_dims(x, axis=2)
    return x
