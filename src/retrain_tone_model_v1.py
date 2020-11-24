"""
Copyright (C) Tu/e.

@author Tuvi Purevsuren t.purevsuren@tue.nl

======================================================

"""

#!/usr/bin/env python

import os
import sys
import time

import librosa
import numpy as np

import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

def get_path(dir_name):
    """Find path of requested directory such as 'models' and 'data'.

    Args:
      dir_name: A name of directory.

    Returns:
      The full path of requested directory.

    """
    path = os.path.join(os.path.dirname(os.path.normpath(os.getcwd())), dir_name)
    return path

def load_model(model_name):
    """Load the pre-trained tone model from model directory.

    Args:
      model_name: A name of model to be loaded.

    Returns:
      The pre-trained tone model.

    """
    model_path = os.path.join(get_path('models'), model_name)
    files = [files for files in os.listdir(model_path) if files.endswith(".h5")]
    if not files:
      files = [files for files in os.listdir(model_path) if files.endswith(".pb")]
      if not files:
        raise Exception("NO MODEL FOUND")
      else:
        # load .pb model (tensorflow 2.2.0)
        model = tf.keras.models.load_model(model_path)
        return model
    else:
      # load .h5 model (tensorflow 1.x.x)
        model_path = os.path.join(model_path, files[0])
        model = tf.keras.models.load_model(model_path)
        return model

def save_model(model, version):
    """Save the retrained tone model in models directory.

    Args:
      model: the retrained tone model.
      version: the version to save retrained model.

    """
    model_path = os.path.join(get_path('models'), '{name}_v{version}'.format(name='retrained_tone_model', version=version))
    model.save(model_path)
    print("The retrained model is saved at {}.".format(model_path))

def prepare_datasets(dataset_name, n_mfcc):
    """Load the dataset from data directory.

    Args:
      dataset_name: A name of dataset to be loaded.
      n_mfcc: parameter for mfcc

    Returns:
      X as mfcc extracted feature of audio.
      y as emotions labels.

    """
    lst = []
    data_path = os.path.join(get_path('data'), dataset_name)

    start_time = time.time()
    files = os.listdir(data_path)
    
    for root, _, files in os.walk(data_path):
        for audio_file in files:
            try:
                # load librosa array, obtain mfcss, store the mfcc information in a new array
                X, sample_rate = librosa.load(os.path.join(root, audio_file), res_type='kaiser_fast')

                mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=n_mfcc).T, axis=0)
                # convert labels from (1 ->8) to (0->7)
                # predict script has emotion index (0 to 7)
                audio_file = int(audio_file[7:8])-1
                arr = mfccs, audio_file
                lst.append(arr)
            # If file is invalid, skip it.
            except ValueError as err:
                print("ERROR ", err)
                continue
    print('======= Audio files Loading time: {time} seconds=========='.format(time =(time.time() - start_time)))
    X, y = zip(*lst)

    # convert into arrays
    X = np.asarray(X)
    y = np.asarray(y)
    print("TOTAL AUDIO FILE: ", len(X))
    return X, y

def retrain_model(parameters):
    """Retrain the tone model with dataset.

    Args:
      parameters: A set of parameters.

    """
    model = load_model(parameters['model_name'])
    model.summary()

    X, y = prepare_datasets(parameters['dataset_name'], parameters['n_mfcc'])
    X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=parameters['ratio_of_validation'], random_state=parameters['shuffle_random_seed'])

    x_train_cnn = np.expand_dims(X_train, axis=parameters['axis'])
    x_validation_cnn = np.expand_dims(X_validation, axis=parameters['axis'])

    opt = tf.keras.optimizers.Adam(learning_rate=parameters['learning_rate'])
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    start_time = time.time()
    model.fit(x_train_cnn, y_train,
              batch_size=parameters['batch_size'], epochs=parameters['epochs'],
              validation_data=(x_validation_cnn, y_validation))

    print('======= Training time: {time} seconds with {epochs} epochs. ======='.format(time = (time.time()- start_time), epochs = parameters['epochs']))
    
    save_model(model, parameters['retrained_model_version'])

if __name__ == "__main__":
    print("==== WELCOME to Sentiment Analysis for Astronauts System ====")
    parameters = {
      # resource related parameters
      'model_name': sys.argv[1], 
      'dataset_name': sys.argv[2], 
      'retrained_model_version': sys.argv[3],
      # dataset related parameters
      'ratio_of_validation': 0.33,
      'shuffle_random_seed': 42, 
      'axis': 2, 
      'n_mfcc': 50,
      # retrain related parameters 
      'batch_size': 8, 
      'epochs': 1,
      'learning_rate': 0.001,
    }
    retrain_model(parameters)