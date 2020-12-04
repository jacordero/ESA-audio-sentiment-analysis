"""
Copyright (c) 2020 TU/e - PDEng Software Technology C2019. All rights reserved. 
@ Authors: Raha Sadeghi r.sadeghi@tue.nl; Parima Mirshafiei p.mirshafiei@tue.nl; Jorge Cordero j.a.cordero.cruz@tue.nl;
Last modified: 01-12-2020
"""

from keras.layers import Conv1D, Dense, Dropout, Activation, Flatten, MaxPooling1D
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential

class SequentialThreeConvModulesGenerator():
    """
    This class creates sequential architecture with three convolutional modules.
    """
    def generate_model(self, n_conv_filters, filters_shape, _input_shape, n_emotions):
        """creates a sequential model with three modules

        Args:
            n_conv_filters: an array indicating number of neurons of each layer respectively
            filters_shape: an array indicating filter shapes of each layer respectively
            _input_shape: the input shape of the convolution layers
            n_emotions: number of emotions to predict

        Returns:
            sequential model with three convolutional modules
        """
        model = Sequential()
        model.add(Conv1D(n_conv_filters[0], filters_shape[0], padding='same',
                         input_shape=_input_shape))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.2))
        model.add(MaxPooling1D(1))

        model.add(Conv1D(n_conv_filters[1], filters_shape[1], padding='same',
                         input_shape=_input_shape))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.3))
        model.add(MaxPooling1D(4))

        model.add(Conv1D(n_conv_filters[2], filters_shape[2], padding='same',
                         input_shape=_input_shape))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.2))
        model.add(MaxPooling1D(2))

        model.add(Flatten())
        model.add(Dense(n_emotions))
        model.add(Dense(n_emotions))
        model.add(Dense(n_emotions))
        model.add(Activation('softmax'))
        return model

  
class SequentialTwoConvModulesGenerator():
    """
    This class creates SeqentialModel architecture with two convolutional modules.
    """
    def generate_model(self, n_conv_filters, filters_shape, _input_shape, n_emotions):
        """creates a sequential model with two layers

        Args:
            n_conv_filters: an array indicating number of neurons of each layer respectively
            filters_shape: an array indicating filter shapes of each layer respectively
            _input_shape: the input shape of the convolution layers
            n_emotions: number of emotions to predict

        Returns:
            sequential model with two convolutional modules
        """
        model = Sequential()
     
        model.add(Conv1D(n_conv_filters[0], filters_shape[0], padding='same',
                        input_shape=_input_shape))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.2))

        model.add(Conv1D(n_conv_filters[1], filters_shape[1], padding='same',
                        input_shape=_input_shape))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.2))

        model.add(Flatten())
        model.add(Dense(n_emotions))
        model.add(Dense(n_emotions))
        model.add(Dense(n_emotions))
        model.add(Activation('softmax'))
        return model

class SequentialOneConvModuleGenerator():
    """
    This class creates sequential architecture with one convolutional module.
    """

    def generate_model(self, n_conv_filters, filters_shape, _input_shape, n_emotions):
        """creates a sequential model with one convolutional module

        Args:
            n_conv_filters: an array indicating number of neurons of each layer respectively
            filters_shape: an array indicating filter shapes of each layer respectively
            _input_shape: the input shape of the convolution layers
            n_emotions: number of emotions to predict

        Returns:
            sequential model with one convolutional module
        """
        model = Sequential()

        model.add(Conv1D(n_conv_filters[0], filters_shape[0], padding='same',
                        input_shape=_input_shape))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.2))

        model.add(Flatten())
        model.add(Dense(n_emotions))
        model.add(Dense(n_emotions))
        model.add(Dense(n_emotions))
        model.add(Activation('softmax'))
        return model