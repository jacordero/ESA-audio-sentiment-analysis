
from keras.layers import Conv1D, Dense, Dropout, Activation, Flatten, MaxPooling1D
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential


__authors__ = "Raha Sadeghi, Parima Mirshafiei, Jorge Cordero",
__email__ = "r.sadeghi@tue.nl; P.mirshafiei@tue.nl; j.a.cordero.cruz@tue.nl;"
__copyright__ = "TU/e ST2019"


class SequentialThreeConvModulesGenerator():
    """
    This class creates SeqentialModel architecture with three convolutional layers.
    """
    def generate_model(self, n_conv_filters, filters_shape, _input_shape):
        """
        create a sequential model with three layers
        :param: n_conv_filter: an array indicating number of neurons of each layer respectively
        :param: filters_shape: an array indicating filter shapes of each layer respectively
        :param: _input_shape: the input shape of the convolution layers
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
        model.add(Dense(5))
        model.add(Dense(5))
        model.add(Dense(5))
        model.add(Activation('softmax'))
        return model

  
class SequentialTwoConvModulesGenerator():
    """
    This class creates SeqentialModel architecture with two convolutional layers .
    """
    def generate_model(self, n_conv_filters, filters_shape, _input_shape):
        """
        create a sequential model with three layers
        :param: n_conv_filter: an array indicating number of neurons of each layer respectively
        :param: filters_shape: an array indicating filter shapes of each layer respectively
        :param: _input_shape: the input shape of the convolution layers
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
        model.add(Dense(5))
        model.add(Dense(5))
        model.add(Dense(5))
        model.add(Activation('softmax'))
        return model

class SequentialOneConvModuleGenerator():
    """
    This class creates SeqentialModel architecture with three convolutional layers.
    """

    def generate_model(self, n_conv_filters, filters_shape, _input_shape):
        """
        create a sequential model with one layer
        :param: n_conv_filter:  number of neurons of the first layer
        :param: filters_shape:  filter shapes of the first layer
        :param: _input_shape: the input shape of the convolution layer
        """
        model = Sequential()

        model.add(Conv1D(n_conv_filters[0], filters_shape[0], padding='same',
                        input_shape=_input_shape))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.2))

        model.add(Flatten())
        model.add(Dense(5))
        model.add(Dense(5))
        model.add(Dense(5))
        model.add(Activation('softmax'))
        return model