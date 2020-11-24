
from keras.layers import Conv1D, Dense, Dropout, Activation, Flatten, MaxPooling1D
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential

class SequentialThreeConvModulesGenerator():

    def generate_model(self, n_conv_filters, filters_shape, _input_shape):
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
        model.add(Dense(8))
        model.add(Dense(8))
        model.add(Dense(8))
        model.add(Activation('softmax'))
        return model

  
class SequentialTwoConvModulesGenerator():

    def generate_model(self, n_conv_filters, filters_shape, _input_shape):
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
        model.add(Dense(8))
        model.add(Dense(8))
        model.add(Dense(8))
        model.add(Activation('softmax'))
        return model

class SequentialOneConvModuleGenerator():

    def generate_model(self, n_conv_filters, filters_shape, _input_shape):
        model = Sequential()

        model.add(Conv1D(n_conv_filters[0], filters_shape[0], padding='same',
                        input_shape=_input_shape))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.2))

        model.add(Flatten())
        model.add(Dense(8))
        model.add(Dense(8))
        model.add(Dense(8))
        model.add(Activation('softmax'))
        return model

class SequentialModelGeneratorFactory():

    def __init__(self):
        self.generators = {
            'sequential_three_conv_modules_generator': SequentialThreeConvModulesGenerator,
            'sequential_two_conv_modules_generator': SequentialTwoConvModulesGenerator,
            'sequential_one_conv_module_generator': SequentialOneConvModuleGenerator
        }

    def get_model_generator(self, model_name):
        return self.generators[model_name]()

if __name__ == "__main__":

    n_conv_filters = [32, 128, 256]
    filters_shape = [5, 5, 5]
    input_shape = (50, 1)

    model_generator_factory = SequentialModelGeneratorFactory()
    
    print("Sequential model with 3 convolutional modules")
    model_generator = model_generator_factory.get_model_generator('sequential_three_conv_modules_generator')
    model = model_generator.generate_model(n_conv_filters, filters_shape, input_shape)
    print(model.summary())

    print("\n\n\nSequential model with 2 convolutional modules")
    model_generator = model_generator_factory.get_model_generator('sequential_two_conv_modules_generator')
    model = model_generator.generate_model(n_conv_filters, filters_shape, input_shape)
    print(model.summary())

    print("\n\n\nSequential model with 1 convolutional module")
    model_generator = model_generator_factory.get_model_generator('sequential_one_conv_module_generator')
    model = model_generator.generate_model(n_conv_filters, filters_shape, input_shape)
    print(model.summary())
