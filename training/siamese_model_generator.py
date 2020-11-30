
from keras.layers import SpatialDropout2D, Activation, Input, Concatenate, Dense
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout
import keras

class SiameseModel():

    def create_siamese_branch_architecture(self, input_shape_x, input_shape_y):
        """
        creating a general model to be used for both mffc and lmfe
        """

        input_ = Input(shape=[input_shape_x, input_shape_y, 1])
        hidden1 = Conv2D(128, kernel_size=5, activation='relu',
                            padding='same',
                            input_shape=(input_shape_x, input_shape_y, 1))(input_)
        bn1 = BatchNormalization()(hidden1)
        pooling1 = MaxPooling2D(pool_size=(2, 2))(bn1)
        dp1 = SpatialDropout2D(0.2)(pooling1)

        hidden2 = Conv2D(256, kernel_size=3, activation='relu',
                            padding='same')(dp1)
        bn2 = BatchNormalization()(hidden2)
        pooling2 = MaxPooling2D(pool_size=(2, 2))(bn2)
        dp2 = SpatialDropout2D(0.2)(pooling2)

        hidden3 = Conv2D(512, kernel_size=3, activation='relu',
                            padding='same')(dp2)
        bn3 = BatchNormalization()(hidden3)
        pooling3 = MaxPooling2D(pool_size=(2, 2))(bn3)
        dp3 = SpatialDropout2D(0.2)(pooling3)

        flatten = Flatten()(dp3)
        dense = Dense(256, activation='relu')(flatten)
        bn = BatchNormalization()(dense)

        return input_, bn

    def generate_model(self, branch1, branch2, input1, input2):
        """
        creating the final model by concatinating two branches (mfcc and lmfe)
        :param branch1: the first branch in Siamese model
        :param branch2: the second branch in Siamese model
        :param input1: the first input for Siamese model
        :param input2: the second input for Siamese model
        """
        concat_ = Concatenate()([branch1, branch2])
        output = Dense(8, activation='softmax')(concat_)
        model = keras.Model(inputs=[input1, input2], outputs=[output])
        return model
 

if __name__ == "__main__":

    n_conv_filters = [32, 128, 256]
    filters_shape = [5, 5, 5]
    input_shape = (50, 1)

    siamese_model = SiameseModel()

    mfcc_input, mfcc_output = siamese_model.create_siamese_branch_architecture()
    lmfe_input, lmfe_output = siamese_model.create_siamese_branch_architecture()
    model = siamese_model.generate_model(mfcc_output, lmfe_output, mfcc_input, lmfe_input)
    print(model.summary())

    '''
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
    '''
