from keras.layers import SpatialDropout2D, Activation, Input, Concatenate, Dense
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout
import keras


__authors__ = "Raha Sadeghi, Parima Mirshafiei",
__email__ = "r.sadeghi@tue.nl; P.mirshafiei@tue.nl;"
__copyright__ = "TU/e ST2019"


class SiameseModel():
    """
    create a Siamese model which is a kind of parallel model, with two inputs: "mfcc" and "lmfe"
    This model is based on this paper [https://link.springer.com/chapter/10.1007%2F978-3-030-51999-5_18]
    """
    def create_siamese_branch_architecture(self, n_conv_filters, filters_shape,  input_shape_x, input_shape_y):
        """
        creating a the model architecture of each branch, to be used for both mffc and lmfe
        :param n_conv_filters: default values based on the paper are 128, 256, 512
        :param filters_shape: default values based on the paper are [3,3,3]
        :param input_shape_x: the input shape of features
        :param input_shape_y: the input shape of labels
        """

        input_ = Input(shape=[input_shape_x, input_shape_y, 1])
        hidden1 = Conv2D(n_conv_filters[0], kernel_size=filters_shape[0], activation='relu',
                            padding='same',
                            input_shape=(input_shape_x, input_shape_y, 1))(input_)
        bn1 = BatchNormalization()(hidden1)
        pooling1 = MaxPooling2D(pool_size=(2, 2))(bn1)
        dp1 = SpatialDropout2D(0.2)(pooling1)

        hidden2 = Conv2D(n_conv_filters[1], kernel_size=filters_shape[1], activation='relu',
                            padding='same')(dp1)
        bn2 = BatchNormalization()(hidden2)
        pooling2 = MaxPooling2D(pool_size=(2, 2))(bn2)
        dp2 = SpatialDropout2D(0.2)(pooling2)

        hidden3 = Conv2D(n_conv_filters[2], kernel_size=filters_shape[2], activation='relu',
                            padding='same')(dp2)
        bn3 = BatchNormalization()(hidden3)
        pooling3 = MaxPooling2D(pool_size=(2, 2))(bn3)
        dp3 = SpatialDropout2D(0.2)(pooling3)

        flatten = Flatten()(dp3)
        dense = Dense(256, activation='relu')(flatten)
        bn = BatchNormalization()(dense)

        return input_, bn

    def concatinate_models(self, branch1, branch2, input1, input2):
        """
        creating the final model by concatinating two branches (mfcc and lmfe)
        :param branch1: the model created based on mfcc feature
        :param branch2: the model created based on lmfe feature
        :param input1: the first input for Siamese model
        :param input2: the second input for Siamese model
        """
        concat_ = Concatenate()([branch1, branch2])
        output = Dense(5, activation='softmax')(concat_)
        model = keras.Model(inputs=[input1, input2], outputs=[output])
        return model

    def generate_model(self, n_conv_filters, filters_shape, _input_shape):
        """
        create a Siamese model 
        :param: n_conv_filter:  nuerons should we used in each layer of branches
        :param: filters_shape:  filter shapes of the first layer
        :param: _input_shape: the input shape is an array with the first two paramters define the mfcc shape (X, y) 
        and the last two parameters define the lmfe shape (X, y)
        """
        mfcc_input, mfcc_output = self.create_siamese_branch_architecture(n_conv_filters, filters_shape, _input_shape[0], _input_shape[1])
        lmfe_input, lmfe_output = self.create_siamese_branch_architecture(n_conv_filters, filters_shape, _input_shape[2], _input_shape[3])
        model = self.concatinate_models(mfcc_output, lmfe_output, mfcc_input, lmfe_input)
        return model


class TwoLayerSiameseModel():

    def create_siamese_branch_architecture(self, n_conv_filters, filters_shape,  input_shape_x, input_shape_y):
        """
        creating a general model to be used for both mffc and lmfe
        :param n_conv_filters: default values based on the paper are 128, 256, 512
        :param filters_shape: default values based on the paper are [3,3,3]
        :param input_shape_x: the input shape of features
        :param input_shape_y: the input shape of labels
        """

        input_ = Input(shape=[input_shape_x, input_shape_y, 1])
        hidden1 = Conv2D(n_conv_filters[0], kernel_size=filters_shape[0], activation='relu',
                            padding='same',
                            input_shape=(input_shape_x, input_shape_y, 1))(input_)
        bn1 = BatchNormalization()(hidden1)
        pooling1 = MaxPooling2D(pool_size=(2, 2))(bn1)
        dp1 = SpatialDropout2D(0.2)(pooling1)

        hidden2 = Conv2D(n_conv_filters[1], kernel_size=filters_shape[1], activation='relu',
                            padding='same')(dp1)
        bn2 = BatchNormalization()(hidden2)
        pooling2 = MaxPooling2D(pool_size=(2, 2))(bn2)
        dp2 = SpatialDropout2D(0.2)(pooling2)

        flatten = Flatten()(dp2)
        dense = Dense(256, activation='relu')(flatten)
        bn = BatchNormalization()(dense)

        return input_, bn

    def concatinate_models(self, branch1, branch2, input1, input2):
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
        
    def generate_model(self, n_conv_filters, filters_shape, _input_shape):
        """
        create a Siamese model 
        :param: n_conv_filter:  nuerons should we used in each layer of branches
        :param: filters_shape:  filter shapes of the first layer
        :param: _input_shape: the input shape is an array with the first two paramters define the mfcc shape (X, y) 
        and the last two parameters define the lmfe shape (X, y)
        """
        mfcc_input, mfcc_output = self.create_siamese_branch_architecture(n_conv_filters, filters_shape, _input_shape[0], _input_shape[1])
        lmfe_input, lmfe_output = self.create_siamese_branch_architecture(n_conv_filters, filters_shape, _input_shape[2], _input_shape[3])
        model = self.concatinate_models(mfcc_output, lmfe_output, mfcc_input, lmfe_input)
        return model
 
