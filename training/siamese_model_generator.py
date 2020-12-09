"""
Copyright (c) 2020 TU/e - PDEng Software Technology C2019. All rights reserved. 
@ Authors: Raha Sadeghi r.sadeghi@tue.nl; Parima Mirshafiei p.mirshafiei@tue.nl;
Last modified: 01-12-2020
"""

from keras.layers import SpatialDropout2D, Activation, Input, Concatenate, Dense
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout
import keras

class SiameseModelGenerator():
    """
    Create a Siamese model which is a kind of parallel model, with two inputs: "mfcc" and "lmfe"
    This model is based on this paper [https://link.springer.com/chapter/10.1007%2F978-3-030-51999-5_18]
    """
    def create_siamese_branch_architecture(self, n_conv_filters, filters_shape,  input_shape_x, input_shape_y):
        """creating a the model architecture of each branch, to be used for both mffc and lmfe

        Args:
            n_conv_filters: default values based on the paper are 128, 256, 512
            filters_shape: default values based on the paper are [3,3,3]
            input_shape_x: the input shape of features
            input_shape_y: the input shape of labels

        Returns:
            Input and ouput tensors of the branch
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

    def concatenate_models(self, branch1, branch2, input1, input2, n_emotions):
        """Creating the final model by concatinating two branches (mfcc and lmfe)

        Args:
            branch1: the model created based on mfcc feature
            branch2: the model created based on lmfe feature
            input1: the first input for Siamese model
            input2: the second input for Siamese model
            n_emotions: number of emotions to predict

        Returns:
            concatenated model
        """        
        
        concat_ = Concatenate()([branch1, branch2])
        output = Dense(n_emotions, activation='softmax')(concat_)
        model = keras.Model(inputs=[input1, input2], outputs=[output])
        return model

    def generate_model(self, n_conv_filters, filters_shape, _input_shape, n_emotions):
        """Create a Siamese model.

        Args:
            n_conv_filters : neurons should we used in each layer of branches
            filters_shape: filter shapes of the first layer
            _input_shape: the input shape is an array with the first two paramters define the mfcc shape (X, y) 
                and the last two parameters define the lmfe shape (X, y)
            n_emotions: number of emotions to predict

        Returns:
            Siamese model
        """
        mfcc_input, mfcc_output = self.create_siamese_branch_architecture(n_conv_filters, filters_shape, _input_shape[0], _input_shape[1])
        lmfe_input, lmfe_output = self.create_siamese_branch_architecture(n_conv_filters, filters_shape, _input_shape[2], _input_shape[3])
        model = self.concatenate_models(mfcc_output, lmfe_output, mfcc_input, lmfe_input, n_emotions)
        return model


class TwoLayerSiameseModelGenerator():

    def create_siamese_branch_architecture(self, n_conv_filters, filters_shape,  input_shape_x, input_shape_y):
        """creating a the model architecture of each branch, to be used for both mffc and lmfe

        Args:
            n_conv_filters: default values based on the paper are 128, 256, 512
            filters_shape: default values based on the paper are [3,3,3]
            input_shape_x: the input shape of features
            input_shape_y: the input shape of labels

        Returns:
            Input and ouput tensors of the branch
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

    def concatenate_models(self, branch1, branch2, input1, input2, n_emotions):
        """Creating the final model by concatinating two branches (mfcc and lmfe)

        Args:
            branch1: the model created based on mfcc feature
            branch2: the model created based on lmfe feature
            input1: the first input for Siamese model
            input2: the second input for Siamese model
            n_emotions: number of emotions to predict

        Returns:
            concatenated model
        """        
        concat_ = Concatenate()([branch1, branch2])
        output = Dense(n_emotions, activation='softmax')(concat_)
        model = keras.Model(inputs=[input1, input2], outputs=[output])
        return model

    def generate_model(self, n_conv_filters, filters_shape, _input_shape, n_emotions):
        """Create a Siamese model.

        Args:
            n_conv_filters : neurons should we used in each layer of branches
            filters_shape: filter shapes of the first layer
            _input_shape: the input shape is an array with the first two paramters define the mfcc shape (X, y) 
                and the last two parameters define the lmfe shape (X, y)
            n_emotions: number of emotions to predict

        Returns:
            Siamese model
        """
        mfcc_input, mfcc_output = self.create_siamese_branch_architecture(n_conv_filters, filters_shape, _input_shape[0], _input_shape[1])
        lmfe_input, lmfe_output = self.create_siamese_branch_architecture(n_conv_filters, filters_shape, _input_shape[2], _input_shape[3])
        model = self.concatenate_models(mfcc_output, lmfe_output, mfcc_input, lmfe_input, n_emotions)
        return model
 
