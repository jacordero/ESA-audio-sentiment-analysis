import os
import time
import librosa
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense
from keras.layers import Conv1D, MaxPooling1D, Conv2D, MaxPooling2D, concatenate
from keras.layers import Flatten
from keras.layers import Dropout, SpatialDropout2D
from keras.layers import Activation, Input, Concatenate
from keras.models import Sequential
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from pathlib import Path
import keras
from keras.layers.normalization import BatchNormalization
import json
import joblib
from us52_utility import UtilityClass
from us52_feature_extraction import FeatureExtractor
from tensorflow.keras.utils import plot_model

# Package metadata
__authors__ = "Raha Sadeghi, Parima Mirshafiei"
__email__ = "r.sadeghi@tue.nl; P.mirshafiei@tue.nl"
__copyright__ = "TU/e ST2019"
__version__ = "1.0"
__status__ = "Prototype"


def create_general_model(input_shape_x, input_shape_y):
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


'''
def create_model():
    model = Sequential()
    model.add(Conv1D(32, 5, padding='same',
                     input_shape=(50, 1)))
    model.add(BatchNormalization())
    model.add(activation('relu'))
    model.add(Dropout(0.2))

    model.add(Conv1D(128, 5, padding='same',
                     input_shape=(50, 1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(MaxPooling1D(pool_size=(10)))

    model.add(Conv1D(256, 5, padding='same',
                     input_shape=(50, 1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(8))
    model.add(Dense(8))
    model.add(Dense(8))
    model.add(Activation('softmax'))
    return model
'''


def create_final_model(branch1, branch2, input1, input2):
    """
    creating the final model by concatinating two branches (mfcc and lmfe)
    """
    concat_ = Concatenate()([branch1, branch2])
    output = Dense(8, activation='softmax')(concat_)
    model = keras.Model(inputs=[input1, input2], outputs=[output])
    return model


def train_neural_network(X, y):
    """
    This function trains the neural network.
    """

    X_mfcc_train, X_mfcc_test, y_mfcc_train, y_mfcc_test = train_test_split(X[0], y[0], test_size=0.33, random_state=42)
    X_lmfe_train, X_lmfe_test, y_lmfe_train, y_lmfe_test = train_test_split(X[1], y[1], test_size=0.33, random_state=42)

    # expanding dimensions, required for using conv2D
    x_mfcc_traincnn = np.expand_dims(X_mfcc_train, axis=3)
    x_mfcc_testcnn = np.expand_dims(X_mfcc_test, axis=3)

    # expanding dimensions, required for using conv2D
    x_lmfe_traincnn = np.expand_dims(X_lmfe_train, axis=3)
    x_lmfe_testcnn = np.expand_dims(X_lmfe_test, axis=3)

    print(x_lmfe_traincnn.shape, x_mfcc_testcnn.shape)

    mfcc_input, mfcc_output = create_general_model(13, 216)

    lmfe_input, lmfe_output = create_general_model(26, 498)

    model = create_final_model(mfcc_output, lmfe_output, mfcc_input, lmfe_input)

    print(model.summary)

    opt = keras.optimizers.Adam(learning_rate=0.001)

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    '''
    cnn_history = model.fit(x=[x_mfcc_traincnn, x_lmfe_traincnn], y=y_mfcc_train,
                            batch_size=8, epochs=50, validation_split=0.2)
    '''

    start_time = time.time()
    # usage = resource.getrusage(resource.resource.RUSAGE_SELF)
    cnn_history = model.fit(x=[x_mfcc_traincnn, x_lmfe_traincnn], y=y_mfcc_train,
                            batch_size=8, epochs=5
                            , validation_data=([x_mfcc_testcnn, x_lmfe_testcnn], y_mfcc_test))
    elapsed_time = time.time() - start_time
    print("elapsed time for training phase: ", elapsed_time)

    UtilityClass.plot_trained_model(cnn_history)

    UtilityClass.model_summary(model, x_mfcc_testcnn, y_mfcc_test)

    return model


if __name__ == '__main__':
    crr_path = Path(os.getcwd())
    parent_path = crr_path.parent
    save_dir = str(parent_path) + '\\data\\features\\tone'

    X_mfcc_file = save_dir + '\\Xmfcc.joblib'
    y_mfcc_file = save_dir + '\\ymfcc.joblib'

    X_lmfe_file = save_dir + '\\Xlmfe.joblib'
    y_lmfe_file = save_dir + '\\ylmfe.joblib'

    feature_extractor = FeatureExtractor()
    # padding audio such that all have the same length
    feature_extractor.preprocessing()
    print("preprocessing done")

    if not Path(X_mfcc_file).exists() or not Path(y_mfcc_file).exists():
        print("Extracting mfcc feature .... ")
        feature_extractor.tone_mfcc_features_creator()
    if not Path(X_lmfe_file).exists() or not Path(y_lmfe_file).exists():
        print("Extracting lmfe feature .... ")
        feature_extractor.tone_lmfe_features_creator()

    Xmfcc = joblib.load(save_dir + '\\Xmfcc.joblib')
    Xlmfe = joblib.load(save_dir + '\\Xlmfe.joblib')
    ymfcc = joblib.load(save_dir + '\\ymfcc.joblib')
    ylmfe = joblib.load(save_dir + '\\ylmfe.joblib')

    trained_model = train_neural_network(X=[Xmfcc, Xlmfe], y=[ymfcc, ylmfe])

    UtilityClass.save_model(trained_model)
