import os
import time
import librosa
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense
from keras.layers import Conv1D, MaxPooling1D
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Activation
from keras.models import Sequential
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from pathlib import Path
import keras
from keras.layers.normalization import BatchNormalization
import json
import joblib
from us41_utility import UtilityClass
from us41_feature_extraction import feature_extractor


def create_model_new():
    model = Sequential()
    model.add(Conv1D(128, 5, padding='same',
                     input_shape=(40, 1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    '''
    model.add(Conv1D(256, 5, padding='same',
                     input_shape=(100, 1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    # model.add(MaxPooling1D(pool_size=(8)))
    '''
    model.add(Flatten())
    model.add(Dense(8))
    model.add(Dense(8))
    model.add(Activation('softmax'))
    return model


def create_model():
    model = Sequential()
    model.add(Conv1D(32, 5, padding='same',
                     input_shape=(40, 1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Conv1D(128, 5, padding='same',
                     input_shape=(40, 1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(MaxPooling1D(pool_size=(10)))

    model.add(Conv1D(256, 5, padding='same',
                     input_shape=(40, 1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    #model.add(MaxPooling1D(pool_size=(8)))

    model.add(Flatten())
    model.add(Dense(8))
    model.add(Dense(8))
    model.add(Dense(8))
    model.add(Activation('softmax'))
    return model

def train_neural_network(X, y):
    """
    This function trains the neural network.
    """

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    x_traincnn = np.expand_dims(X_train, axis=2)
    x_testcnn = np.expand_dims(X_test, axis=2)

    print(x_traincnn.shape, x_testcnn.shape)

    model = create_model()
    print(model.summary)
    opt = keras.optimizers.Adam(learning_rate=0.001)

    model.compile(loss='sparse_categorical_crossentropy',
                  # categorical_crossentropy sparse_categorical_crossentropy
                  optimizer=opt,
                  metrics=['accuracy'])

    cnn_history = model.fit(x_traincnn, y_train,
                            batch_size=8, epochs=200,
                            validation_data=(x_testcnn, y_test))

    #UtilityClass.plot_trained_model(cnn_history)

    UtilityClass.model_summary(model, x_testcnn, y_test )

    return model


if __name__ == '__main__':
    crr_path = Path(os.getcwd())
    parent_path = crr_path.parent
    save_dir = str(parent_path) + '\\data\\features\\tone'

    X_file = save_dir + '\\X.joblib'
    y_file = save_dir + '\\y.joblib'

    if not Path(X_file).exists() or not Path(y_file).exists():
        feature_extractor = feature_extractor()
        X, y = feature_extractor.tone_features_creator()
        print("Extracting features .... ")
    else:
        X = joblib.load(save_dir + '\\X.joblib')
        y = joblib.load(save_dir + '\\y.joblib')

    trained_model = train_neural_network(X=X, y=y)
    UtilityClass.save_model(trained_model)
