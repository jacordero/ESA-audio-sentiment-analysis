import os
import time
import numpy as np
from keras.layers import Conv1D, MaxPooling1D, Conv2D, MaxPooling2D, concatenate, Flatten, Dropout
from keras.layers import SpatialDropout2D, Activation, Input, Concatenate, Dense
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from pathlib import Path
import keras
from keras.layers.normalization import BatchNormalization
import joblib
import tensorflow as tf
from sklearn.metrics import accuracy_score
from us52_utility import UtilityClass
from us52_feature_extraction import FeatureExtractor


# Package metadata
__authors__ = "Raha Sadeghi, Parima Mirshafiei"
__email__ = "r.sadeghi@tue.nl; P.mirshafiei@tue.nl"
__copyright__ = "TU/e ST2019"
__version__ = "1.0"
__status__ = "Prototype"


def create_siamese_model(input_shape_x, input_shape_y):
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


def create_sequential_model():
    """
        This function generates a sequential model
    """
    model = Sequential()
    model.add(Conv1D(32, 5, padding='same',
                     input_shape=(50, 1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
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


def create_final_model(branch1, branch2, input1, input2):
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


def train_neural_network(X, y, testing_with_different_dataset):
    """
    This function trains the neural network.
    :param X: the input data (it is a list)
    :param y: the input label (it is a list)
    :param testing_with_different_dataset: to show whether the model should be validated with a new test dataset or
    split the current dataset for validation.
    """

    x_mfcc_train, x_mfcc_test, y_mfcc_train, y_mfcc_test = \
        train_test_split(X[0], y[0], test_size=0.15, random_state=42)
    x_mfcc_train, x_mfcc_validation, y_mfcc_train, y_mfcc_validation = \
        train_test_split(x_mfcc_train, y_mfcc_train, test_size=0.15, random_state=42)

    x_lmfe_train, x_lmfe_test, y_lmfe_train, y_lmfe_test = \
        train_test_split(X[1], y[1], test_size=0.15, random_state=42)
    x_lmfe_train, x_lmfe_validation, y_lmfe_train, y_lmfe_validation = \
        train_test_split(x_lmfe_train, y_lmfe_train, test_size=0.15, random_state=42)

    # expanding dimensions, required for using conv2D
    x_mfcc_traincnn = np.expand_dims(x_mfcc_train, axis=3)
    x_mfcc_testcnn = np.expand_dims(x_mfcc_test, axis=3)
    x_mfcc_validationcnn = np.expand_dims(x_mfcc_validation, axis=3)

    # expanding dimensions, required for using conv2D
    x_lmfe_traincnn = np.expand_dims(x_lmfe_train, axis=3)
    x_lmfe_testcnn = np.expand_dims(x_lmfe_test, axis=3)
    x_lmfe_validationcnn = np.expand_dims(x_lmfe_validation, axis=3)

    mfcc_input, mfcc_output = create_siamese_model(x_mfcc_traincnn.shape[1], x_mfcc_traincnn.shape[2])

    lmfe_input, lmfe_output = create_siamese_model(x_lmfe_traincnn.shape[1], x_lmfe_traincnn.shape[2])

    model = create_final_model(mfcc_output, lmfe_output, mfcc_input, lmfe_input)
    print(model.summary)

    opt = keras.optimizers.Adam(learning_rate=0.001)

    checkpoint_filepath = get_checkpoint_path()
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=False,
        monitor='val_accuracy',
        mode='auto',
        save_freq="epoch",
        save_best_only=False)

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    start_time = time.time()
    # usage = resource.getrusage(resource.resource.RUSAGE_SELF)
    cnn_history = model.fit(x=[x_mfcc_traincnn, x_lmfe_traincnn], y=y_mfcc_train, batch_size=8, epochs=2,
                            callbacks=[model_checkpoint_callback],
                            validation_data=([x_mfcc_validationcnn, x_lmfe_validationcnn], y_mfcc_validation))
    elapsed_time = time.time() - start_time
    print("elapsed time for training phase: ", elapsed_time)
    UtilityClass.plot_trained_model(cnn_history)

    # testing
    if not testing_with_different_dataset:
        preds = model.predict(x=[x_mfcc_testcnn, x_lmfe_testcnn],
                              batch_size=8,
                              verbose=1)
        preds1 = preds.argmax(axis=1)
        final_result = accuracy_score(y_mfcc_test, preds1)
    else:
        preds = model.predict(x=[X[2], X[3]], batch_size=8, verbose=1)
        preds1 = preds.argmax(axis=1)
        final_result = accuracy_score(y[2], preds1)
    print(final_result)
    # UtilityClass.model_summary(model, x_mfcc_testcnn, y_mfcc_test)
    return model


def get_checkpoint_path():
    """
        This function returns the checkpoint location. It uses MFCC as input.
    """
    crr_path = Path(os.getcwd())
    checkpoint_dir = os.path.join(crr_path, "models", "checkpoint")
    return checkpoint_dir


def get_saved_feature_path():
    """
        This function returns the saved feature location (.joblib files)
    """
    crr_path = Path(os.getcwd())
    save_feature_dir = os.path.join(crr_path, "data", "features", "tone")
    return save_feature_dir


def get_data_path():
    """
        This function returns the dataset location
    """
    crr_path = Path(os.getcwd())
    path = os.path.join(crr_path, "data", "tone_cnn_8_emotions_dataset")
    return path


def get_test_data_path():
    """
        This function returns the test dataset location
    """
    crr_path = Path(os.getcwd())
    path = os.path.join(crr_path, "data", "tone_test")
    return path


if __name__ == '__main__':

    feature_save_dir = get_saved_feature_path()

    X_mfcc_file = os.path.join(feature_save_dir, "Xmfcc.joblib")
    y_mfcc_file = os.path.join(feature_save_dir, "ymfcc.joblib")
    X_lmfe_file = os.path.join(feature_save_dir, "Xlmfe.joblib")
    y_lmfe_file = os.path.join(feature_save_dir, "ylmfe.joblib")

    validation_testing_enabled = True
    feature_extractor = FeatureExtractor()
    # padding audio such that all have the same length
    feature_extractor.preprocessing()
    print("Preprocessing is done")

    if not Path(X_mfcc_file).exists() or not Path(y_mfcc_file).exists():
        print("Extracting MFCC feature .... ")
        feature_extractor.tone_mfcc_features_creator()
    if not Path(X_lmfe_file).exists() or not Path(y_lmfe_file).exists():
        print("Extracting LMFE feature .... ")
        feature_extractor.tone_lmfe_features_creator()

    #   checking whether we want to test our model against a separate dataset
    if validation_testing_enabled:

        X_mfcc_test_file = os.path.join(feature_save_dir, "Xmfcc_testing.joblib")
        y_mfcc_test_file = os.path.join(feature_save_dir, "ymfcc_testing.joblib")
        X_lmfe_test_file = os.path.join(feature_save_dir, "Xlmfe_testing.joblib")
        y_lmfe_test_file = os.path.join(feature_save_dir, "ylmfe_testing.joblib")

        if not Path(X_lmfe_test_file).exists() or not Path(y_lmfe_test_file).exists():
            print("Extracting test LMFE feature .... ")
            feature_extractor.tone_lmfe_features_creator(goal="testing")
        if not Path(X_mfcc_test_file).exists() or not Path(y_mfcc_test_file).exists():
            print("Extracting test MFCC feature .... ")
            feature_extractor.tone_mfcc_features_creator(goal="testing")

    Xmfcc = joblib.load(os.path.join(feature_save_dir, "Xmfcc.joblib"))
    Xlmfe = joblib.load(os.path.join(feature_save_dir, "Xlmfe.joblib"))
    ymfcc = joblib.load(os.path.join(feature_save_dir, "ymfcc.joblib"))
    ylmfe = joblib.load(os.path.join(feature_save_dir, "ylmfe.joblib"))

    if validation_testing_enabled:
        Xmfcc_test = joblib.load(os.path.join(feature_save_dir, "Xmfcc_testing.joblib"))
        Xlmfe_test = joblib.load(os.path.join(feature_save_dir, "Xlmfe_testing.joblib"))
        ymfcc_test = joblib.load(os.path.join(feature_save_dir, "ymfcc_testing.joblib"))
        ylmfe_test = joblib.load(os.path.join(feature_save_dir, "ylmfe_testing.joblib"))

        trained_model = train_neural_network(X=[Xmfcc, Xlmfe, Xmfcc_test, Xlmfe_test],
                                             y=[ymfcc, ylmfe, ymfcc_test, ylmfe_test],
                                             testing_with_different_dataset=validation_testing_enabled)
    else:
        trained_model = train_neural_network(X=[Xmfcc, Xlmfe], y=[ymfcc, ylmfe],
                                             testing_with_different_dataset=validation_testing_enabled)
    UtilityClass.save_model(trained_model)
