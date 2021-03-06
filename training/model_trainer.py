"""
Copyright (c) 2020 TU/e - PDEng Software Technology C2019. All rights reserved. 
@ Authors: Raha Sadeghi r.sadeghi@tue.nl; Parima Mirshafiei p.mirshafiei@tue.nl; Jorge Cordero j.a.cordero.cruz@tue.nl;
Last modified: 01-12-2020
"""
import os
import yaml
import time
from model_generator_factory import ModelGeneratorFactory
from data_loader import SequentialDataLoader, SiameseDataLoader
import keras
import numpy as np
from pathlib import Path

def create_trained_model_path(trained_models_dir, trained_model_name):
    """This function creates the path to the model based on the training parameters yaml file.

    Args:
        trained_models_dir: the directory contatining the model
        trained_model_name: the model name (.h5)

    Returns:
        the model path
    """
    root_path = Path(os.getcwd())
    tmp_trained_model_dir_path = os.path.join(root_path, trained_models_dir, trained_model_name)
    return os.path.normpath(tmp_trained_model_dir_path)

def save_training_info(model_type, parameters, training_history):
    """This function saves the training information.

    Args:
        model_type: the type of the model, either sequential or siamese
        parameters: a dictionary of parameters defined in the training parameters yaml file
        training_history: the history of the training as numpy array
    """
    trained_model_dir_path = create_trained_model_path(parameters['trained_models_dir'], parameters['trained_model_name'])

    yaml_file_path = os.path.join(trained_model_dir_path, "training_parameters.yml")
    parameters['input_shape'] = list(parameters['input_shape'])
    yaml_content = {
        "model_type": model_type,
        "parameters": parameters
    }
    with open(yaml_file_path, 'w') as f:
        yaml.dump(yaml_content, f)

    history_file_path = os.path.join(trained_model_dir_path, "history.npy")
    np.save(history_file_path, training_history.history)

def save_model(trained_model, trained_model_dir_path):
    """This function saves the model as .h5 format in the saved_models directory

    Args:
        trained_model: model to be saved as .h5 format
        trained_model_dir_path: directory where the trained model will be saved
    """
    model_name = 'Emotion_Voice_Detection_Model.h5'
    save_dir = os.path.join(trained_model_dir_path, 'saved_models')
    # Save model and weights
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model_path = os.path.join(save_dir, model_name)
    trained_model.save(model_path)
    print('Saved trained model at %s ' % model_path)

    model_json = trained_model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)


def train_sequential_model(parameters):
    """This function loads the joblib (features) for training the sequential model, and based on the parameters specified in the config file
    trains the model and finally saves the model.

    Args:
        parameters: a list of required parameters read from the training_parameters.yml file

    Raises:
        ValueError: the number of emotions does not match the number of unique training labels
    """
    # load data
    root_path = Path(os.getcwd())
    feature_train_data_directory = os.path.normpath(os.path.join(root_path, parameters["feature_train_data_directory"]))

    feature_validation_data_directory = os.path.normpath(os.path.join(root_path, parameters["feature_validation_data_directory"]))

    feature_test_data_directory = os.path.normpath(os.path.join(root_path, parameters["feature_test_data_directory"]))

    data_loader = SequentialDataLoader()
    mfcc_train, labels_train = data_loader.load_train_data(
        feature_train_data_directory)
    mfcc_val, labels_val = data_loader.load_validation_data(
        feature_validation_data_directory)
    mfcc_test, labels_test = data_loader.load_test_data(
        feature_test_data_directory)

    if parameters["n_emotions"] != len(np.unique(labels_train)):
        message = "The value n_emotions and the number of labels in the training dataset are different.\n"
        message += "n_emotions: {}, number of labels: {}".format(parameters["n_emotions"], len(np.unique(labels_train)))
        raise ValueError(message)

    # create model
    parameters["input_shape"] = [mfcc_train.shape[1], mfcc_train.shape[2]]

    model_generator_factory = ModelGeneratorFactory()
    model_generator = model_generator_factory.get_model_generator(
        parameters["model_generator"])
    model = model_generator.generate_model(
        parameters["n_conv_filters"], parameters["filters_shape"], parameters["input_shape"], parameters["n_emotions"])
    print(model.summary())

    # compile model
    opt = keras.optimizers.Adam(learning_rate=parameters["adam_lr"])
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer=opt, metrics=["accuracy"])

    # train model
    start_time = time.time()
    history = model.fit(x=[mfcc_train], y=labels_train, batch_size=parameters["batch_size"], epochs=parameters["epochs"],
                         validation_data=(mfcc_val, labels_val))
    elapsed_time = time.time() - start_time
    print("elapsed time for training phase: ", elapsed_time)

    # save model
    trained_model_dir_path = create_trained_model_path(parameters['trained_models_dir'], parameters['trained_model_name'])
    save_model(model, trained_model_dir_path)

    save_training_info(model_type, parameters, history)

def train_siamese_model(parameters):
    """This function loads the joblib (features) for training the siamese model, and based on the parameters specified in the config file
    trains the model and finally saves the model.

    Args:
        parameters: a list of required parameters read from the training_parameters.yml file
    """
    # load data
    root_path = Path(os.getcwd())
    feature_train_data_directory = os.path.normpath(os.path.join(root_path, parameters["feature_train_data_directory"]))
    feature_test_data_directory = os.path.normpath(os.path.join(root_path, parameters["feature_test_data_directory"]))
    feature_validation_data_directory = os.path.normpath(os.path.join(root_path, parameters["feature_validation_data_directory"]))

    data_loader = SiameseDataLoader()
    mfcc_train, lmfe_train, labels_train = data_loader.load_train_data(
        feature_train_data_directory)
    mfcc_val, lmfe_val, labels_val = data_loader.load_validation_data(
        feature_validation_data_directory)
    mfcc_test, lmfe_test, labels_test = data_loader.load_test_data(
        feature_test_data_directory)

    # create model
    print("Shape: {}".format(mfcc_train.shape))
    parameters["input_shape"] = [mfcc_train.shape[1], mfcc_train.shape[2]]

    siamese_factory = ModelGeneratorFactory()
    siamese_model = siamese_factory.get_model_generator(parameters["model_generator"])
    inputes_shape = [mfcc_train.shape[1], mfcc_train.shape[2],lmfe_train.shape[1], lmfe_train.shape[2] ]
    model = siamese_model.generate_model(parameters["n_conv_filters"], parameters["filters_shape"], inputes_shape, parameters["n_emotions"])
    print(model.summary())

    # compile model
    opt = keras.optimizers.Adam(learning_rate=parameters["adam_lr"])
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer=opt, metrics=["accuracy"])


    # train model
    start_time = time.time()

    history = model.fit(x=[mfcc_train, lmfe_train], y=labels_train,
                         batch_size=parameters["batch_size"], 
                         epochs=parameters["epochs"],
                         validation_data=([mfcc_val, lmfe_val], labels_val))
    elapsed_time = time.time() - start_time
    print("elapsed time for training phase: ", elapsed_time)

    # save model
    trained_model_dir_path = create_trained_model_path(parameters['trained_models_dir'], parameters['trained_model_name'])
    save_model(model, trained_model_dir_path)
    save_training_info(model_type, parameters, history)



def train(model_type, parameters):
    """This function call the required training function based on the model_type specified in the training_parameters.yml file

    Args:
        model_type: the type of the model to be trained: either "Sequential" or "Siamese"
        parameters: parameters required to train the models

    Raises:
        ValueError: model type is invalid
    """
    if model_type == "Sequential":
        train_sequential_model(parameters)
    elif model_type == "Siamese":
        train_siamese_model(parameters)
    else:
        raise ValueError("Invalid model type: {}".format(model_type))

def load_training_parameters(training_parameters_filename):
    """This function read the training_parameters.yml file to train the model based on the specified configuration parameters.

    Args:
        training_parameters_filename: the path/filename of the configuration file

    Returns:
        loaded training parameters
    """
    # load training parameters from yaml file
    root_path = Path(os.getcwd())
    with open(os.path.join(root_path, "training", training_parameters_filename)) as f:
        training_parameters = yaml.load(f, Loader=yaml.FullLoader)

    # preprocess training parameters
    model_type = training_parameters['model_type']
    parameters = training_parameters['parameters']
    input_shape = parameters['input_shape']
    parameters['input_shape'] = (input_shape[0], input_shape[1])

    return model_type, parameters

if __name__ == "__main__":
    """
    Loads training parameters from a configuration file and trains a tone sentiment detection model.
    """
    training_parameters_filename = "training_parameters.yml"
    model_type, parameters = load_training_parameters(training_parameters_filename)
    train(model_type, parameters)
