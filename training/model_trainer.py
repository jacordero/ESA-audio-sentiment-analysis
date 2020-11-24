import os
import yaml

from sequential_model_generator import SequentialModelGeneratorFactory
from data_loader import SequentialToneModelDataLoader
import keras
import numpy as np


def create_trained_model_path(trained_models_dir, trained_model_name):

    script_dir = os.path.dirname(os.path.abspath(__file__))
    tmp_trained_model_dir_path = os.path.join(script_dir, trained_models_dir, trained_model_name)
    return os.path.normpath(tmp_trained_model_dir_path)

def save_training_info(model_type, parameters, training_history):

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

def train_sequential_model(parameters):
    # load data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir_path = os.path.normpath(os.path.join(script_dir, parameters["data_directory"]))

    data_loader = SequentialToneModelDataLoader()
    mfcc_train, labels_train = data_loader.load_train_data(
        data_dir_path)
    mfcc_val, labels_val = data_loader.load_validation_data(
        data_dir_path)
    mfcc_test, labels_test = data_loader.load_test_data(
        data_dir_path)

    # create model
    #print(parameters["n_conv_filters"])
    #print(parameters["filters_shape"])
    print("Shape: {}".format(mfcc_train.shape))
    parameters["input_shape"] = [mfcc_train.shape[1], mfcc_train.shape[2]]

    model_generator_factory = SequentialModelGeneratorFactory()
    model_generator = model_generator_factory.get_model_generator(
        parameters["model_generator"])
    model = model_generator.generate_model(
        parameters["n_conv_filters"], parameters["filters_shape"], parameters["input_shape"])
    print(model.summary())

    # compile model
    opt = keras.optimizers.Adam(learning_rate=parameters["adam_lr"])
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer=opt, metrics=["accuracy"])

    # train model
    history = model.fit(x=[mfcc_train], y=labels_train, batch_size=parameters["batch_size"], epochs=parameters["epochs"],
                         validation_data=(mfcc_val, labels_val))

    # save model
    trained_model_path = create_trained_model_path(parameters['trained_models_dir'], parameters['trained_model_name'])
    model.save(trained_model_path)

    save_training_info(model_type, parameters, history)

def train_siamese_model(parameters):
    pass


def train(model_type, parameters):
    if model_type == "sequential":
        train_sequential_model(parameters)
    elif model_type == "siamese":
        train_siamese_model(parameters)
    else:
        raise ValueError("Invalid model type: {}".format(model_type))

def load_training_parameters(training_parameters_filename):
    
    # load training parameters from yaml file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(script_dir, training_parameters_filename)) as f:
        training_parameters = yaml.load(f, Loader=yaml.FullLoader)

    # preprocess training parameters
    model_type = training_parameters['model_type']
    parameters = training_parameters['parameters']
    input_shape = parameters['input_shape']
    parameters['input_shape'] = (input_shape[0], input_shape[1])

    return model_type, parameters

if __name__ == "__main__":

    training_parameters_filename = "training_parameters.yml"
    model_type, parameters = load_training_parameters(training_parameters_filename)
    train(model_type, parameters)
