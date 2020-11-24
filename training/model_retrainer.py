import os
import yaml

from sequential_model_generator import SequentialModelGeneratorFactory
from data_loader import SequentialToneModelDataLoader
import keras
import numpy as np
import tensorflow as tf


def create_model_dir_path(models_dir, model_name):

    script_dir = os.path.dirname(os.path.abspath(__file__))
    tmp_model_dir_path = os.path.join(script_dir, models_dir, model_name)
    return os.path.normpath(tmp_model_dir_path)


def save_retraining_info(model_type, parameters, training_history):

    trained_model_dir_path = create_model_dir_path(
        parameters['retrained_models_dir'], parameters['retrained_model_name'])

    yaml_file_path = os.path.join(
        trained_model_dir_path, "retraining_parameters.yml")
    yaml_content = {
        "model_type": model_type,
        "parameters": parameters
    }
    with open(yaml_file_path, 'w') as f:
        yaml.dump(yaml_content, f)

    history_file_path = os.path.join(trained_model_dir_path, "history.npy")
    np.save(history_file_path, training_history.history)


def load_model(models_dir, model_name):

    model_dir_path = create_model_dir_path(models_dir, model_name)
    print(model_dir_path)
    model = tf.keras.models.load_model(model_dir_path)
    return model


def retrain_sequential_model(parameters):
    # load data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_data_dir_path = os.path.normpath(os.path.join(
        script_dir, parameters["train_data_directory"]))
    val_data_dir_path = os.path.normpath(os.path.join(
        script_dir, parameters["validation_data_directory"]))
    test_data_dir_path = os.path.normpath(os.path.join(
        script_dir, parameters["test_data_directory"]))

    data_loader = SequentialToneModelDataLoader()
    mfcc_train, labels_train = data_loader.load_train_data(
        train_data_dir_path)
    mfcc_val, labels_val = data_loader.load_validation_data(
        val_data_dir_path)
    mfcc_test, labels_test = data_loader.load_validation_data(
        test_data_dir_path)

    # load model to retrain
    model = load_model(
        parameters["pretrained_models_dir"], parameters["pretrained_model_name"])
    print(model.summary())

    # compile model
    opt = keras.optimizers.Adam(learning_rate=parameters["adam_lr"])
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer=opt, metrics=["accuracy"])

    # train model
    history = model.fit(x=[mfcc_train], y=labels_train, batch_size=parameters["batch_size"], epochs=parameters["epochs"],
                        validation_data=(mfcc_val, labels_val))

    # save model
    retrained_model_path = create_model_dir_path(
        parameters['retrained_models_dir'], parameters['retrained_model_name'])
    model.save(retrained_model_path)

    save_retraining_info(model_type, parameters, history)


def retrain_siamese_model(parameters):
    pass


def retrain(model_type, parameters):
    if model_type == "sequential":
        retrain_sequential_model(parameters)
    elif model_type == "siamese":
        retrain_siamese_model(parameters)
    else:
        raise ValueError("Invalid model type: {}".format(model_type))


def load_retraining_parameters(training_parameters_filename):

    # load training parameters from yaml file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(script_dir, training_parameters_filename)) as f:
        training_parameters = yaml.load(f, Loader=yaml.FullLoader)

    # preprocess training parameters
    model_type = training_parameters['model_type']
    parameters = training_parameters['parameters']

    return model_type, parameters


if __name__ == "__main__":

    training_parameters_filename = "retraining_parameters.yml"
    model_type, parameters = load_retraining_parameters(
        training_parameters_filename)
    retrain(model_type, parameters)
