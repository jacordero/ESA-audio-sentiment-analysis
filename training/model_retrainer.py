import os
import yaml
from pathlib import Path

from sequential_model_generator import SequentialModelGeneratorFactory
from data_loader import SequentialDataLoader
import keras
import numpy as np
import tensorflow as tf


def create_model_dir_path(models_dir, model_name):

    root_path = Path(os.getcwd())
    tmp_model_dir_path = os.path.join(root_path, models_dir, model_name)
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


def save_model(trained_model, trained_model_dir_path):
    """[summary]

    Args:
        trained_model : model to be saved as .h5 format
        trained_model_dir_path : directory where the trained model will be saved
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


def load_model(models_dir, model_dir, model_name):

    model_dir_path = create_model_dir_path(models_dir, model_dir)
    model_file_path = os.path.join(model_dir_path, model_name)
    print(model_file_path)
    model = tf.keras.models.load_model(model_file_path)
    return model


def retrain_sequential_model(parameters):
    # load data
    root_path = Path(os.getcwd())
    train_data_dir_path = os.path.normpath(os.path.join(
        root_path, parameters["train_data_directory"]))
    val_data_dir_path = os.path.normpath(os.path.join(
        root_path, parameters["validation_data_directory"]))
    test_data_dir_path = os.path.normpath(os.path.join(
        root_path, parameters["test_data_directory"]))

    data_loader = SequentialDataLoader()
    mfcc_train, labels_train = data_loader.load_train_data(
        train_data_dir_path)
    mfcc_val, labels_val = data_loader.load_validation_data(
        val_data_dir_path)
    mfcc_test, labels_test = data_loader.load_validation_data(
        test_data_dir_path)

    # load model to retrain
    model = load_model(
        parameters["pretrained_models_dir"], parameters["pretrained_model_name"], "saved_models/Emotion_Voice_Detection_Model.h5")
    print(model.summary())

    # compile model
    opt = keras.optimizers.Adam(learning_rate=parameters["adam_lr"])
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer=opt, metrics=["accuracy"])

    # train model
    history = model.fit(x=[mfcc_train], y=labels_train, batch_size=parameters["batch_size"], epochs=parameters["epochs"],
                        validation_data=(mfcc_val, labels_val))

    # save model
    retrained_model_dir_path = create_model_dir_path(
        parameters['retrained_models_dir'], parameters['retrained_model_name'])
    save_model(model, retrained_model_dir_path)
    #model.save(retrained_model_path)

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
