import os
import yaml
import time
from pathlib import Path
from model_generator_factory import ModelGeneratorFactory
from data_loader import SequentialDataLoader, SiameseDataLoader
import keras
import numpy as np
import tensorflow as tf

__authors__ = "Raha Sadeghi, Parima Mirshafiei, Jorge Cordero ",
__email__ = "r.sadeghi@tue.nl; P.mirshafiei@tue.nl;j.a.cordero.cruz@tue.nl;"
__copyright__ = "TU/e ST2019"

def create_model_dir_path(models_dir, model_name):
    """
    This function creates the path to the model based on the training parameters yaml file.
    :param trained_models_dir: the directory contatining the model
    :param trained_model_name: the model name (.h5)
    :Return the model path
    """
    root_path = Path(os.getcwd())
    tmp_model_dir_path = os.path.join(root_path, models_dir, model_name)
    return os.path.normpath(tmp_model_dir_path)


def save_retraining_info(model_type, parameters, retraining_history):
    """
    This function saves the retraining information.
    :param  model_type: the type of the model, either sequential or siamese
    :param  parameters: a dictionary of parameters defined in the retraining parameters yaml file
    :param  retraining_history:    the hisory of the retraining as numpy array
    """
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
    np.save(history_file_path, retraining_history.history)


def save_model(retrained_model, retrained_model_dir_path):
    """
    This function saves the model as .h5 format in the saved_models directory
    params:
        retrained_model : model to be saved as .h5 format
        retrained_model_dir_path : directory where the trained model will be saved
    """ 
    model_name = 'Emotion_Voice_Detection_Model.h5'
    save_dir = os.path.join(retrained_model_dir_path, 'saved_models')
    # Save model and weights
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model_path = os.path.join(save_dir, model_name)
    retrained_model.save(model_path)
    print('Saved trained model at %s ' % model_path)

    model_json = retrained_model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)


def load_model(model_dir, model_name):
    """
    This function loads the pretrained model and returns the model
    :param model_dir: the path to the model
    :param model_name: the name of the model (.h5 model)
    """
    model_path = create_model_dir_path(model_dir, model_name)
    model = tf.keras.models.load_model(model_path)
    return model


def retrain_sequential_model(parameters):
    """
    This function loads the joblib (features) for retraining the sequential model,
    additionally, it loads the previous model, and based on the parameters specified in the config file
    retrains the model and finally saves the model.
    Params:
        parameters: a list of required parameters read from the retraining_parameters.yml file
    """
    # load data
    root_path = Path(os.getcwd())
    train_data_dir_path = os.path.normpath(os.path.join(
        root_path, parameters["feature_train_data_directory"]))
    val_data_dir_path = os.path.normpath(os.path.join(
        root_path, parameters["feature_validation_data_directory"]))
    test_data_dir_path = os.path.normpath(os.path.join(
        root_path, parameters["feature_test_data_directory"]))

    data_loader = SequentialDataLoader()
    mfcc_train, labels_train = data_loader.load_train_data(
        train_data_dir_path)
    mfcc_val, labels_val = data_loader.load_validation_data(
        val_data_dir_path)
    mfcc_test, labels_test = data_loader.load_test_data(
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
    retrained_model_dir_path = create_model_dir_path(
        parameters['retrained_models_dir'], parameters['retrained_model_name'])
    save_model(model, retrained_model_dir_path)
    save_retraining_info(model_type, parameters, history)


def retrain_siamese_model(parameters):
    """
    This function loads the joblib (features) for retraining the siamese model,
    additionally, it loads the previous model, and based on the parameters specified in the config file
    retrains the model and finally saves the model.
    Params:
        parameters: a list of required parameters read from the retraining_parameters.yml file
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

    
    # load model to retrain
    model = load_model(
        parameters["pretrained_models_dir"], parameters["pretrained_model_name"])
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
    retrained_model_dir_path = create_model_dir_path(parameters['retrained_models_dir'], parameters['retrained_model_name'])
    save_model(model, retrained_model_dir_path)
    save_retraining_info(model_type, parameters, history)


def retrain(model_type, parameters):
    """
    This function call the required retraining function based on the model_type specified in the retraining_parameters.yml file
    Params:
        model_type: the type of the model to be retrained: either "Sequential" or "Siamese"
    """
    if model_type == "Sequential":
        retrain_sequential_model(parameters)
    elif model_type == "Siamese":
        retrain_siamese_model(parameters)
    else:
        raise ValueError("Invalid model type: {}".format(model_type))


def load_retraining_parameters(retraining_parameters_filename):
    """
    This function read the training_parameters.yml file to train the model based on the specified configuration parameters.
    Param: retraining_parameters_filename: the path/filename of the configuration file
    """
    # load retraining parameters from yaml file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(script_dir, retraining_parameters_filename)) as f:
        training_parameters = yaml.load(f, Loader=yaml.FullLoader)

    # preprocess retraining parameters
    model_type = training_parameters['model_type']
    parameters = training_parameters['parameters']
    return model_type, parameters


if __name__ == "__main__":

    training_parameters_filename = "retraining_parameters.yml"
    model_type, parameters = load_retraining_parameters(training_parameters_filename)
    retrain(model_type, parameters)
