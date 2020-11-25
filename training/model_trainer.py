import os
import yaml
import time
from model_generator_factory import ModelGeneratorFactory
from data_loader import SequentialToneModelDataLoader, SiameseToneModelDataLoader
import keras
import numpy as np
from pathlib import Path


__authors__ = "Raha Sadeghi, Parima Mirshafiei, Jorge Cordero ",
__email__ = "r.sadeghi@tue.nl; P.mirshafiei@tue.nl;j.a.cordero.cruz@tue.nl;"
__copyright__ = "TU/e ST2019"

def create_trained_model_path(trained_models_dir, trained_model_name):
    """
    The 
    """
    root_path = Path(os.getcwd())
    tmp_trained_model_dir_path = os.path.join(root_path, trained_models_dir, trained_model_name)
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

def save_model(trained_model, trained_model_dir_path):
    """
    This function saves the model as .h5 format in the saved_models directory
    params:
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


def train_sequential_model(parameters):
    """
    This function loads the joblib (features) for training the sequential model, and based on the parameters specified in the config file
    trains the model and finally saves the model.
    Params:
        parameters: a list of required parameters read from the training_parameters.yml file
    """
    # load data
    root_path = Path(os.getcwd())
    feature_train_data_directory = os.path.normpath(os.path.join(root_path, parameters["feature_train_data_directory"]))

    feature_validation_data_directory = os.path.normpath(os.path.join(root_path, parameters["feature_validation_data_directory"]))

    feature_test_data_directory = os.path.normpath(os.path.join(root_path, parameters["feature_test_data_directory"]))

    data_loader = SequentialToneModelDataLoader()
    mfcc_train, labels_train = data_loader.load_train_data(
        feature_train_data_directory)
    mfcc_val, labels_val = data_loader.load_validation_data(
        feature_validation_data_directory)
    mfcc_test, labels_test = data_loader.load_test_data(
        feature_test_data_directory)

    # create model
    print("Shape: {}".format(mfcc_train.shape))
    parameters["input_shape"] = [mfcc_train.shape[1], mfcc_train.shape[2]]

    model_generator_factory = ModelGeneratorFactory()
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
    """
    This function loads the joblib (features) for training the siamese model, and based on the parameters specified in the config file
    trains the model and finally saves the model.
    Params:
        parameters: a list of required parameters read from the training_parameters.yml file
    """
    # load data
    root_path = Path(os.getcwd())
    feature_train_data_directory = os.path.normpath(os.path.join(root_path, parameters["feature_train_data_directory"]))
    feature_test_data_directory = os.path.normpath(os.path.join(root_path, parameters["feature_test_data_directory"]))
    feature_validation_data_directory = os.path.normpath(os.path.join(root_path, parameters["feature_validation_data_directory"]))

    data_loader = SiameseToneModelDataLoader()
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
    mfcc_input, mfcc_output = siamese_model.create_siamese_branch_architecture(parameters["n_conv_filters"], parameters["filters_shape"], mfcc_train.shape[1], mfcc_train.shape[2]  )
    lmfe_input, lmfe_output = siamese_model.create_siamese_branch_architecture(parameters["n_conv_filters"], parameters["filters_shape"], lmfe_train.shape[1], lmfe_train.shape[2] )
    model = siamese_model.generate_model(mfcc_output, lmfe_output, mfcc_input, lmfe_input)
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
    """
    This function call the required training function based on the model_type specified in the training_parameters.yml file
    Params:
        model_type: the type of the model to be trained: either "Sequential" or "Siamese"
    """
    if model_type == "Sequential":
        train_sequential_model(parameters)
    elif model_type == "Siamese":
        train_siamese_model(parameters)
    else:
        raise ValueError("Invalid model type: {}".format(model_type))

def load_training_parameters(training_parameters_filename):
    """
    This function read the training_parameters.yml file to train the model based on the specified configuration parameters.
    Param: 
        training_parameters_filename: the path/filename of the configuration file
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
    main function to define the configuration file and train the model
    """
    training_parameters_filename = "training_parameters.yml"
    model_type, parameters = load_training_parameters(training_parameters_filename)
    train(model_type, parameters)
