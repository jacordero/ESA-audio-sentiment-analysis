import os
import yaml
import time
from sequential_model_generator import SequentialModelGeneratorFactory
from siamese_model_generator import SiameseModel
from data_loader import SequentialToneModelDataLoader, SiameseToneModelDataLoader
import keras
import numpy as np
from pathlib import Path



def create_trained_model_path(trained_models_dir, trained_model_name):

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


def train_sequential_model(parameters):
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

    if parameters["n_emotions"] != len(np.unique(labels_train)):
        message = "The value n_emotions and the number of labels in the training dataset are different.\n"
        message += "n_emotions: {}, number of labels: {}".format(parameters["n_emotions"], len(np.unique(labels_train)))
        raise ValueError(message)

    # create model
    parameters["input_shape"] = [mfcc_train.shape[1], mfcc_train.shape[2]]

    model_generator_factory = SequentialModelGeneratorFactory()
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
    history = model.fit(x=[mfcc_train], y=labels_train, batch_size=parameters["batch_size"], epochs=parameters["epochs"],
                         validation_data=(mfcc_val, labels_val))

    # save model
    trained_model_dir_path = create_trained_model_path(parameters['trained_models_dir'], parameters['trained_model_name'])
    save_model(model, trained_model_dir_path)

    save_training_info(model_type, parameters, history)

def train_siamese_model(parameters):
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
    #print(parameters["n_conv_filters"])
    #print(parameters["filters_shape"])
    print("Shape: {}".format(mfcc_train.shape))
    parameters["input_shape"] = [mfcc_train.shape[1], mfcc_train.shape[2]]

    siamese_model = SiameseModel()
    mfcc_input, mfcc_output = siamese_model.create_siamese_branch_architecture(mfcc_train.shape[1], mfcc_train.shape[2])
    lmfe_input, lmfe_output = siamese_model.create_siamese_branch_architecture(lmfe_train.shape[1], lmfe_train.shape[2])
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
    trained_model_path = create_trained_model_path(parameters['trained_models_dir'], parameters['trained_model_name'])
    model.save(trained_model_path)

    save_training_info(model_type, parameters, history)


def train(model_type, parameters):
    if model_type == "Sequential":
        train_sequential_model(parameters)
    elif model_type == "Siamese":
        train_siamese_model(parameters)
    else:
        raise ValueError("Invalid model type: {}".format(model_type))

def load_training_parameters(training_parameters_filename):
    
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

    training_parameters_filename = "training_parameters.yml"
    model_type, parameters = load_training_parameters(training_parameters_filename)
    train(model_type, parameters)
