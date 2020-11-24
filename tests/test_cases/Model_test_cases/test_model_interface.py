# test_model_interface.py

import pytest
import yaml
import os
import json
import keras
import tensorflow as tf


# load model
def load_tone_model(model_dir, model_name, script_dir):
    full_path_to_model = os.path.normpath(
        os.path.join(
            script_dir, "../prod_models/candidate", model_dir, model_name))
    print("Full path to model: {}".format(full_path_to_model))
    model = keras.models.load_model(full_path_to_model)
    print(model.input_shape)
    print(model.output_shape)
    return model


# load configuration files
def load_configuration(filename):
    prod_config_file = filename
    with open(prod_config_file) as input_file:
        config_parameters = yaml.load(input_file, Loader=yaml.FullLoader)
    return config_parameters


def test_model_input_interface():
    # load configuration file
    # Configuration.json will be provided
    # by the architect or TM with all the needed information
    conf_parameters = load_configuration("tests/configuration.yml")
    # load candidate configuration
    # This file contains all the information for the candidate model
    candidate = load_configuration("src/raspi_candidate_config.yml")['model']

    # This may change since we don't have text model any more
    model = load_tone_model(
        candidate['dir'],
        candidate['file'],
        candidate['type'])

    # get input shape of candidate model
    input_shape = str(model.input_shape)

    # assert input shape of model with configuration input shape
    if candidate['type'] == "Siamese":
        conf_input_shape = conf_parameters['siamese_model']['input_shape']
    elif candidate['type'] == "Sequential":
        conf_input_shape = conf_parameters['sequential_model']['input_shape']

    assert conf_input_shape == input_shape

def test_model_output_interface():
    # load configuration file
    # Configuration.json will be provided
    # by the architect or TM with all the needed information
    conf_parameters = load_configuration("tests/configuration.yml")
    # load candidate configuration
    # This file contains all the information for the candidate model
    candidate = load_configuration("src/raspi_candidate_config.yml")['model']
    print(candidate)

    # This may change since we don't have text model any more
    model = load_tone_model(
        candidate['dir'],
        candidate['file'],
        candidate['type'])

    # get output shape of candidate model
    output_shape = str(model.output_shape)

    # assert output shape of model with configuration output shape
    if candidate['type'] == "Siamese":
        conf_output_shape = conf_parameters['siamese_model']['output_shape']
    elif candidate['type'] == "Sequential":
        conf_output_shape = conf_parameters['sequential_model']['output_shape']

    assert conf_output_shape == output_shape