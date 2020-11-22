# test_model_interface.py

import pytest
import yaml
import os
import json
import keras
import tensorflow as tf


# load model
def load_tone_model(model_dir, model_name, script_dir):
    full_path_to_model =  os.path.normpath(os.path.join(script_dir, "../prod_models/candidate", model_dir, model_name))
    print("Full path to model: {}".format(full_path_to_model))
    model = keras.models.load_model(full_path_to_model)
    print(model.input_shape)
    print(model.output_shape)
    return model


# load configuration files
def load_configuration(filename):
 #   prod_config_file = "tests/test_cases/configuration.yml"
    prod_config_file = filename
    with open(prod_config_file) as input_file:
        config_parameters = yaml.load(input_file, Loader=yaml.FullLoader)
        #config_parameters = json.load(filename)
    return config_parameters


def test_model_interface():
    # load configuration file (Configuration.json will be provided by the architect or TM with all the needed information)
    configuration_parameters = load_configuration("tests/configuration.yml")
    
    # load candidate configuration (This file contains all the information for the candidate model)
    candidate = load_configuration("src/raspi_candidate_config.yml")

    # load candidate model
    models = candidate['models']
    # This may change since we don't have text model any more
    model = load_tone_model(models['tone_model']['dir'], models['tone_model']['file'], models['tone_model']['type'])

    # get input and output shape of candidate model
    input_shape = str(model.input_shape)
    output_shape = str(model.output_shape)

    # assert values from file with values from model
    # Double assertion is a bad habbit to have on a test case
    # However, here it is overkill to create a second test case for asserting just the output_shape
    # I will discuss it with the architect and CM. But as for now it will stay as is.
    if models['tone_model']['type'] == "Siamese":
        assert input_shape == configuration_parameters['siamese_model']['input_shape']
        assert output_shape == configuration_parameters['siamese_model']['output_shape']
    elif models['tone_model']['type'] == "Sequential":
        assert input_shape == configuration_parameters['sequential_model']['input_shape']
        assert output_shape == configuration_parameters['sequential_model']['output_shape']
