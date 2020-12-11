import pytest
import yaml
import os
import json

def load_configuration(filename):
    prod_config_file = filename
    with open(prod_config_file) as input_file:
        config_parameters = yaml.load(input_file, Loader=yaml.FullLoader)
    return config_parameters

def load_model_performance(filename):
    with open(filename, "r") as input_file:
        model_performance = json.load(input_file)
    return model_performance

def test_ATTC_5():
    conf_parameters = load_configuration("tests/configuration.yml")
    candidate_model_performance = load_model_performance("tests/candidate_models_performance.json")

    model_confidence = candidate_model_performance["general_metrics"]["confidence"]
    minimal_confidence = conf_parameters["target_metrics"]["confidence"]

    assert model_confidence >= minimal_confidence