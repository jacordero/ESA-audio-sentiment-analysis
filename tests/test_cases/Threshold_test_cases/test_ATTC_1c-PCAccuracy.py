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

def test_ATTC_1a():
    conf_parameters = load_configuration("tests/configuration.yml")
    candidate_model_performance = load_model_performance("tests/candidate_models_performance.json")

    class_accuracies = candidate_model_performance["perclass_metrics"]["accuracy"]
    minimal_accuracy = conf_parameters["target_metrics"]["per_class_accuracy"]

    for label, class_accuracy in class_accuracies.items():
        assert label and class_accuracy >= minimal_accuracy