import json
import yaml
import os
import time

import keras
import tensorflow as tf

import numpy as np
from librosa.feature import mfcc

from context import tone_emotions
from context import text_emotions
from context import SequentialToneModelDataLoader


def compute_text_model_performance(model_dir, test_data_dir):
    """[summary]

    Args:
        model_dir ([type]): [description]
        test_data_dir ([type]): [description]
    """

    print("compute_text_model_performance")    

    # load model
    print("Load: {}".format(model_dir))

    # load data
    print("Load: {}".format(test_data_dir))

    # compute performance
    return None

def compute_measures(predicted_values, truth_values):
    """[summary]

    Args:
        predicted_values ([type]): [description]
        truth_values ([type]): [description]
    """    
    return None

def compute_tone_model_performance(model_name, test_data_dir):
    """[summary]

    Args:
        model_dir ([type]): [description]
        test_data_dir ([type]): [description]
    """

    print("compute_tone_model_performance")    
    # load model
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_full_path =  os.path.normpath(os.path.join(script_dir, "../prod_models/candidate", model_name))
    print("Relative path: {}".format(model_name))
    print("Full path: {}".format(model_full_path))
    
    model = keras.models.load_model(model_full_path)
    print(model.summary())
    
    time.sleep(5)
    return 
    

    # load data
    print("Load: {}".format(test_data_dir))
    data_loader = SequentialToneModelDataLoader()
    audios, labels = data_loader.load_test_data(test_data_dir)
    predictions = np.squeeze(model.predict(audios))
    
    # compute performance
    return compute_measures(predictions, labels)


def save_performance_results(performance_results):
    """[summary]

    Args:
        performance_results ([type]): [description]
    """    

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_filename = os.path.join(script_dir, "candidate_models_performance.json")
    with open(output_filename, 'w') as output_file:
        json.dump(performance_results, output_file)    


def generate_performance_summary(prod_config_file):
    """[summary]

    Args:
        prod_config_file ([type]): [description]
    """    

    with open(prod_config_file) as input_file:
        prod_config_parameters = yaml.load(input_file, Loader=yaml.FullLoader)

    print(prod_config_parameters)

    models = prod_config_parameters['models']
    test_data_dir = prod_config_parameters['test_data_dir']
    performance_results = {}

    for key in models:
        if "tone" in key:
            performance_results[key] = compute_tone_model_performance(models[key], test_data_dir)
            #performance_results[key] = performance
        elif "text" in key:
            pass
#            performance = compute_text_model_performance(models[key], test_data_dir)
        else:
            raise ValueError("Value in configuration file {} is not supported".format(key))

    save_performance_results(performance_results)


if __name__ == "__main__":
    
    prod_config_file = "src/raspi_candidate_config.yml"
    generate_performance_summary(prod_config_file)