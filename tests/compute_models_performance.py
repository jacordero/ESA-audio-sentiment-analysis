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
from context import SequentialToneModelDataLoader, SiameseToneModelDataLoader


def compute_text_model_performance(model_dir, model_name, model_type, test_data_dir):
    """[summary]

    Args:
        model_dir ([type]): [description]
        test_data_dir ([type]): [description]
    """

    print("compute_text_model_performance")    

    # load model
    script_dir = os.path.dirname(os.path.abspath(__file__))
    full_path_to_dir =  os.path.normpath(os.path.join(script_dir, "../prod_models/candidate", model_dir))
    full_path_to_model =  os.path.normpath(os.path.join(script_dir, "../prod_models/candidate", model_dir, model_name))
    print("Full path to directory: {}".format(full_path_to_dir))
    print("Full path to model: {}".format(full_path_to_model))

    if "Sequential" in model_type:
        print("Loading sequential text model...")
        model = keras.models.load_model(full_path_to_dir)
        data_loader = SequentialToneModelDataLoader()
    elif "Siamese" in model_type:
        print("Loading siamese text model...")
        model = keras.models.load_model(full_path_to_model)
        data_loader = SiameseToneModelDataLoader()
    else:
        raise ValueError("Value in configuration file {} is not supported".format('type'))

    print(model.summary())

    # load data
    print("Load: {}".format(test_data_dir))
 #   data_loader = SequentialToneModelDataLoader()
 #   audios, labels = data_loader.load_test_data(test_data_dir)
 #   predictions = np.squeeze(model.predict(audios))
    
    # compute performance
#    return compute_measures(predictions, labels)
    return None

def compute_measures(predicted_values, truth_values):
    """[summary]

    Args:
        predicted_values ([type]): [description]
        truth_values ([type]): [description]
    """    

    # Confusion matrix for all categories


    # Total accuracy of the model

    # Perclass accuracy of the model

    # F1 Score

    # Confidence 

    return None


def compute_tone_model_performance(model_dir, model_name, model_type, test_data_dir):
    """[summary]

    Args:
        model_dir ([type]): [description]
        test_data_dir ([type]): [description]
    """

    print("compute_tone_model_performance")    
    # load model
    script_dir = os.path.dirname(os.path.abspath(__file__))
    full_path_to_dir = os.path.normpath(os.path.join(script_dir, "../prod_models/candidate", model_dir))
    full_path_to_model =  os.path.normpath(os.path.join(script_dir, "../prod_models/candidate", model_dir, model_name))
    print("Full path to directory: {}".format(full_path_to_dir))
    print("Full path to model: {}".format(full_path_to_model))
    
    if "Sequential" in model_type:
        print("Loading sequential tone model...")
        model = keras.models.load_model(full_path_to_dir)
        data_loader = SequentialToneModelDataLoader()
    elif "Siamese" in model_type:
        print("Loading siamese tone model...")
        model = keras.models.load_model(full_path_to_model)
        data_loader = SiameseToneModelDataLoader()
    else:
        raise ValueError("Value in configuration file {} is not supported".format('type'))

    print(model.summary())
    
    time.sleep(5)
    return 

    # load data
    print("Load: {}".format(test_data_dir))
    audios, labels = data_loader.load_test_data(test_data_dir)
    predictions = np.squeeze(model.predict(audios))
    
    # compute performance
    return compute_measures(predictions, labels)


def compute_model_performance(model_dir, model_name, model_type, test_data_dir):
    """[summary]

    Args:
        model_dir ([type]): [description]
        test_data_dir ([type]): [description]
    """

    print("compute_tone_model_performance")    
    # load model
    script_dir = os.path.dirname(os.path.abspath(__file__))
    full_path_to_dir = os.path.normpath(os.path.join(script_dir, "../prod_models/candidate", model_dir))
    full_path_to_model =  os.path.normpath(os.path.join(script_dir, "../prod_models/candidate", model_dir, model_name))
    print("Full path to directory: {}".format(full_path_to_dir))
    print("Full path to model: {}".format(full_path_to_model))
    
    if "Sequential" in model_type:
        print("Loading sequential model...")
        model = keras.models.load_model(full_path_to_dir)
        data_loader = SequentialToneModelDataLoader()
    elif "Siamese" in model_type:
        print("Loading siamese model...")
        model = keras.models.load_model(full_path_to_model)
        data_loader = SiameseToneModelDataLoader()
    else:
        raise ValueError("Value in configuration file {} is not supported".format('type'))

    print(model.summary())
    
    time.sleep(5)
    return 

    # load data
    print("Load: {}".format(test_data_dir))
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
        if "tone_model" in key:
            print("Computing performance of tone model...")
            performance_results[key] = compute_model_performance(models[key]['dir'], models[key]['file'], models[key]['type'], test_data_dir)
        elif "text_model" in key:
            print("Computing performance of test model...")
            performance_results[key] = compute_model_performance(models[key]['dir'], models[key]['file'], models[key]['type'], test_data_dir)
        else:
            raise ValueError("Value in configuration file {} is not supported".format('type'))
###
#    save_performance_results(performance_results)


if __name__ == "__main__":
    
    prod_config_file = "src/raspi_candidate_config.yml"
    generate_performance_summary(prod_config_file)