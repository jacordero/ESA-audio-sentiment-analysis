"""
Copyright (c) 2020 TU/e - PDEng Software Technology C2019. All rights reserved.
@ Authors: George Azis gazis@tue.nl; Jorge Cordero j.a.cordero.cruz@tue.nl;
@ Contributors: Niels Rood n.rood@tue.nl;
Last modified: 07-12-2020
"""

import json
import yaml
import os
import time
from statistics import mean

import keras
import tensorflow as tf

import numpy as np
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score, f1_score, classification_report
from context import SequentialDataLoader, SiameseDataLoader



def compute_measures(predicted_values, truth_values, emotions):
    """This function will perform an assessment of the model
        based on the truth values. I will return a dictionary,
        which contains all the fields of this assessment.

    Args:
        predicted_values (list): candidate model predictions of the testing data
        truth_values (list): labels of the testing
        emotions (dictionary): dictionary of the predefined set of classes

    Returns:
        A dictionary with the following schema
        {
        general_metrics:
            accuracy: num(max=1.0, min=0.0)
            f1_score: num(max=1.0, min=0.0)
            confidence: num(max=1.0, min=0.0)
        perclass_metrics:
            confidence:
                neutral: num(max=1.0, min=0.0)
                happy: num(max=1.0, min=0.0)
                sad: num(max=1.0, min=0.0)
                angry: num(max=1.0, min=0.0)
                fearful: num(max=1.0, min=0.0)
           accuracy:
                neutral: num(max=1.0, min=0.0)
                happy: num(max=1.0, min=0.0)
                sad: num(max=1.0, min=0.0)
                angry: num(max=1.0, min=0.0)
                fearful: num(max=1.0, min=0.0)
        }
    """

    measures = dict()

    # Total accuracy of the model
    predicted_sentiment = np.argmax(predicted_values, axis = 1)
    truth_values = truth_values.astype(int)
    
    # Total accuracy
    accuracy = accuracy_score(truth_values, predicted_sentiment, normalize=True)
    
    measures['general_metrics'] = {}
    measures['perclass_metrics'] = {}
    measures['general_metrics']['accuracy'] = float(accuracy)

    # F1 Score
    f1 = f1_score(truth_values, predicted_sentiment, average = 'weighted')
    measures['general_metrics']['f1_score'] = float(f1)

    # Confidence
    sum = 0.0
    for prediction,truth in zip(predicted_values, truth_values):
        sum += prediction[truth]
    avg = sum / len(truth_values)
    measures['general_metrics']['confidence'] = float(avg)

    # Per class confidence
    label_lists = {}
    for label in set(truth_values):
        label_lists[label] = []

    for prediction, truth in zip(predicted_values, truth_values):
        label_lists[truth].append(prediction[truth])

    avg = {l: mean(v) for l, v in label_lists.items()}
    avg = {emotions[k]: float(v) for k,v in avg.items()}
    measures['perclass_metrics']['confidence'] = avg

    # Per class accuracy
    for label in set(truth_values):
        label_lists[label] = []

    for prediction, truth in zip(np.argmax(predicted_values, axis = 1), truth_values):
        if (prediction == truth):
            label_lists[truth].append(1)
        else:
            label_lists[truth].append(0)
    
    avg = {l: mean(v) for l, v in label_lists.items()}
    avg = {emotions[k]: float(v) for k,v in avg.items()}
    measures['perclass_metrics']['accuracy'] = avg

    return measures

def load_tone_model(model_dir, model_name, script_dir):
    """This function loads the tone model based on the input
        of the configuration file.
    Args:
        model_dir (string): candidate model directory
        model_name (string): candidate model name
        script_dir (string): script directory

    Returns:
        Model
    """

    full_path_to_model =  os.path.normpath(os.path.join(script_dir, "../prod_models/candidate", model_dir, model_name))
    print("Full path to model: {}".format(full_path_to_model))
    model = keras.models.load_model(full_path_to_model)
    return model

def predict_sequential_tone_model(model, mfcc_features):
    """This function performs a prediction based on the mfcc features
        with the sequential tone model.

    Args:
        model (model): candidate model
        mfcc_features (list): mfcc features of the testing data set

    Returns:
        List of predictions based on testing data set.
    """  
    return np.squeeze(model.predict(mfcc_features))

def predict_siamese_tone_model(model, mfcc_features, lmfe_features):
    """This function performs a prediction based on the mfcc and lmfe features
        with the siamese tone model.
        
    Args:
        model (model): candidate model
        mfcc_features (list): mfcc features of the testing data set
        lmfe_features (list): lmfe features of the testing data set

    Returns:
        List of predictions based on testing data set.
    """  
    return np.squeeze(model.predict([mfcc_features, lmfe_features]))


def compute_tone_model_performance(model_dir, model_name, model_type, test_data_dir, emotions):
    """This function performns the assessment of the model based on the testing data.

    Args:
        model_dir (string): candidate model directory
        model_name (string): candidate model name
        model_type (string): candidate model type
        test_data_dir(string): testing data directory
        emotions(dictionary): dictionary of the predefined set of classes

    Returns:
        A dictionary with the following schema
        {
        general_metrics:
            accuracy: num(max=1.0, min=0.0)
            f1_score: num(max=1.0, min=0.0)
            performance: num(min=0.0)
            confidence: num(max=1.0, min=0.0)
        perclass_metrics:
            confidence:
                neutral: num(max=1.0, min=0.0)
                happy: num(max=1.0, min=0.0)
                sad: num(max=1.0, min=0.0)
                angry: num(max=1.0, min=0.0)
                fearful: num(max=1.0, min=0.0)
           accuracy:
                neutral: num(max=1.0, min=0.0)
                happy: num(max=1.0, min=0.0)
                sad: num(max=1.0, min=0.0)
                angry: num(max=1.0, min=0.0)
                fearful: num(max=1.0, min=0.0)
        }        
    """

    print("compute_tone_model_performance")    
    # load model
    script_dir = os.path.dirname(os.path.abspath(__file__))

    print("Loading tone model...")
    model = load_tone_model(model_dir, model_name, script_dir)

    test_data_dir_path = os.path.normpath(os.path.join(script_dir, "../prod_data/test"))
    print(model.summary())

    if "Sequential" in model_type:
        data_loader = SequentialDataLoader()
        mfcc_features, labels = data_loader.load_test_data(test_data_dir_path)
        start = time.time_ns()
        predictions = predict_sequential_tone_model(model, mfcc_features)
        end = time.time_ns()
    elif "Siamese" in model_type:
        data_loader = SiameseDataLoader()
        mfcc_features, lmfe_features, labels = data_loader.load_test_data(test_data_dir_path)
        start = time.time_ns()
        predictions = predict_siamese_tone_model(model, mfcc_features, lmfe_features)
        end = time.time_ns()
    else:
        raise ValueError("Value in configuration file {} is not supported".format('type'))



    
    # load data   
    performance_metrics = compute_measures(predictions, labels, emotions)
    performance_metrics['model_name'] = model_name
    performance_metrics['model_type'] = model_type
    time_in_s_per_input = (end - start) / ((10 ** 9)*(labels.shape[0]))
    performance_metrics['general_metrics']['performance'] = time_in_s_per_input
    print(json.dumps(performance_metrics, indent=4))

    return performance_metrics

def save_performance_results(performance_results):
    """This function saves the results of the candidate assessment
        to a json file.

    Args:
        performance_results (dictionary): Results of the candidate model assessment
    """

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_filename = os.path.join(script_dir, "candidate_models_performance.json")
    with open(output_filename, 'w') as output_file:
        json.dump(performance_results, output_file)    


def generate_performance_summary(prod_config_file):
    """This function loads the configurations from the file, computes the
        candidate model assessment, and saves the results.

    Args:
        prod_config_file (string): Path to configuration file
    """

    with open(prod_config_file) as input_file:
        prod_config_parameters = yaml.load(input_file, Loader=yaml.FullLoader)
    
    model = prod_config_parameters['model']
    test_data_dir = prod_config_parameters['test_data_dir']
    emotions = prod_config_parameters['emotions']
    performance_results = {}


    print("Computing performance of tone model...")
    performance_results = compute_tone_model_performance(model['dir'], model['file'], model['type'], test_data_dir, emotions)

    save_performance_results(performance_results)


if __name__ == "__main__":
    
    prod_config_file = "src/raspi_candidate_config.yml"
    generate_performance_summary(prod_config_file)