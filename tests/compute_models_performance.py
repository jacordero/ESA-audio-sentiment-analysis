import json
import yaml
import os
import time
from statistics import mean

import keras
import tensorflow as tf

import numpy as np
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score, f1_score, classification_report
from context import SequentialDataLoader, SiameseDataLoader, TextModelDataLoader



def compute_measures(predicted_values, truth_values, emotions):
    """[summary]

    Args:
        predicted_values ([type]): [description]
        truth_values ([type]): [description]
    """    

    '''
    {
        model_name: 
        model_type:
        general_metrics:
            accuracy:
            f1_score:
            performance:
            confidence:
        perclass_metrics:
            confidence:
                happy:
                sad:
                .
                .

            accuracy:
                happy:
                sad:
                .
                .

    }
    '''

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
    full_path_to_model =  os.path.normpath(os.path.join(script_dir, "../prod_models/candidate", model_dir, model_name))
    print("Full path to model: {}".format(full_path_to_model))
    model = keras.models.load_model(full_path_to_model)
    return model

def predict_sequential_tone_model(model, mfcc_features):
    return np.squeeze(model.predict(mfcc_features))

def predict_siamese_tone_model(model, mfcc_features, lmfe_features):
    return np.squeeze(model.predict([mfcc_features, lmfe_features]))


def predict_text_model(model, test_data_dir_path):
    data_loader = TextModelDataLoader()
    sentences = data_loader.load_test_data(test_data_dir_path)
    return model.predict(np.squeeze(sentences))


def compute_tone_model_performance(model_dir, model_name, model_type, test_data_dir, emotions):
    """[summary]

    Args:
        model_dir ([type]): [description]
        test_data_dir ([type]): [description]
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