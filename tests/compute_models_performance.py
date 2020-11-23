import json
import yaml
import os
import time

import keras
import tensorflow as tf

import numpy as np
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score, f1_score, classification_report
from librosa.feature import mfcc

from context import tone_emotions
from context import text_emotions
from context import SequentialToneModelDataLoader, SiameseToneModelDataLoader, TextModelDataLoader


def compute_text_model_performance(model_dir, model_name, model_type, test_data_dir):
    """[summary]

    Args:
        model_dir ([type]): [description]
        test_data_dir ([type]): [description]
    """

    print("compute_text_model_performance")    
    # load model
    script_dir = os.path.dirname(os.path.abspath(__file__))

    print("Loading text model...")
    model = load_text_model(model_dir, script_dir)

    test_data_dir_path = os.path.normpath(os.path.join(script_dir, "../prod_data/test/test_encoded_sentences.npz"))
    print(model.summary())
    
    predictions = predict_text_model(model, test_data_dir_path)
    print(predictions)
    # load data
    print("Load: {}".format(test_data_dir))   
    return None 


def compute_measures(predicted_values, truth_values):
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
    accuracy = accuracy_score(truth_values, predicted_sentiment, normalize=False)
    print("Accuracy: {}".format(accuracy))
    
    measures['general_metrics'] = {}
#   measures['perclass_metrics'] = {}

    measures['general_metrics']['accuracy'] = float(accuracy)

    # Confusion matrix for all categories
    # Position 0,0 ---> TN
    # Position 1,0 ---> FN
    # Position 1,1 ---> TP
    # Position 0,1 ---> FP
    mcm = multilabel_confusion_matrix(truth_values, predicted_sentiment)
    print("Multilabel confusion matrix: {}".format(mcm))
    
    

    # Perclass accuracy of the model
    cr = classification_report(truth_values, predicted_sentiment)
    print(cr)

    # F1 Score
    f1 = f1_score(truth_values, predicted_sentiment, average = 'weighted')
    print("F1 Score: {}".format(f1))
    measures['general_metrics']['f1_score'] = float(accuracy)


    # Confidence 
    print(predicted_values)
    i = 0
    confidence = []
    for prediction in predicted_sentiment:
        confidence.append(predicted_values[i][prediction])
        i = i + 1
    print("Condidence: {}". format(confidence))
 
    return measures


def load_text_model(model_dir, script_dir):
    full_path_to_dir = os.path.normpath(os.path.join(script_dir, "../prod_models/candidate", model_dir))
    print("Full path to directory: {}".format(full_path_to_dir))
    model = tf.keras.models.load_model(full_path_to_dir)
    return model

def load_tone_model(model_dir, model_name, script_dir):
    full_path_to_model =  os.path.normpath(os.path.join(script_dir, "../prod_models/candidate", model_dir, model_name))
    print("Full path to model: {}".format(full_path_to_model))
    model = keras.models.load_model(full_path_to_model)
    return model

def predict_sequential_tone_model(model, test_data_dir_path, mfcc_features):
    print(mfcc_features.shape)
    return np.squeeze(model.predict(mfcc_features))

def predict_siamese_tone_model(model, test_data_dir_path, mfcc_features, lmfe_features):
    print(mfcc_features.shape)
    print(lmfe_features.shape)
    return np.squeeze(model.predict([mfcc_features, lmfe_features]))


def predict_text_model(model, test_data_dir_path):
    data_loader = TextModelDataLoader()
    sentences = data_loader.load_test_data(test_data_dir_path)
    #print(sentences['arr_0'])
    #print(sentences.shape)
    return model.predict(np.squeeze(sentences))


def compute_tone_model_performance(model_dir, model_name, model_type, test_data_dir):
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
        data_loader = SequentialToneModelDataLoader()
        mfcc_features, labels = data_loader.load_test_data(test_data_dir_path)
        print(mfcc_features.shape)
        print(labels.shape)
        predictions = predict_sequential_tone_model(model, test_data_dir_path, mfcc_features)
    elif "Siamese" in model_type:
        data_loader = SiameseToneModelDataLoader()
        mfcc_features, lmfe_features, labels, labels_lmfe = data_loader.load_test_data(test_data_dir_path)
        predictions = predict_siamese_tone_model(model, test_data_dir_path, mfcc_features, lmfe_features)
    else:
        raise ValueError("Value in configuration file {} is not supported".format('type'))

   
    
    # load data
    # print("Load: {}".format(test_data_dir))
    print("Printing predictions")
    print(predictions)
    print(labels)
    
    # compute performance
    #return None
    performance_metrics = compute_measures(predictions, labels)
    performance_metrics['model_name'] = model_name
    performance_metrics['model_type'] = model_type
    # TODO calculate performance
    performance_metrics['general_metrics']['performance'] = -1
    #print(performance_metrics)
    print(json.dumps(performance_metrics, indent=4))

    return performance_metrics

def save_performance_results(performance_results):
    """[summary]

    Args:
        performance_results ([type]): [description]
    """    

    script_dir = os.path.dirname(os.path.abspath(__file__))
    print(script_dir)
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
            performance_results[key] = compute_tone_model_performance(models[key]['dir'], models[key]['file'], models[key]['type'], test_data_dir)
            print("===============================================================")
            print(performance_results[key])
        elif "text_model" in key:
            pass
#            print("Computing performance of test model...")
#            performance_results[key] = compute_text_model_performance(models[key]['dir'], models[key]['file'], models[key]['type'], test_data_dir)
        else:
            raise ValueError("Value in configuration file {} is not supported".format('type'))
###
    save_performance_results(performance_results)


if __name__ == "__main__":
    
    prod_config_file = "raspi_candidate_config.yml"
    generate_performance_summary(prod_config_file)