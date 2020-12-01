"""
Copyright (C) Tu/e.

@author Tuvi Purevsuren t.purevsuren@tue.nl

======================================================
To run the TFlite models and evaluate the performance (accuracy and prediction time).

"""
import os
import sys
import joblib
import resource

import tensorflow as tf
import numpy as np
import matplotlib.pylab as plt

def load_test_data(type):
    """Load the test data for sequential and siamese models.

    Args:
        type: A type of the TFlite model.

    Returns:
        mfcc and lmfe features of test audio file with corresponding labels.
    
    """
    if type == 'sequential':
        data_path = os.path.abspath('prod_data/test/sequential')
        mfcc = joblib.load(os.path.join(data_path, 'mfcc_test.joblib'))
        labels = joblib.load(os.path.join(data_path, 'labels_test.joblib'))

        return np.expand_dims(mfcc, axis =2), labels

    else:
        data_path = os.path.abspath('prod_data/test/siamese')
        labels = joblib.load(os.path.join(data_path, 'labels_test.joblib'))

        mfcc = joblib.load(os.path.join(data_path, 'mfcc_test.joblib'))
        lmfe = joblib.load(os.path.join(data_path, 'lmfe_test.joblib'))

        mfcc = np.expand_dims(mfcc, axis=3)
        mfcc.astype('float32')

        lmfe =  np.expand_dims(lmfe, axis=3)
        lmfe.astype('float32')

        return mfcc, lmfe, labels

def evaluate_model(interpreter, mfcc, labels, predictions):
    """Evalute the model accuracy. 

    Args: 
        interpreter: An object of TFlite interpreter. 
        mfcc: An array of mfcc feature.
        labels: A label of test data.
        predictions: A result of the model on the test data.
    """
    input_type = interpreter.get_input_details()[0]['dtype']
    #print("INPUT:", input_type)
    output_type = interpreter.get_output_details()[0]['dtype']
    #print("OUTPUT:", output_type)
    accuracy = (np.sum(labels == predictions) * 100)/ len(labels)

    print('The TFlite model accuracy is {accuracy:.4f} (number of test data: {test})'.format(accuracy=accuracy, test = len(labels)))

def execute_lite_model(model, type):
    """Run the TensorFlow lite models.

    Args:
        model: A model.
        type: A type of the model (sequential or siamese).

    """
    start = resource.getrusage(resource.RUSAGE_SELF)
    if type == 'A':
        model_path = os.path.join(os.path.abspath('prod_models/tensorflow_lite/sequential'), model)
        mfcc, labels = load_test_data('sequential')
    
    else:
        model_path = os.path.join('prod_models/tensorflow_lite/siamese', model)
        mfcc, lmfe, labels = load_test_data('siamese')

    # load the TFLite model and allocate tensors. 
    interpreter = tf.lite.Interpreter(model_path=model_path)

    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    predictions = []
    if type == 'A':
        for i in range(len(labels)):
            # input_details[0]['index']  -> the index accepts the input.
            interpreter.set_tensor(input_details[0]['index'], [mfcc[i]])
            # run the inference
            interpreter.invoke()
            # output_details[0]['index'] -> the index provides the output
            output_data = interpreter.get_tensor(output_details[0]['index'])
            #pred_emotion = []
            #for e in output_data[0]:
            #    pred_emotion.append('{:.2f}'.format(e))
            predictions.append(output_data.argmax())
            #print('Label {label} ==> prediction {pred}.'.format(label=labels[i], pred=pred_emotion))
    else:
        lmfe = np.float32(lmfe)
        for i in range(len(labels)):
            interpreter.set_tensor(input_details[0]['index'], [mfcc[i]])
            interpreter.set_tensor(input_details[1]['index'], [lmfe[i]])
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])
            predictions.append(output_data.argmax())
    end = resource.getrusage(resource.RUSAGE_SELF)
    print("==================================================================")
    print("User CPU time = {} seconds.".format(end[0] - start[0]))
    print("System CPU time = {} seconds.".format(end[1] - start[1]))
    print("==================================================================")
    print("MEMORY = {} kilobytes.".format(end[2] - start[2]))

    evaluate_model(interpreter, mfcc, labels, predictions)
if __name__ == "__main__": 
    if len(sys.argv) == 1: 
        print("MODEL: A - SEQUENTIAL")
        print("MODEL: B - SIAMESE")
        print("OPTION:  1 - FLOAT32")
        print("OPTION:  2 - INT8")
        print(" To run the converter script... PLEASE PROVIDE VERSION ")
        print(" python convert_ft_to_lite.py [MODEL] [OPTION] [version to save lite model - example: 0_1]")
        exit(0)

    model_type = sys.argv[1]
    option = sys.argv[2]

    model_option = {
        'A1': 'sequential_float32_v1.tflite', 
        'A2': 'sequential_int8_v1.tflite',
        'B1': 'siamese_float32_v1.tflite', 
        'B2': 'siamese_int8_v1.tflite'
    }
    model = sys.argv[1] + sys.argv[2]
    execute_lite_model(model_option[model], model_type)