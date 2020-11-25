"""
Copyright (C) Tu/e.

@author Tuvi Purevsuren t.purevsuren@tue.nl

======================================================
This converter takes a TensorFlow trained model and generates a TensorFlow Lite model.
The TensorFlow Lite model has (.ftlite) file extension (Flatbuffer format),

"""
import os
import sys
import joblib

import tensorflow as tf
import numpy as np

def get_path(dir_name):
    """Find path of requested directory such as 'models' and 'data' and 'docs'.

    Args:
        dir_name: A name of directory.

    Returns:
        The full path of requested directory.

    """
    path = os.path.join(os.path.dirname(os.path.normpath(os.getcwd())), dir_name)
    if path:
        return path
    else:
        raise Exception("NO DIRECTORY FOUND")

def load_test_data_sequential_model():
    """Load the test data for sequential model.

    Returns:
        mfcc features of test audio file with corresponding labels.
    
    """
    data_path = get_path('prod_data/sequential_model/test')

    mfcc = joblib.load(os.path.join(data_path, 'mfcc_test.joblib'))
    labels = joblib.load(os.path.join(data_path, 'labels_test.joblib'))

    return np.expand_dims(mfcc, axis =2), labels

def execute_lite_model(parameters):
    """Load the TensorFlow lite model.

    Args:
        parameters: A dictionary of parameters.

    """
    model_path = os.path.join(get_path('prod_models/tflite_model'), parameters['lite_model_name'])

    # load the TFLite model and allocate tensors. 
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    mfcc, labels = load_test_data_sequential_model()

    for i in range(len(labels)):
        # input_details[0]['index']  -> the index accepts the input.
        interpreter.set_tensor(input_details[0]['index'], [mfcc[i]])

        # run the inference
        interpreter.invoke()

        # output_details[0]['index'] -> the index provides the output
        output_data = interpreter.get_tensor(output_details[0]['index'])
        pred_emotion = []
        for e in output_data[0]:
            pred_emotion.append('{:.2f}'.format(e))

        print('Label {label} ==> prediction {pred}.'.format(label=labels[i], pred=pred_emotion))

if __name__ == "__main__": 
    parameters = {
        'lite_model_name': 'tone_cnn_5_emotions_v0_2.tflite'
    }
    execute_lite_model(parameters)