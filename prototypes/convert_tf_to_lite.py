"""
Copyright (C) Tu/e.

@author Tuvi Purevsuren t.purevsuren@tue.nl

======================================================
This converter takes a TensorFlow trained model and generates a TensorFlow Lite model into float32 and int8.
The TensorFlow Lite model has (.ftlite) file extension (Flatbuffer format),

"""
import os
import sys
import joblib
import tensorflow as tf
import numpy as np

def convert_saved_models(parameters):
    """Convert tensorflow model (.pb) into a TensorFlow Lite model.

    Args:
        parameters: A dictionary of parameters.

    """
    # find tone in prod_models folder
    saved_model_dir_path = os.path.join(os.path.abspath('prod_models'), parameters['model_name'])
 
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir_path)

    save_model(converter, parameters)
    
def quantization_float32(parameters):
    """Convert keras model (.h5) into a TensorFlow Lite model (post-training float32).

    Args:
        parameters: A dictionary of parameters.

    """
    model_path = os.path.join(os.path.abspath('prod_models'), parameters['model_name'])
    model = tf.keras.models.load_model(model_path)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    converter.post_training_quantize = True

    # DEFAULT -  to improve size and latency based on the information provided.
    #         - does the same thing as OPTIMIZE_FOR_SIZE and OPTIMIZE_FOR_LATENCY.
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    tflite_float32_model = converter.convert()

    save_model(tflite_float32_model, parameters)

def save_model(model, parameters):
    """Save a TensorFlow Lite model.

    Args:
        model: A tflite model.
        parameters: A dictionary of parameters.

    """
    folder_path = get_output_folder_path(parameters)
    # save the quantized model:
    with open(folder_path, 'wb') as f:
        f.write(model)
    
def load_model(path):
    """Load a Tensorflow model.

    Args:
        path: a path of the model.
    
    Returns: 
        A model.

    """
    model = tf.keras.models.load_model(path)
    return model
def get_output_folder_path(parameters):
    """Create a director to save TFlite models.

    Args: 
        parameters: A dictionary of parameters. 
    
    Returns: 
        The full path of the TFlite model.

    """
    tflite_dir = os.path.join(os.path.abspath('prod_models'), parameters['output_folder'])
    # if output_folder_name directory not exists, create it.
    if not os.path.exists(tflite_dir): 
        os.makedirs(tflite_dir)

    return os.path.join(tflite_dir, '{name}_{type}_v{version}.tflite'.format(name=parameters['tflite_model_name'], 
                                                                            type=parameters['optimization_type'],
                                                                            version=parameters['version']))

def representative_data_gen():
    """Provide the generator to represent typical large enough input values and
       allow the converter to estimate a dynamic range for all the variable data. 
       (The dataset does not need to be unique compared to the training or evaluation dataset.)

    """
    data_path = os.path.join(os.path.abspath('prod_data'), 'sequential_model/test')

    mfcc = joblib.load(os.path.join(data_path, 'mfcc_test.joblib'))
    mfcc = np.expand_dims(mfcc, axis =2)

    for input_value in tf.data.Dataset.from_tensor_slices(mfcc).batch(1).take(100):
        # Model has only one input so each data point has one element.
        yield [input_value]

def quantization_int8(parameters):
    """Convert keras model (.h5) into a TensorFlow Lite model (post-training float32).

    Args:
        parameters: A dictionary of parameters.

    """
    model_path = os.path.join(os.path.abspath('prod_models'), parameters['model_name'])

    model = load_model(model_path)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.post_training_quantize = True
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # representative_data_gen : provides a set of input data that's large enough to represent typical values
    converter.representative_dataset = representative_data_gen
    # ensure that if any ops can't be quantized, the converter throws an error
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    # Set the input and output tensors to uint8 (APIs added in r2.3)
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8

    tflite_int8_model = converter.convert()

    save_model(tflite_int8_model, parameters)
    
if __name__ == "__main__":
    if len(sys.argv) == 1: 
        print("MODEL: A - SQUENTIAL")
        print("MODEL: B - SIAMESE")
        print("OPTION:  1 - FLOAT32")
        print("OPTION:  2 - INT16")
        print(" To run the converter script... PLEASE PROVIDE VERSION ")
        print(" python convert_ft_to_lite.py [MODEL] [OPTION] [version to save lite model - example: 0_1]")
        exit(0)

    model = sys.argv[1]
    option = sys.argv[2]
    version = sys.argv[3]
    print(model)
    print(option)
    print(version)
    if model == 'A':
        parameters = {
            # resource related parameters
            'model_name': 'final_models/squential/Emotion_Voice_Detection_Model.h5', 
            'tflite_model_name': 'squential'
        }
    else:
        parameters = {
            # resource related parameters
            'model_name': 'final_models/siamese/Emotion_Voice_Detection_Model.h5',
            'tflite_model_name': 'siamese'
        }
    parameters['version'] = version
    parameters['output_folder'] = 'tensorflow_lite'
    
    if option =='1':
        # float32
        parameters['optimization_type'] = 'float32'
        quantization_float32(parameters)
    else:
        # int16
        parameters['optimization_type'] = 'int8'
        quantization_int8(parameters)
      
       
    

    
