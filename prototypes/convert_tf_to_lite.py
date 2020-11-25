"""
Copyright (C) Tu/e.

@author Tuvi Purevsuren t.purevsuren@tue.nl

======================================================
This converter takes a TensorFlow trained model and generates a TensorFlow Lite model.
The TensorFlow Lite model has (.ftlite) file extension (Flatbuffer format),

"""
import os
import sys
import tensorflow as tf

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


def convert_saved_models(parameters):
    """Convert tensorflow model (.pb) into a TensorFlow Lite model.

    Args:
        parameters: A dictionary of parameters.

    """
    # find tone in prod_models folder
    saved_model_dir_path = os.path.join(get_path('prod_models'), parameters['model_name']) 
    print(saved_model_dir_path)
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir_path)

    save_model(converter, parameters)
    
def convert_keras_model(parameters):
    """Convert keras model (.h5) into a TensorFlow Lite model.

    Args:
        parameters: A dictionary of parameters.

    """
    saved_model_dir_path = os.path.join(get_path('prod_models'), parameters['model_name']) 

    model = tf.keras.models.load_model(saved_model_dir_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    save_model(converter, parameters)

def save_model(converter, parameters):
    """Save a TensorFlow Lite model.

    Args:
        converter: An object of TFLite
        parameters: A dictionary of parameters.

    """
    tflite_model = converter.convert()

    # save the lite model TODO: DISCUSS FOLDER FOR TENSORFLOW LITE - currently lite model is saved in models/lite
    lite_dir = os.path.join(get_path('models'), 'tensorflow_lite')
    print(lite_dir)
    # if lite directory not exists, create it.
    if not os.path.exists(lite_dir): 
        os.makedirs(lite_dir)
    
    lite_model_name = '{name}_v{version}.tflite'.format(name=parameters['lite_model_name'], version=parameters['version'])
    
    saved_lite_model_path = os.path.join(lite_dir, lite_model_name)
    
    # save the lite model
    with open(saved_lite_model_path, 'wb') as f:
        f.write(tflite_model)

if __name__ == "__main__":
    if len(sys.argv) == 1: 
        print("OPTION:  1 - SavedModel format (.pb)")
        print("OPTION:  2 - Keras Model (.h5)")
        print(" To run the converter script... PLEASE PROVIDE VERSION ")
        print(" python convert_ft_to_lite.py  [version to save lite model - example: 0_1] [OPTION]")
        exit(0)
    option = sys.argv[2]
    print(option)
    if option == '1':
        parameters = {
            # resource related parameters
            'model_name': 'deployed/seq_3conv_modules_be_nice', 
            'lite_model_name': 'seq_3conv_modules_be_nice',
            'version': sys.argv[1]
        }
        convert_saved_models(parameters)
    else:
        parameters = {
            # resource related parameters
            'model_name': 'deployed/tone_cnn_8_emotions_v0_1/saved_models/Emotion_Voice_Detection_Model.h5', 
            'lite_model_name': 'tone_cnn_8_emotions',
            'version': sys.argv[1]
        }
        convert_keras_model(parameters)
    

    
