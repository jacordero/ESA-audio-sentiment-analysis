# Model Training

This file contains instructions for training the models. One can create either a sequential or non-sequential (simese) model quite dynamically with setting parameters in the *training_parameters.yml* file and executing model_trainer.py script.
extracting features from datasets. 
For training the model you should specify the type of the model, and also directories containing the features, as well as some other parameters described in more details in further sections.
For more information on how to extract features, please read [here](/docs/feature_Extraction.md).

## Configuration File

In the training directory you can find a Yaml file, called _training_parameters.yml_. It contains all the parameters required for creating the model, extracting features, training the model, and saving the model. Below you can learn about each of these paramaters in detail.

 -  ***model_type***: It can be either "Sequential" or "Siamese". It specifies the type of the model to be created.
 -  ***model_generator***: Each model can be created with different convolutional layers/internal structure. Using this parameter, you could more precisely define your desired model.
 -  ***n_conv_filters***:  An array specifying number of filters in each convolutional layers.
 -  ***filters_shape***: An array specifying filter shapes in each convolutional layers.
 -  ***n_emotions***: The number of emotions to be predicted. It should be 5 based on our agreement. Supported emotions are *happy, sad, neutral, fearful, angry.*
 -  ***n_mfcc***: The number of coefficients required for extracting MFCC feature.
 -  ***n_lmfe***: The number of filterbanks to be used in LMFE extraction.
 -  ***adam_lr***: The learning rate of Adam optimizer.
 -  ***batch_size***: The number of bachtes in each epoch
 -  ***epochs***: The number of epochs in training phase
 -  ***trained_models_dir***: The path to the directory from the root directory contating the model (.h5).
 -  ***trained_model_name***: The name of the .h5 model.
 -  ***feature_train_data_directory***: The path to the directory from the root directory  containing joblib files for  training.
 -  ***feature_test_data_directory***: The path to the directory from the root directory  containing joblib files for testing.
 -  ***feature_validation_data_directory***: The path to the directory from the root directory contaning the validation joblib files.
 -  ***raw_train_data_directory***: The path to the directory containing dataset (audio tracks in the form of **.wav**) required for training purpose.
 -  ***raw_test_data_directory***: The path to the directory containing dataset (audio tracks in the form of **.wav**) required for testing purpose.
 -  ***raw_validation_data_directory***: The path to the directory containing dataset (audio tracks in the form of **.wav**) required for validation purpose.
 -  ***chunk_length_in_milli_sec***: As a preprocessing step, we assumed that all audio tracks should be in the same length. Using this feature, you can configure the desired length in milliseconds.

-  ***Note***: This length should be at least the same size as the longest audio file in your dataset.
- ***Note***: The model generator should corresponds to the model type. 
- ***Note***: Make sure you are passing the correct features to the model. for More information please read [feature_extraction](/docs/feature_Extraction.md) details.

## A sample configuration file
Here you can find a sample of *training_parameter.yml* file.

    
    model_type: "Sequential"
    parameters:
	    model_generator: "sequential_three_conv_modules_generator"
	    n_conv_filters: [32, 128, 256]
	    filters_shape: [5, 5, 5]	    
	    input_shape: [50, 1]	    
	    n_emotions: 5	    
	    n_mfcc: 50	    
	    n_lmfe: 26	    
	    adam_lr: 0.005	    
	    batch_size: 8	    
	    epochs: 100
	    trained_models_dir: "./prod_models/candidate"	    
	    trained_model_name: "seq_3conv_modules_5emotions"	    
	    feature_train_data_directory: "./data/features/Sequential_5emotions/train"	    
	    feature_test_data_directory: "./data/features/Sequential_5emotions/test"	    
	    feature_validation_data_directory: "./data/features/Sequential_5emotions/validation"	    
	    raw_train_data_directory: "./data/tone_5_emotions_dataset_v1/train"    
	    raw_test_data_directory: "./data/tone_5_emotions_dataset_v1/test"	    
	    raw_validation_data_directory: "./data/tone_5_emotions_dataset_v1/validation"	    
	    chunk_length_in_milli_sec: 7000
  

## Usage

As mentioned beforehand, you need to set parameters in the *training_paramaters.yml* file. Then, you should execute the below command from the root directory (Audio-Sentiment-Analysis).

    python training/model_trainer.py

## Saving Models
The model will be saved as .h5 format in a saved_models directory, inside the directory you specified. Additionally, a YAML file, containing the training hyperparameters will be saved next to the model for further information.


