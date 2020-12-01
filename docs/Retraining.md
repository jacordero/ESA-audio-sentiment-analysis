# Model Retraining

This file contains instructions for retraining the model. one can retrain either the existing Sequential or Siamese (non-sequential) model with setting proper parameters in the retraining_parameters.yml file.
Another prerequisite is the features as joblib files. For more information how to extract features and save them as joblib files, please read [feature_extration help](feature_extraction).

For retraining the model you should specify the type of the model, and also directories containing the features, as well as some other parameters described in more details in further sections.


## Configuration File

In the training directory you can find a YAML file, called ***retraining_parameters.yml*** . It contains all the parameters required for loading the trained  model, accessing features, retraining the model, and saving the retrained model. Below you can learn about each of these paramaters in detail.

 - [ ] ***model_type***: It can be either "Sequential" or "Siamese". It specifies the type of the trained model .
 - [ ] ***adam_lr***: The learning rate of Adam optimizer.
 - [ ] ***batch_size***: The number of bachtes in each epoch.
 - [ ] ***epochs***: The number of epochs in training phase.
 - [ ] ***pretrained_models_dir***: The path to the directory from the root directory contating the pretrained model (.h5).
 - [ ] ***pretrained_model_name***: The name of the .h5 model.
 - [ ]  ***retrained_models_dir***: The path to the directory from the root directory that the new retrained model will be saved.
 - [ ] ***retrained_model_name***: The name of the new retrained model  in .h5 format.
 - [ ] ***feature_train_data_directory***: The path to the directory from the root directory  containing joblib files for  training.
 - [ ] ***feature_test_data_directory***: The path to the directory from the root directory  containing joblib files for testing.
 - [ ] ***feature_validation_data_directory***: The path to the directory from the root directory contaning the validation joblib files.
 
- ***Note***: Make sure you are passing the correct features to the model. for More information please read [feature_extraction](feature_extraction) details.

## A sample configuration file
Here you can find a sample of training_parameter.yml file.

    
    model_type: "Sequential"
	parameters:		
		adam_lr: 0.001
		batch_size: 32
		epochs: 1
		pretrained_models_dir: "./prod_models/candidate/seq_3conv_modules_2layer_Sequential/saved_models"
		pretrained_model_name: "Emotion_Voice_Detection_Model.h5"
		retrained_models_dir: ./prod_models/candidate/
		retrained_model_name: seq_3conv_modules_be_nice_ret_v1
		feature_test_data_directory: ./data/features/Sequential/test
		feature_train_data_directory: ./data/features/Sequential/train
		feature_validation_data_directory: ./data/features/Sequential/validation
	

## Usage

As mentioned beforehand, you need to set parameters in the retraining_paramaters.yml file. Then, you should execute the below command from the root directory (Audio-Sentiment-Analysis).

    python training/model_retrainer.py

## Saving Models
The model will be saved as .h5 format in a saved_models directory, inside the directory you specified. Additionally, a YAML file, containing the retraining hyperparameters will be saved next to the model for further information.


