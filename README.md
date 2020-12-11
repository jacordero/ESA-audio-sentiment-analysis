# STERN Audio module

STERN Audio module repository contains the STERN software, test cases, and libraries for traing and retraining the models. 

- A [preprocessing](#preprocessing) library provides the functions to extract feautures from audial input data. 
- A [training](#training) library provides the functions to train the models on the extract features. 
- A [retraining](#retraining) library provides the functions to retrain the models on the new extracted features or with hyperparameters (.yml) files. 
- The [emotion recognition](#emotion-recognition) software that detects emotions from audial input data. 
- A [testing](#testing) that verifies the correctness of the STERN audio module.

## Repository structure

- **prod_data:** contains datasets such as raw data, preprocessed data, test data, and description.md for training and retraining the models.
- **prod_models:** contains trained or retrained models such as Sequential and Siamese model. 
- **prototypes:** contains the scrips and their description for development purposes.
- **src:** contains the STERN software of the audio module and includes configuration files. 
- **training:** contains the libraries for training and retraining the models.
- **tests:** contains test scripts with the configuration file.
- **docs:** contains the readme.md files for documentation.
- **README.md:** contains the instruction for setting up the environment and running the STERN software.
- **requirements.txt:** contains all the dependencies, which need to be installed in order to run the STERN software.

## Requirements

The code in this repository requires several libraries, which are listed in the [requirements.txt](requirements.txt) document. In particular, the [use cases](#use-cases) in this document asume that you have installed `Python3.7+` and `TensorFlow2.2.0`.

Before installing the required libraries, we recomend creating a dedicated virtual environment. In Python 3, virtual environments are created as follows: 
```
python -m venv [name] 
```

where `[name]` corresponds to the name of the virtual environment that is created.

After the virtual environment is created, activate it.
* On Windows

```
[name]/Scripts/activate
```

* On Linux

```
source [name]/bin/activate
```

Next, install the required libraries.
### On your PC
Execute the following command.
```
pip install -r requirements.txt
```

### On a Raspberry Pi V4
First install the [TensorFlow2.2.0 version](https://qengineering.eu/install-tensorflow-2.2.0-on-raspberry-pi-4.html)

```
sudo apt-get install gfortran libhdf5-dev libc-ares-dev libeigen3-dev libatlas-base-dev libopenblas-dev libblas-dev 
pip3 install gdown
gdown https://drive.google.com/uc?id=11mujzVaFqa7R1_lB7q0kVPW22Ol51MPg
pip3 install tensorflow-2.2.0-cp37-cp37m-linux_armv7l.whl 
```
Then install the remaining libraries

```
# (https://stackoverflow.com/a/48244595)
sudo apt-get install llvm-9
LLVM_CONFIG=/usr/bin/llvm-config-9
pip3 install -r requirements.txt
```

## Use cases

The following use cases show the main functionality of the source code in this repository.

### Preprocessing
Audio preprocessing aims to make the audio files ready for being used in the training phase. There are various audio preprocessing techniques, including noise reduction, padding, and windowing. Different techniques can be used to extract audio features. According to the most recent researches, Mel-Frequency Cepstral Coefficients (MFCC) and Log Mel-Filter bank Energies (LMFE) are the most important audio features. Extracting the mentioned features requires techniques like windowing or making all the audio tracks the same length.

Notably, features can be extracted once and be reused a hundred times for training or retraining purposes. Therefore, as feature extraction could be time-consuming depending on the feature to be extracted and data size, saving the required features can help. 

The ```feature_extraction.py``` script requires a configuration file to set up the properties, specifically the file path to dataset. Once the setup is configured you could simply run the below command. For detailed information on the configuration file, the required libraries, and  feature extraction procedure, please read the [feature extraction](./docs/FeatureExtraction.md) page.

	python training/feature_extraction.py [desired_feature]

> [desired_feature] can be "mfcc", "lmfe", "mfcc_sequential", or
> "lmfe_sequential"

  ***Note:*** You need to extract both mfcc and lmfe for training the siamese model. This means you need to run the above command twice, one using "mfcc" and one more time with "lmfe" as *'desired_feature'*.
  

### Training 

In this stage, the model is trained with the training data to generate the right output. In the learning phase, the system uses a configuration file named _training_parameters.yml_ to manage the inputs. _training_parameters.yml_ consists of different editable factors in the model training, such as the model type, batch size, number of epochs, train, test, and validation data directory. After training the model, the result will be saved for further use. After setting the parameters in the configuration file, training will be started by running the following command:
       
       python training/model_trainer.py
  
For more information about training the model and configuration file, please check [training](./docs/Training.md) page.

### Retraining
After training the model, it is possible to retrain it with a different dataset and improve accuracy. In order to retrain the model, some parameters should be set in the related configuration file named _retraining_parameters.yml_. This information includes the model type, batch size, number of epochs, train, test, and validation data. To retrain the model, the follwoing command should be executed in command line:

    python training/model_retrainer.py


It is good to mention that the dataset should be in the “.joblib” format.To find more about the retraining procedure and specifying the parameters in configuration file, please visit [retraining](./docs/Retraining.md) page.
### Emotion recognition

#### Description
This example shows how to perform audio emotion recognition using the ```stern_audio.py``` script. A detailed description of how this script works is available on the [emotion recognition](./docs/EmotionRecognition.md) page.

This ```stern_audio.py``` script requires a configuration file to set up the properties and load the tone-based model required for emotion recognition. The ```src``` directory contains two similar configuration files: ```src/raspi_candidate_config.yml``` for development tasks and ```src/raspi_deployment_config.yml``` for deployment on the Raspberry Pi. The content of these files is described on the [emotion recognition](./docs/EmotionRecognition.md) page.

#### Execution during the development process
* First,  modify the `raspi_candidate_config.yml` file.
* Then, run the following command

```
python src/stern_audio.py src/raspi_candidate_config.yml
```

#### Execution on a Raspberry Pi
* First,  modify the `raspi_deployment_config.yml` file.
* Then, run the following command

```
python src/stern_audio.py src/raspi_deployment_config.yml
```

### Testing

This repository contains testing codes to verify the STERN audio module. All automated test cases can be executed with the following command:

```
pytest tests/test_cases/ --disable-warnings
```

More information about the testing code can be found on the [testing](/docs/Testing.md) page.

