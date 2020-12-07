# STERN Audio module

STERN Audio module repository contains the software, testing pipeline,and libraries such as traing and retraining the models. 

- A preprocessing that extract [feautures](./docs/FeatureExtraction.md) from audial input data. 
- A [training](./docs/Training.md) library provides the functions to train the models on the extract features. 
- A [retraining](./docs/Retraining.md) library provides the functions to retrain the models on the new extracted features or with hyperparameters (.yml) files. 
- The software implementation of the [emotion recognition](./docs/emotion_recognition.md) module.
- A [testing pipeline](./docs/Testing.md) pipeline that assess the models. 

## Repository structure

- code_verification: contains the configuration file for conditional testing pipeline.
- data: contains images.
- docs: contains the readme.md files for documentation.
- model_verification: contains the configition file of Artificial Intelligence (AI) component for conditional pipeline.
- prototypes: contains the scrips for development purpose.
- src : contains the STERN software of audio module and including configuration files. 
- test: contains test scripts with configuration file.
- training: contains the libraries for training and retraining the models.
- requirements.txt: contains all the dependencies which need to be installed in order to run the STERN software.

## Requirements

` pip ` is already installed with `Python` downloaded from [python.org](https://www.python.org/). 

The software requires ` Python3.7+` and `TensorFlow2.2.0`. 

All the dependencies needed to run the software are listed in [requirements.txt](requirements.txt). 

The following command installs the packages according to the requirements file. 

` pip install -r requirements.txt `

The following command installs [TensorFlow](https://qengineering.eu/install-tensorflow-2.2.0-on-raspberry-pi-4.html) on the Raspberry Pi:

```
sudo apt-get install gfortran libhdf5-dev libc-ares-dev libeigen3-dev libatlas-base-dev libopenblas-dev libblas-dev liblapack-dev cython
pip3 install gdown
gdown https://drive.google.com/uc?id=11mujzVaFqa7R1_lB7q0kVPW22Ol51MPg
pip3 install tensorflow-2.2.0-cp37-cp37m-linux_armv7l.whl

```

The following command installs all dependencies on the Raspberry Pi:

```
# (https://stackoverflow.com/a/48244595)
sudo apt-get install llvm-9
LLVM_CONFIG=/usr/bin/llvm-config-9 pip3 install -r requirements.txt
```
**NOTE:** On the Raspberry Pi, ` python2` and ` python3` both are installed. 

In order to prevent cluttering the Raspberry Pi's system libraries, a `Python Virtual Environment` is recommended. The following command is used to setup the environment. 

```
pip install virtualenv

virtualenv [name]

```
To activate the python virtual environment by running the following command:

`source [name]/bin/activate`

To deactivate the python virtual environment by running the following command:

`deactivate`

In oder to setup the audio sentiment analysis modelule, [please check](README.md)

## Tutorials (Use cases)

### Preprossing 
Audio preprocessing aims to make the audio files ready for being used in the training phase. There are various audio preprocessing techniques, including noise reduction, padding, and windowing. Different techniques can be used to extract audio features. According to the most recent researches, Mel-Frequency Cepstral Coefficients (MFCC) and Log Mel-Filter bank Energies (LMFE) are the most important audio features. Extracting the mentioned features requires techniques like windowing or making all the audio tracks the same length.

Notably, features can be extracted once and be reused a hundred times for training or retraining purposes. Therefore, as feature extraction could be time-consuming depending on the feature to be extracted and data size, saving the required features can help. For more details on our feature extraction procedure, required libraries, and its usage, please check [here](./docs/FeatureExtraction.md)

### Training 

[Training](./docs/Training.md)

### Retraining
[Retraining](./docs/Retraining.md)

### Emotion recognition
This example shows how to perform audio emotion recognition using the ```stern_audio.py``` script. A detailed description about the way this script works is available in the [emotion recognition](./docs/emotion_recognition.md) page.


This ```stern_audio.py``` script requires a configuration file to set up the properties and load the tone-based model required for emotion recognition. The ```src``` directory contains two configuration files that can be used for this purpose: ```src\raspi_candidate_config.yml```or ```src\raspi_deployment_config.yml```. 

Even thought both files contains the same information, during development tasks we recommend to use the candidate configuration file. In constrast, the deployment configuration file should be used to execute the emotion recognition software in the Raspberry Pi. The content of both configuration files is shown below.

```
input_type: "mic" 
input_directory: "./prod_data/prediction" # used only with input_type "recorded"

model:
 dir: "seq_3conv_modules_5emotions_ret/saved_models"
 file: "Emotion_Voice_Detection_Model.h5"
 type: "Sequential"
 n_mfcc: 50

test_data_dir: "./prod_data/test"
prod_models_dir: "./prod_models/candidate"

#audio recording properties
audio_channels: 1
audio_frequency: 44100
audio_length: 7

#list of emotions being used in the prediction
#These codes should comply with the feature extraction part in training code
emotions:
  0: "neutral"
  1: "happy"
  2: "sad"
  3: "angry"
  4: "fearful"

# interaction properties
iterations: 2 # for an infinite number of iterations, use -1
after_audio_analysis_pause: 1 # length of pause in seconds
before_recording_pause: 1

# logging
logging_directory: './logs'
logging_file_prefix: 'test_logging_file'
```

This configuration indicates that:
* The audio emotion recognition software uses the microphone to capture (record) input audio.
* The ```Emotion_Voice_Detection_model.h5``` model inside the ```seq_3conv_modules_5emotions_ret/saved_models``` directory will be used for emotion recognition.
* A frequency of 44100hz will be used to record audios.
* The length of each recorded audio will be 7 seconds.
* Only five emotions will be detected.

Once the configuration file is ready, run the emotion recognition software as follows.

* In the development environment:
```
python src\stern_audio.py src\raspi_candidate_config.yml
```

* In the Raspberry Pi:
```
python src\stern_audio.py src\raspi_deployment_config.yml
```

** TODO: update this image **

<img src="./docs/images/sentiment-analysis-prediction.PNG" alt="drawing" width="400"/>


### Testing pipeline
[Testing]('./docs/Testing.md')
