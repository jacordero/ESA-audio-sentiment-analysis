# STERN Audio module

STERN Audio module repository contains the software and, testing pipeline, and libraries such as traing and retraining the models. 

- A preprocessing that extract [feautures](./docs/FeatureExtraction.md) from audial input data. 
- A [training](./docs/Training.md) library provides the API to train the models on the extract features. 
- A [retraining](./docs/Retraining.md) library provides the API to retrain the models on the new extracted features or with hyperparameters (.yml) files. 
- A [sentiment analysis](./docs/SentimentAnalysis.md) software that analyses the sentiment of the audial input. 
- A [test](./Testing.md) pipeline that assess the models. 

## Repository structure

- .dvc
- docs
- src
- test
- training
- prod_data.dvc
- prod_models.dvc
- requirements.txt

## Requirements

` pip ` is already installed with `Python` downloaded from [python.org](https://www.python.org/). 

The software requires ` Python3.7+` and `TensorFlow2.2.0`. 

All the dependencies needed to run the software are listed in [requirements.txt](requirements.txt). The command will install the packages according to the requirements file. 

` pip install -r requirements.txt `

In order to install [TensorFlow](https://qengineering.eu/install-tensorflow-2.2.0-on-raspberry-pi-4.html) on the Raspberry Pi, the following commands can be used:

```
sudo apt-get install gfortran libhdf5-dev libc-ares-dev libeigen3-dev libatlas-base-dev libopenblas-dev libblas-dev liblapack-dev cython
pip3 install gdown
gdown https://drive.google.com/uc?id=11mujzVaFqa7R1_lB7q0kVPW22Ol51MPg
pip3 install tensorflow-2.2.0-cp37-cp37m-linux_armv7l.whl

```

In order to install all dependencies on the Raspberry Pi, the following command can be used:

```
# (https://stackoverflow.com/a/48244595)
sudo apt-get install llvm-9
LLVM_CONFIG=/usr/bin/llvm-config-9 pip3 install -r requirements.txt
```
**NOTE:** On the Raspberry Pi, ` python2` and ` python3` both are installed. 

In order to prevent cluttering the Raspberry Pi's system libraries, a `Python Virtual Environment` is recommended. The following command can be used to setup the environment. 

```
pip install virtualenv

virtualenv [name]

```
To activate the python virtual environment by running the following command:

`source [name]/bin/activate`

To deactivate the python virtual environment by running the following command:

`deactivate`

## Tutorials (Use cases)

### Preprossing 

[Feature extraction](./docs/feature_Extraction.md)

### Training 

[Training](./docs/Training.md)

### Retraining
[Retraining](./docs/Retraining.md)

### Sentiment Analysis
This example shows how to perform tone sentiment analysis using a tone model previously (re)trained, as described in the steps above. A detailed description of this process is described in the [sentiment analysis process document](./docs/SentimentAnalysis.md).

First, update the contents of the configuration files that can be used for tone sentiment analysis: either ```src\raspi_candidate_config.yml```or ```src\raspi_deployment_config.yml```. The contents of ```src\raspi_candidate_config.yml``` are shown below.

```
model:
 dir: "seq_3conv_modules_5emotions_ret/saved_models"
 file: "Emotion_Voice_Detection_Model.h5"
 type: "Sequential"
 n_mfcc: 50

test_data_dir: "./prod_data/test"
prod_models_dir: "./prod_models/candidate"

#audio recording properties
audio_channels: 1
audio_frequency: 22050 
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
* The ```Emotion_Voice_Detection_model.h5``` model inside the ```seq_3conv_modules_5emotions_ret/saved_models``` directory will be used for sentiment analysis.
* A frequency of 22050hz will be used to record audios.
* The length of each recorded audio will be 7 seconds.
* Only five emotions will be detected.

Once the configuration file has been modified, execute the sentiment analysis script using the command below. For this example, we use the ```src\raspi_candidate_configuration.yml``` file.

```python src\stern_audio.py src\raspi_candidate_configuration.yml```

The image below shows an output of the sentiment analysis script.

<img src="./docs/images/sentiment-analysis-prediction.PNG" alt="drawing" width="400"/>


### Test
[Testing pipeline](./docs/Test.md)