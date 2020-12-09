# STERN Audio module

STERN Audio module repository contains the software, testing pipeline,and libraries such as traing and retraining the models. 

- A [preprocessing](#preprossing ) that extract feautures from audial input data. 
- A [training](#training) library provides the functions to train the models on the extract features. 
- A [retraining](#retraining) library provides the functions to retrain the models on the new extracted features or with hyperparameters (.yml) files. 
- The [emotion recognition](#emotion-recognition) software that detects emotions from audial input data. 
- A [testing pipeline](#testing-pipeline) pipeline that assess the models. 

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

The code in this repository requires several libraries, which are listed in the [requirements.txt](requirements.txt) document. In particular, the [use cases](#use-cases) in this document asume that you have installed `Python3.7+` and `TensorFlow2.2.0`.

Before installing the required libraries, we recomend to create a dedicated virtual environment. In Python 3, virtual environments are created as follows: 
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
First install the [appropriate version of TensorFlow](https://qengineering.eu/install-tensorflow-2.2.0-on-raspberry-pi-4.html)

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

Notably, features can be extracted once and be reused a hundred times for training or retraining purposes. Therefore, as feature extraction could be time-consuming depending on the feature to be extracted and data size, saving the required features can help. For more details on our feature extraction procedure, required libraries, and its usage, please check [here](./docs/FeatureExtraction.md)

### Training 

In this stage, the model is trained with the training data to generate the right output. In the learning phase, the system uses a configuration file to manage the inputs. This configuration file consists of different editable factors in the model training, such as the model type, batch size, number of epochs, train, test, and validation data directory. After training the model, the result will be saved for further use. For more information about training the model, please check [here](./docs/Training.md).

### Retraining
After training the model, it is possible to retrain it with a different dataset and improve accuracy. In order to retrain the model, some parameters should be set in the related configuration file. This information includes the model type, batch size, number of epochs, train, test, and validation data. It is good to mention that the dataset should be in the “.joblib” format. To find more about the retraining procedure, please visit [here](./docs/Retraining.md).
### Emotion recognition

#### Description
This example shows how to perform audio emotion recognition using the ```stern_audio.py``` script. A detailed description about how this script works is available in the [emotion recognition](./docs/emotion_recognition.md) page.

This ```stern_audio.py``` script requires a configuration file to set up the properties and load the tone-based model required for emotion recognition. The ```src``` directory contains two similar configuration files: ```src/raspi_candidate_config.yml``` for development tasks and ```src/raspi_deployment_config.yml``` for deployment on the Raspberry Pi. The content of these files is described in the [emotion recognition](./docs/emotion_recognition.md) page.

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

To verify the correctness of the different software modules within the STERN audio module, this repository contains testing code. All automated test cases can be executed with the following command:

```
pytest tests/test_cases/ --disable-warnings
```

More information about the testing code can be found in the [testing documentation](/docs/Testing.md).

