## Sentiment prediction

This document describes the audio sentiment prediction module. In the following sections, we present its design, configuration, and usage. 

### Design
The sentiment prediction module uses a tone-based CNN model to detect sentiment (emotions) in audios. The components of this prediction module are shown in the figure below. Overall, these components belong to one of the following types: prediction engine, audio analyzer, and tone-based sentiment predictor. The implementation of the components in the class diagram is available in the ```src``` folder of the source code. 

<img src="./images/predict-class-diagram.png" alt="drawing" width="800"/>


#### Prediction engine
The prediction engine is the component that handles the general audio sentiment analysis process. It interacts with the users to record audio and displays the emotions predict for each recorded audio. The ```SternAudio``` class represents this prediction engine. 

Overall, ```SternAudio``` performs the sentiment analysis process in a loop as follows:
1. record audio from a microphone
2. analyze the recorded audio
3. show the emotions resulting from the analysis
4. pause for a short period before recording the next audio (step 1)

The ```SternAudio``` class uses the ```AudioAnalyzer``` class to perform the audio analysis operation. The behavior of the prediction loop can be altered by modifying the following properties in the configuration file: ```audio_length```, ```audio_frequency```, ```audio_channels```, ```before_recording_pause```, ```after_audio_analysis_pause```, and ```iterations```. In the next section we describe the contents of the configuration file.

#### Audio analyzer
The audio analyzer functionality, which is described by the ```AudioAnalyzer``` class, uses a tone-based sentiment predictor to compute the emotions associated with a given audio. The computed emotions are then preprocessed to generate human-readable messages that will be presented to the users of the sentiment prediction module. In addition, the computed emotions are logged in a file specified by the user.

The behavior of the ```AudioAnalyzer``` can be altered by modifying the following properties in the configuration file: ```audio_frequency```, ```emotions```, ```logging_file_prefix```, and ```logging_directory```. In the next section we describe the contents of the configuration file.

#### Tone-based sentiment predictor
The tone-based sentiment predictor component is responsible for computing features from a recorded audio and analyzing these features to predict its emotions. In the class diagram shown above, this component is represented as the ```ToneSentimentPredictor``` class. As shown in the class diagram, two specialized tone-based sentiment predictors inherit from this class: ```SequentialToneSentimentPredictor``` and ```SiameseToneSentimentPredictor```. 

The ```SequentialToneSentimentPredictor``` class uses sequential CNN models trained with mfcc features. Therefore, in order to perform sentiment prediction using its CNN model, this class computes mfcc features of a given audio. In contrast, the ```SiameseToneSentimentPredictor``` uses siamese CNN models trained using mfcc and lmfe features. Thus, this class computes mfcc and lmfe features in order to predict the sentiment of a given audio.

The ```src``` folder contains only the ```SequentialToneSentimentPredictor``` and ```SiameseToneSentimentPredictor```. The ```ToneSentimentPredictor``` class is part of the design but was not implemented. Nonetheless, if more tone-based predictors classes are required, the ```ToneSentimentPredictor``` could be implemented in order to couple the functionality of new tone-base predictors with the required interface.

The behavior of the implemented tone-based predictor classes can be altered by modifying the following properties of the configuration file: ```prod_models_dir```, ```model_dir```, ```model_name```, ```model_type```, ```audio_frequency```, and ```n_mfcc```. In the next section we describe the contents of the configuration file.

### Configuration
Users of the sentiment prediction module can configure its behavior using one of the configuration files available in the ```src``` directory: ```raspi_candidate_config.yml``` and ```raspi_deployment_config.yml```. These configuration files contain the parameters required by the stern audio, the audio analyzer, and the tone-based sentiment predictors. As both configuration files contain the same parameters, both can be used to execute prediction engine. Nonetheless, we recommend using ```raspi_candidate_config.yml``` during the development process and using ```raspi_deployment_config.yml``` to execute the code in the Raspberry Pi.

These configuration files contain the following parameters.
* ```input_type``` is a value that indicates whether audios are recorded from a microphone (```mic```) or loaded from a directory (```recorded```).
* ```input_directory``` indicates the directory containing recorded audios to be analyzed. This property is used only when the ```input_type``` is set to ```recorded```.
* ```audio_length``` is the length (in seconds) of the audios to be recorded or loaded for analyzis. The length of these audios must be the same as the length of the audios used to train the CNN models that perform sentiment prediction. The length of the training audios is given by the ```chunk_length_in_milli_sec``` parameter inside the ```training/training_parameters.yml``` and ```training/retraining_parameters.yml``` files. 
* ```audio_frequency``` corresponds to the frequency (in hertz) of the audios to be analyzed, which can be recorded from a microphone or loaded from a directory. This frequency must be the same as the frequency used to compute features from training audios. Some audio frequencies might not be supported by the microphones connected to the Raspberry Pi. For instance, the Raspberry Pi cannot record audio at 22.5Khz. 
* ```audio_channels``` is the number of channels used to record audios. For instance, the value 1 indicates a mono audio and the value 2 indicates a stereo audio. This property is used only when the ```input_type``` is set to ```mic```.
* ```before_recording_pause``` corresponds to number of seconds to pause after showing a message to the user indicating that the recording process is about to start. This pause helps the user to prepare for speaking to the microphone. This property is used only when the ```input_type``` is set to ```mic```.
* ```after_audio_analysis_pause``` is the number of seconds to pause after an audio record was analyzed. This property is used only when the ```input_type``` is set to ```mic```.
* ```iterations``` corresponds to the number of audios to be recorded from the microphone and analyzed. To indicate an infinite number of audios, set this value to ```-1```. This property is used only when the ```input_type``` is set to ```mic```.
* ```logging_directory``` is the directory that stores the log files produced by sentiment prediction module. In case this directory does not exists, it will be created. This the path of this directory must be specified relative to the root folder of the source code repository. For instance, the following value ```logging_directory = ./logs``` will create a ```logs``` directory at the same level as the ```src``` and ```training``` directories.
* ```logging_subdirectory_prefix``` corresponds to the prefix of the subdirectory that contains information about the predictions of the sentiment prediction module. Each time the ```stern_audio.py``` script is executed, a new logging subdirectory (with prefix given by ```logging_subdirectory_prefix```) is created inside the ```logging_directory```.
*  ```emotions``` is a dictionary that maps numerical values representing emotions to names of these emotions. This dictionary is used to show the predicted emotions in a human readable format. The values and names of the emotions in this dictionary must match with the labels used to train the tone-based detection model.
* ```prod_models_dir``` is the path of directory that contains candidate models for sentiment prediction. This path must also be specified relative to the root folder of the source code repository.
* ```model``` corresponds to a dictionary containing information about the tone-based model to be used for sentiment prediction.
    * ```model_dir``` corresponds to the path of the directory that contains the tone-based model to be used by the sentiment prediction module. This path must be indicated relative to the directory indicated in the ```prod_models_dir``` parameter.
    * ```model_name``` corresponds to the name of the ```.h5``` model inside ```model_dir```.
    * ```model_type``` indicates the type of tone-based detection model. Currently, the sentiment prediction module supports two types: ```sequential``` and ```siamese```. The ```sequential``` type is related to the class ```SequentialToneSentimentPredictor``` and the ```siamese``` type to the class ```SiameseToneSentimentPredictor```.
    * ```n_mfcc``` corresponds to the number of mfcc features to be computed by the ```ToneSentimentPredictor``` classes. This value affects the shape of the input feature arrays used for sentiment prediction. Therefore, it must match the ```n_mfcc``` value inside the ```training/training_parameters.yml``` file.
* ```test_data_dir``` corresponds to the directory containing test data. This value is used by the testing scripts inside the ```tests``` directory.


Below, you can see an example of a configuration file (either ```raspi_candidate_config.yml``` or ```raspi_deployment_config.yml```).
model:
```
input_type: "mic" 
input_directory: "./prod_data/prediction" # used only with input_type: "recorded"

# Parameters of models and data
test_data_dir: "./prod_data/test"
prod_models_dir: "./prod_models/candidate"

model:
    dir: "seq_3conv_modules_2layer_Raha_Siamese_44100/saved_models"
    file: "Emotion_Voice_Detection_Model.h5"
    type: "Siamese"
    n_mfcc: 50

# Audio recording properties
audio_channels: 1
audio_frequency: 44100 
audio_length: 7 

# User interaction properties
iterations: 2 # for an infinite number of iterations, use -1
after_audio_analysis_pause: 1 # length of pause in seconds
before_recording_pause: 1

# List of emotions used for prediction
emotions:
    0: "neutral"
    1: "happy"
    2: "sad"
    3: "angry"
    4: "fearful"

# Logging properties
logging_directory: './logs'
logging_file_prefix: 'test_logging_file'
```

### Usage
**TODO**
