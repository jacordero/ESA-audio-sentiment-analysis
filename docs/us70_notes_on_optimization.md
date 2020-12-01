:alien: :one:
The script (` convert_tf_to_lite.py` ) contains four functions 

` get_path() ` - find the path of the request directory.

` convert_saved_models() ` - convert the TensorFlow savedModels.

` convert_keras_models() ` -  convert the Keras model.

` save_model() ` - save the TensorFlow Lite model in folder :arrow_right:  ` models/tensorflow_lite`

The Quantization (post training float32) is used for converter  :arrow_right:  [DOCUMENTATION](https://tuenl.sharepoint.com/sites/gad_cbo/JPC/MC/ESA%20PDEng%20ST%20Project/Reports%20(user%20stories)/Audio/Sprint%207/US70_Model_Optimization.docx)

:alien: :two: To execute the script: 

` python convert_tf_to_lite.py [version] [OPTION] `


**OPTION** has two choices: 

:heavy_check_mark: **1** : TensorFlow model (.pb) format 

:heavy_check_mark: **2** : Keras model (.h5) format

**version**  is used to save the TFLite model with [version] in `models/tensorflow_lite` folder. 

Script reads the models from ` prod_models/deployed `

:round_pushpin:  PLEASE run the followin command  inside `prototype` folder. If not, the script will not find the model directory path **ERROR**.

For example:  ` python convert_tf_to_lite.py [0_1] [2]`   

- reads the Tensorflow model `deployed/tone_cnn_8_emotions_v0_1/saved_models/Emotion_Voice_Detection_Model.h5`,
- converts into TFLite model, 
- saves TFLite model in `models/tensorflow_lite` with name `tone_cnn_5_emotions_v0_1.tflite` 

:round_pushpin: You can change the model path inside the script at main function.

```
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
            'lite_model_name': 'tone_cnn_5_emotions',
            'version': sys.argv[1]
        }
        convert_keras_models(parameters)

```

The preparation to execute TFLite model, please do the following.

1. To copy `tone_cnn_5_emotions_v0_1.tflite` TFlite model into `prod_models/tflite_model`
2. To prepare the test dataset and copy it into `prod_data/sequantial_model `. 
 
If the latest version of test dataset with five emotoins is inside `prod_data/test` folder, we can use it instead of step 2. 

3. To add the modification into prod_data and prod_models by   
    ``` 
    dvc add prod_models 
    dvc add prod_data 
    dvc commit 
    dvc push 
    ```
4. To update the gitlab 
    ``` 
    git add prod_models.dvc
    git add prod_data.dvc
    git commit -m ""
    git push
    ```

    If you face complict, please contact configration manager.

:alien: :three:  The script (` execute_lite_model.py `) , with :black_heart: _FIVE_ emotions, contains three functions.

` get_path() ` - find the path of the request directory.

` load_test_data_sequential_model() ` - loads the test dataset (to total number of record is 624) from ` prod_data/sequential_model/test`

` execute_lite_model () ` - load the TFLite model from ` prod_models/tflite_model` and  run the TFlite model on test dataset and make prediction 

In order to run the  ` execute_lite_model.py `, please do following:

1. pull us70_audio_module_optimization branch

2. pull dvc

The above steps fetch the models (` prod_models/tflite_model `) and datasets (` prod_data/ sequential_model`). 

:round_pushpin: PLEASE run the followin command  inside `prototype` folder. If not, the script will not find the model directory path **ERROR**.

3.  ` python execute_lite_model.py ` 

:round_pushpin: You can change the model name inside the script at main function.

```
 parameters = {
        'lite_model_name': 'tone_cnn_5_emotions_v0_1.tflite'
    }

```

The prediction output is illustrated in the following image. 

![Capture](/uploads/2ee6bfecf4e4bc189d792cb1e569da61/Capture.PNG)

**Label** - indicates the original label of the audio and 

**prediction** - indicated the prediction result of TFlite model

 :loudspeaker: **RESULT** 

The trained keras sequantial model size was: **2,362KB**

TFLite model size was: **201KB**