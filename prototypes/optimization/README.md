:one: To run the `convert_tf_to_lite.py ` script:

The script contains four functions 

` load_model():`   loads the models from ` prod_models/candidate/sequential ` and ` prod_models/candidate/siamese ` by default. 

Path and type of model can be changed in `main` function
``` 
parameters = {
            # resource related parameters
            'model_name': 'candidate/sequential/saved_models/Emotion_Voice_Detection_Model.h5', 
            'tflite_model_name': 'sequential'
}   
```

` convert_saved_models(): ` converts tensorflow model (.pb) into a TensorFlow Lite model.

` quantization_float32(): ` converts keras model (.h5) into a TensorFlow Lite model (post-training float32).

` quantization_int8(): ` converts keras model (.h5) into a TensorFlow Lite model (post-training int8).

` siamese_representative_data_gen(): ` provides the generator to represent typical large enough input values and allow the converter to estimate a dynamic range for all the variable data. (The dataset does not need to be unique compared to the training or evaluation dataset.)

` sequential_representative_data_gen(): ` provides the generator to represent typical large enough input values and allow the converter to estimate a dynamic range for all the variable data. (The dataset does not need to be unique compared to the training or evaluation dataset.)

**Sequential** model :arrow_right: loads the mfcc_test.joblib from the `prod_data/test/sequential/test`. 

**Siamese** models :arrow_right: loads lmfe_test.joblib and mfcc_test.joblib from the ` prod_data/test/siamese/test`. 

` get_output_folder_path(parameters): ` creates a director to save TFlite models.

` save_model(): ` saves the models in the  ` prod_models/tensorflow_lite`

**Sequential** model :arrow_right: `sequential_float32_v[version].tflite` and `sequential_int8_v[version].tflite`

**Siamese** model :arrow_right:  `siamese_float32_v[version].tflite` and `siamese_int8_v[version].tflite`


To run the script by the following command:

` python prototypes/convert_tf_to_lite.py [model type] [tflite type] [version] `

Where, **[model type]** indicates the A: sequential and B: siamese model                          
       **[tflite type]** indicates the 1: float32 and 2: int8. 

Example: `python prototype/convert_tf_to_lite.py A 1 [version] ` 

:two: To run (` execute_lite_model.py `) script:

The script contains three functions.

` load_test_data(): ` loads the test dataset ((to total number of record is 624)) for sequential and siamese models from `prod_data/test/sequential` .

` execute_lite_model(): ` loads the TFLite model from ` prod_models/tensorflow_lite` and  run the TFlite model on test dataset and make prediction.

` evaluate_model(): ` evalutes the model accuracy. 

To run the script by the following command:
` python prototypes/execute_lite_model.py [model type] [optimization option] `

```
'A1': 'sequential_float32_v1.tflite', 
'A2': 'sequential_int8_v1.tflite',
'B1': 'siamese_float32_v1.tflite', 
'B2': 'siamese_int8_v1.tflite'
```
Example:  To run the sequential model with FLOAT 32 optimization.

`python prototype/execute_lite_model.py A 1 ` 

 :loudspeaker: **RESULT** 
 
The post-training **FLOAT32** and **INT8** quantization applied on the Sequential and Siamese models. The following are the results:


**SIZE**:

| Model Name | Siamese | Sequential |
| ------ | ------ | ------ |
| Original | 258,998 KB| 2,360 KB|
| FLOAT32 |  64,751 KB| 201 KB |
| INT8    |  64,797 KB|  214 KB |

**ACCURACY**:

| Model Name | Siamese | Sequential |
| ------ | ------ | ------ |
| Original | 68 % | 59 %  |
| FLOAT32 |  59.8756 %| 58.6314 % |
| INT8    |  18.9736 %|  48.056 %|

**TIME TO PREDICT**:
The performance time includes the loading models and loading test dataset time. 

1. Sequential FLOAT32 TFlite model:

![A1](/uploads/3cdbed0a2db006390dec632a7e865202/A1.PNG)

2. Sequential INT8 TFlite model:

![A2](/uploads/f865c01288d26fc22be6c8d01d81fa8f/A2.PNG)

3. Siamese FLOAT32 TFlite model:

![B1](/uploads/4beec86241f262c8159af661f459eee6/B1.PNG)

4. Siamese INT8 TFlite model:

![B2](/uploads/d4dfee2eb1640bb92ebfedc93a2b83ff/B2.PNG)