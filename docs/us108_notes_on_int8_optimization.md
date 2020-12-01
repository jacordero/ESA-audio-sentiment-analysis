To run the `convert_tf_to_lite.py ` script:

**1.** The script loads the models from ` prod_models/candidate/sequential ` and ` prod_models/candidate/siamese ` by default. 

You change the path attribute of parameters in `main` function
``` 
parameters = {
            # resource related parameters
            'model_name': 'candidate/sequential/saved_models/Emotion_Voice_Detection_Model.h5', 
            'tflite_model_name': 'sequential'
}   
```
**1.1** For post-training int8, the script needs to read the training sample. 
- Sequential model :arrow_right: loads the mfcc_test.joblib from the `prod_data/test/sequential/test`. 
- Siamese models :arrow_right: loads lmfe_test.joblib and mfcc_test.joblib from the ` prod_data/test/siamese/test`. 

**2.** The script complies with the absolute path. Thus, it can be execute from root folder by the following command:

` python prototype/convert_tf_to_lite.py [model type] [tflite type] [version] `

Where, [model type] indicates the A: sequential and B: siamese model and [tflite type] indicates the 1: float32 and 2: int8. 

Example: `python prototype/convert_tf_to_lite.py A 1 [version] `

**3.** The TFLite models are saved in the  ` prod_models/tensorflow_lite`
- sequential -  `sequential_float32_v[version].tflite` and `sequential_int8_v[version].tflite`

- siamese -  `siamese_float32_v[version].tflite` and `siamese_int8_v[version].tflite`

The TFlite models are already uploaded to dvc:
```
git pull 
dvc pull
```