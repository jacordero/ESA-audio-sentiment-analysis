# How to run the text based emotion model:

## Installation

##### Dependencies
* Python (>=3.7)
* Tensorflow (>=2.2.0)
* Numpy

## To run the script
1. Download the esa_datasets.csv, which is located [here](https://tuenl.sharepoint.com/sites/gad_cbo/JPC/MC/ESA%20PDEng%20ST%20Project/ModelsAndData/Audio/Development/data/esa_datasets.csv). 

2. To train the model on the above datasets. 

    step 1: Download the model from [here](https://tuenl.sharepoint.com/sites/gad_cbo/JPC/MC/ESA%20PDEng%20ST%20Project/ModelsAndData/Audio/Development/models/emotion_text_model) and place the model in **models** directory of audio-sentiment-analysis 

    step 2: Run the following script
    
>   python us24_train_model.py path_datasets

3. The following script predicts an input text, which is saved in [file](https://ooti-projects.win.tue.nl/gitlab/st-c2019/esa/audio-sentiment-analysis/-/blob/us24_text_based_model/data/empty.txt)

> python us24_predict_emotion.py