# Running us52_neuralnetwork.py 

To run the script, you need to run us52_neuralnetwork from the audio_sentiment_analysis directory using the following command:

> linux:`python prototypes/us52_neuralnetwork.py`
> windows: python prototypes\us52_neuralnetwork.py

## Installation of dependencies
However, before running this command please make sure you have installed all the required libraries. The new libraries that you might need to install in compare to the previous ones are listed below:

    pip install python_speech_features
    pip install pydub
  
## Loading the data and models
please make sure that you have already created a directory called features/tone inside your data directory. The extracted features will be saved in this directory. In case you decided to make some changes in your data, please first remove the joblib files created in this directory and rerun the python script [us52_neuralnetwork.py].

Using the joblib files you only need to extract features once, and use them for later training in the future, which could save a lot of time.

## Related Report to this user story
You could find the report related to this user story [here](https://tuenl.sharepoint.com/:w:/r/sites/gad_cbo/JPC/MC/ESA%20PDEng%20ST%20Project/Reports%20(user%20stories)/Audio/Sprint%207/US52_report.docx?d=wf04749618f6a4b0a9372e816c1ea7153&csf=1&web=1&e=kfdfGN)
