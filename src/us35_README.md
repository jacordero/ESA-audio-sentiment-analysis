The aim of user story 35 is to convert audio to text.

To review this user story, do as follows:

First, pull this branch into your local directory.

Then, if required, activate your virtual environment.

For vosk-api you need Python version: 3.5-3.8 (Linux), 3.6-3.7 (ARM), 3.8 (OSX), 3.8-64bit (Windows)

Next, install the following libraries.
```
pip install vosk pyaudio
```
There are two scripts in the src folder namely `us35_audio_2_txt.py` and `us35_test_microphone.py`.
Before running the scripts you need **vosk-model-en-us-daanzu-20200905-lgraph** model. You can download it from [here ](https://tuenl.sharepoint.com/:f:/r/sites/gad_cbo/JPC/MC/ESA%20PDEng%20ST%20Project/ModelsAndData/Audio/Development/models/vosk-model-en-us-daanzu-20200905-lgraph?csf=1&web=1&e=6xZPzp) and extract that in the models directory.

Run the following command in your terminal to get text from the microphone:

```
python us35_test_microphone.py
```

To get text from sample audio file run the following command:

```
python us35_audio_2_txt.py
```

Before running the above command put a sample audio file in the data directory and give the name of the file script for the variable wav_file:

`self.wav_file = "../data/03-02-01-01-01-01-01.wav"`

After running the script you will see the text from the audio file in the terminal.