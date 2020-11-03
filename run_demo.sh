#!/bin/bash

script=src/demo.py
audio_model="models/tone_cnn_8_emotions/saved_models/Emotion_Voice_Detection_Model.h5"
text_model=

python $script $audio_model $text_model