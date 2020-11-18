#!/bin/bash

script=src/stern_audio.py
audio_model="prod_models/deployed/tone_cnn_8_emotions_v0_1/saved_models/Emotion_Voice_Detection_Model.h5"
text_model="prod_models/deployed/emotion_text_model_v0_1"
mode="live"
iterations=20
pause_duration=15

python $script $audio_model $text_model $mode $iterations $pause_duration