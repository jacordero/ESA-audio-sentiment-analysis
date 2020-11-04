#!/bin/bash

script=src/sentiment_analyzer.py
model_directory=prod_models/audio_tone_binary_sentiment_analysis_v0_6
data_directory=prod_data/test

python $script $model_directory $data_directory