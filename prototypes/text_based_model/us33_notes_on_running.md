Notes on running text-based emotion detection model on Raspberry Pi
===================================================================

Dependencies
------------
The text-based emotion detection model has minimal dependencies: it only relies on the installation of Tensorflow 2.2.0 and Numpy. These are already available under the virtual environment located at `~/Documents/esa/venv`.

Running the script
------------------
The script can be run using the following chain of commands:

```bash
### activate environment when not yet done so
# source ~/Documents/esa/venv/bin/activate
cd ~/Documents/esa/experiments/audio-sentiment-analysis/prototypes

python3 us24_predict_emotion.py ~/Documents/esa/experiments/audio-sentiment-analysis/data/empty.txt
```
Expected output (will when another input file is provided; see [closing notes](#Closing-notes)):

```python
==== WELCOME TO THE TEXT EMOTION PREDICTION ====
MODEL PATH == /home/pi/Documents/esa/experiments/audio-sentiment-analysis/models/emotion_text_model
SCORE FOR ALL 8 EMOTIONS [[4.6695732e-07 6.4992980e-09 1.5724783e-10 3.4699005e-05 1.3001249e-03
  9.9115330e-01 2.0690798e-03 5.4423278e-03]]
INPUT TEXT ==  [["You stole them from me! Hello World! I don't like you "]]
PREDICTED EMOTION == DISGUST
```

Closing notes
-------------

Some changes had to be made to the script in order to run it on the Raspberry Pi. The biggest change that was made was that it is now possible and required to provide the path to an input file as the first argument of the script. The complete text in this input file will be used as the input for the emotion detection model.