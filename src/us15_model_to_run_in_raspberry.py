from keras.models import model_from_json
from sklearn.preprocessing import LabelEncoder
import keras
import librosa
import librosa.display
#%pylab inline
import os
import pandas as pd
import librosa
import glob 

#load model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
#load weights into new model
loaded_model.load_weights("models/tone_cnn_happy_angry/saved_models/Emotion_Voice_Detection_Model.h5")
print("Loaded model from disk")

X, sample_rate = librosa.load('data/tone_cnn_happy_angry_dataset_sample_prediction_tracks/output10.wav', res_type='kaiser_fast',duration=2.5,sr=22050*2,offset=0.5)
sample_rate = np.array(sample_rate)
mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13),axis=0)
featurelive = mfccs
livedf2 = featurelive

livedf2= pd.DataFrame(data=livedf2)

livedf2 = livedf2.stack().to_frame().T

livedf2

twodim= np.expand_dims(livedf2, axis=2)

livepreds = loaded_model.predict(twodim, 
                         batch_size=32, 
                         verbose=1)

livepreds

livepreds1=livepreds.argmax(axis=1)

liveabc = livepreds1.astype(int).flatten()

#load lable encoding

lb = LabelEncoder()
lb.classes_ = numpy.load('models/tone_cnn_happy_angry/encoder.npy',allow_pickle=True)

livepredictions = (lb.inverse_transform((liveabc)))
livepredictions
