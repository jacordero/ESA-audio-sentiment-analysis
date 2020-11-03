import librosa
import numpy as np
import tensorflow as tf

import keras
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Input, Flatten, Dropout, Activation
from keras.layers import Conv1D, MaxPooling1D, AveragePooling1D
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix
import os
from keras import regularizers

import pandas as pd

from pathlib import Path
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder


dataset_directory = '\\data\\tone_cnn_happy_angry_dataset\\'
models_directory = '../models/audio_tone_binary_sentiment_analysis_v0_6'


def prepare_data():
	curr_path = os.getcwd()
	crr_path = Path(curr_path)
	parent_path = crr_path.parent
	path = str(parent_path) +  dataset_directory
	print(path)
	mylist= os.listdir(path)
	
	## setting labels
	feeling_list=[]
	for item in mylist:
		if item[6:-16]=='03':
			feeling_list.append('Happy')
		elif item[6:-16]=='05':
			feeling_list.append('Angry')

	labels = pd.DataFrame(feeling_list)

    ## get audio features using librosa
	df = pd.DataFrame(columns=['feature'])
	bookmark=0
	skip_numbers = ['01', '07', '08']


	for index,y in enumerate(mylist):
		if mylist[index][6:-16] not in skip_numbers and mylist[index][:2]!='su' and mylist[index][:1]!='n' and mylist[index][:1]!='d':
			X, sample_rate = librosa.load(path+y, 
				res_type='kaiser_fast',
				duration=2.5,
				sr=22050*2,
				offset=0.5)

			sample_rate = np.array(sample_rate)
			mfccs = np.mean(librosa.feature.mfcc(y=X, 
                                            sr=sample_rate, 
                                            n_mfcc=13), axis=0)
			feature = mfccs
        	#[float(i) for i in feature]
			df.loc[bookmark] = [feature]
			bookmark=bookmark+1        


	df3 = pd.DataFrame(df['feature'].values.tolist())
	newdf = pd.concat([df3,labels], axis=1)
	rnewdf = newdf.rename(index=str, columns={"0": "label"})
	
	from sklearn.utils import shuffle
	rnewdf = shuffle(newdf)
	rnewdf[:10]
	rnewdf=rnewdf.fillna(0)


	# divide data into test and train 
	newdf1 = np.random.rand(len(rnewdf)) < 0.8
	train = rnewdf[newdf1]
	test = rnewdf[~newdf1]

	trainfeatures = train.iloc[:, :-1]
	trainlabel = train.iloc[:, -1:]
	testfeatures = test.iloc[:, :-1]
	testlabel = test.iloc[:, -1:]


	X_train = np.array(trainfeatures)
	y_train = np.array(trainlabel)
	X_test = np.array(testfeatures)
	y_test = np.array(testlabel)

	return X_train, y_train, X_test, y_test


def create_model():
	model = Sequential()

	model.add(Conv1D(256, 5,padding='same',
                 input_shape=(216,1)))
	model.add(Activation('relu'))
	model.add(Conv1D(128, 5,padding='same'))
	model.add(Activation('relu'))
	model.add(Dropout(0.1))
	model.add(MaxPooling1D(pool_size=(8)))
	model.add(Conv1D(128, 5,padding='same',))
	model.add(Activation('relu'))
	model.add(Conv1D(128, 5,padding='same',))
	model.add(Activation('relu'))
	model.add(Flatten())
	model.add(Dense(2))
	model.add(Activation('softmax'))
	opt = keras.optimizers.Adam(learning_rate=0.0001)	

	model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
	
	return model	



def run():


	X_train, y_train, X_test, y_test = prepare_data()

	lb = LabelEncoder()
	encoder = LabelEncoder()

	y_train = np_utils.to_categorical(lb.fit_transform(y_train))
	y_test = np_utils.to_categorical(lb.fit_transform(y_test))


	if not os.path.isdir(models_directory):
		os.makedirs(models_directory)
	np.save(os.path.join(models_directory, 'encoder.npy'), lb.classes_)

	## Change dimension of CNN model
	x_traincnn =np.expand_dims(X_train, axis=2)
	x_testcnn= np.expand_dims(X_test, axis=2)	

	model = create_model()
	cnnhistory=model.fit(x_traincnn, y_train, batch_size=16, epochs=200, validation_data=(x_testcnn, y_test))

	model_name = 'weights.h5'
	save_dir = os.path.join(os.getcwd(), models_directory)

	# Save model and weights
	if not os.path.isdir(save_dir):
		os.makedirs(save_dir)
	model_path = os.path.join(save_dir, model_name)
	model.save(model_path)
	print('Saved trained model at %s ' % model_path)


	# Save model description
	description_path = os.path.join(os.path.join(os.getcwd(), models_directory), "model_description.txt")

	with open(description_path,'w') as f:
    	# Pass the file handle in as a lambda function to make it callable
		model.summary(print_fn=lambda x: f.write(x + '\n'))


if __name__ == "__main__":
	run()