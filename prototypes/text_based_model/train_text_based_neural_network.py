import os
import csv
import sys

import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras.layers import Embedding
from tensorflow.keras import layers

class Emotion(object):
    def emotion_encode(self, argument):
        method_name = 'encode_' + argument
        method = getattr(self, method_name, lambda: "INVALID EMOTION")
        return method()
    
    def encode_neutral(self):
        return 0
    def encode_joy(self):
        return 1
    def encode_sadness(self):
        return 2
    def encode_anger(self):
        return 3
    def encode_fear(self):
        return 4
    def encode_disgust(self):
        return 5
    def encode_surprise(self):
        return 6
    def encode_calm(self):
        return 7

def prepare_data(input_path):
    samples = []
    labels = []
    a = Emotion()
    with open(input_path, mode='r',  encoding='latin1') as reader:
        reader = csv.reader(reader)
        # to read the header
        header = next(reader)   
        for rows in reader:
            # ignore the empty line in datasets.csv file
            if '' not in rows:
                samples.append([rows[1]])
                labels.append([a.emotion_encode(rows[2])])
    # to check labels and count the samples
    training_data = np.array(samples)
    training_labels = np.array(labels)
    # 8 emotions
    emotions = len(np.unique(training_labels))
    print("TOTAL DATAS", len(training_data))
    print("TOTAL LABELS", len(training_labels))

    # shuffle the data
    seed= 1337
    rng = np.random.RandomState(seed)
    rng.shuffle(training_data)
    rng = np.random.RandomState(seed)
    rng.shuffle(training_labels)

    # extract a training and validation split
    # train data -   80
    # test data -    20
    validation_split= 0.2 
    num_test_samples = int(validation_split * len(samples))
    # sample data
    train_samples = training_data[:-num_test_samples]
    test_samples = training_data[-num_test_samples:]
    # labels
    train_labels = training_labels[:-num_test_samples]
    test_labels = training_labels[-num_test_samples:]

    print("================================================================")
    print("TRAIN DATA ==: ", len(train_samples))
    print("LABELS ==:", len(train_labels))

    print("================================================================")
    print("TEST DATA ==:", len(test_samples))
    print("LABELS ==:", len(test_labels))
 

    train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=emotions)
    test_labels = tf.keras.utils.to_categorical(test_labels, num_classes=emotions)

    train_model(train_samples, train_labels, test_samples, test_labels, emotions)


    
def train_model(train_samples, train_labels, test_samples, test_labels, emotions):

    max_features = 20000
    embedding_dim = 128
    sequence_length = 20
    vectorizer = TextVectorization(max_tokens=max_features, output_mode="int", output_sequence_length=sequence_length)
    
    vectorizer.adapt(train_samples)
    vectorizer.adapt(test_samples)

    model = build_model(vectorizer, emotions)

    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    #model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.compile(loss='categorical_crossentropy', optimizer= opt, metrics=["accuracy"])

    # to train the model 
    model.fit(train_samples, train_labels, batch_size=10, epochs=10, validation_data=(test_samples, test_labels))

    model.summary()
    save_model(model)

def build_model(vectorizer, emotions):

    model = tf.keras.Sequential([
    tf.keras.Input(shape=(1,), dtype=tf.string),
    vectorizer,
    tf.keras.layers.Embedding(
        input_dim=len(vectorizer.get_vocabulary())+2,
        output_dim=64,
        # Use masking to handle the variable sequence lengths
        mask_zero=True),
    tf.keras.layers.BatchNormalization(),
    #tf.keras.layers.Conv1D(128, emotions, padding="valid", activation="relu"),
    #tf.keras.layers.BatchNormalization(),
    #tf.keras.layers.Conv1D(128, emotions, padding="valid", activation="relu"),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),
    
    #tf.keras.layers.GlobalMaxPooling1D(),
    #tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(emotions, activation="softmax", name="predictions")
    ])

    return model


def save_model(model):
    model_name = "emotion_text_model"
    save_dir = os.path.dirname(os.path.normpath(os.getcwd()))
    save_dir = os.path.join(save_dir, 'prod_models')
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    model_path = os.path.join(save_dir, model_name)
    model.save(model_path)
    print("The trained model is saved at {}.".format(model_path))

def main():
    print("==== WELCOME to the EXPLORE DATASETS ====")
    datasets_file_name = str(sys.argv[1])
    print(datasets_file_name)
    prepare_data(datasets_file_name)


if __name__ == "__main__":
    main()

    