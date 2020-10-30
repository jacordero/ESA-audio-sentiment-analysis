import os
import csv

import pandas as pd
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

# Divide datasets into sample and labels
# Shuffle the sample and labels
def prepare_data(input_path):
    samples = []
    labels = []
    a = Emotion()
    with open(input_path, mode='r',  encoding='latin1') as reader:
        reader = csv.reader(reader)
        # to read the header
        header = next(reader)   
        for rows in reader:
            samples.append([rows[1]])
            labels.append([a.emotion_encode(rows[2])])
    # to check labels and count the samples
    training_data = np.array(samples)
    training_labels = np.array(labels)
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
    print("TRAIN ==: ", len(train_samples))
    print("LABELS ==:", len(train_labels))

    print("================================================================")
    print("TEST ==:", len(test_samples))
    print("LABEL ==:", len(test_labels))
    
    # Model constants.
    max_features = 20000
    embedding_dim = 128
    sequence_length = 50
    vectorizer = TextVectorization(max_tokens=max_features, output_mode="int", output_sequence_length=sequence_length)
    
    vectorizer.adapt(train_samples)
    vectorizer.adapt(test_samples)

    train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=7)
    test_labels = tf.keras.utils.to_categorical(test_labels, num_classes=7)
    # vectorize the text to numbers.
    integer_version_training_data = vectorizer(train_samples)
    integer_version_testing_data = vectorizer(test_samples)

    model = build_model(max_features, embedding_dim, sequence_length )

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    model.summary()

    # to train the model 
    model.fit(integer_version_training_data, train_labels, batch_size=32, epochs=10)

    # to evaluate the model
    result = model.evaluate(integer_version_testing_data, test_labels)

    print(result)

    save_model(model)


def build_model(max_features, embedding_dim, sequence_length):
    # BUILD MODEL - including text vectorization layer
    inputs = tf.keras.Input(shape=(None,), dtype="int64")
    x = layers.Embedding(max_features, embedding_dim)(inputs)
    x = layers.Dropout(0.5)(x)

    # Conv1D + global max pooling
    x = layers.Conv1D(128, 7, padding="valid", activation="relu")(x)
    x = layers.Conv1D(128, 7, padding="valid", activation="relu")(x)
    x = layers.GlobalMaxPooling1D()(x)

    # We add a vanilla hidden layer:
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)

    predictions = layers.Dense(7, activation="softmax", name="predictions")(x)

    model = tf.keras.Model(inputs, predictions)

    return model
   
def save_model(model):
    model_name = "emotion_text_model"
    save_dir = os.path.join(os.getcwd(), 'models')
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    model_path = os.path.join(save_dir, model_name)
    model.save(model_path)
    print("The trained model is saved at {}.".format(model_path))


def explore_datasets(input_path):
    # load data from the file
    data = pd.read_csv(input_path)
    # show the first five record 
    print(data.head())

def main():
    print("==== WELCOME to the EXPLORE DATASETS ====")
    input_data_path = os.path.join(os.getcwd(), 'output_datasets/esa_datasets.csv')
    print(input_data_path)
    prepare_data(input_data_path)
    #explore_datasets(input_data_path)



if __name__ == "__main__":
    main()