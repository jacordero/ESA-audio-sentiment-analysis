import os
import numpy as np

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds

import matplotlib.pyplot as plt 

def prepare_datasets():
    train_data, test_data = tfds.load(name="imdb_reviews", split=["train", "test"], 
                                  batch_size=-1, as_supervised=True)

    train_examples, train_labels = tfds.as_numpy(train_data)
    test_examples, test_labels = tfds.as_numpy(test_data)

    print("Training entries: {}, test entries: {}".format(len(train_examples), len(test_examples)))
    print("=====================================================================")
    print(train_examples[:1])
    print("=====================================================================")
    print("Labels:", train_labels[:1])

    model = create_model()
    
    x_val = train_examples[:10000]
    partial_x_train = train_examples[10000:]

    y_val = train_labels[:10000]
    partial_y_train = train_labels[10000:]

    # To train the model
    history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=40,
                    batch_size=512,
                    validation_data=(x_val, y_val),
                    verbose=1)

    results = model.evaluate(test_data, test_labels)

    print(results)
    # To save the model
    save_model(model)
    
def save_model(model):
    model_name = 'sentiment_analysis_anger_happy_text_model'
    save_dir = os.path.join(os.getcwd(), 'trained models')
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model_path = os.path.join(save_dir, model_name)
    model.save(model_path)
    print("The trained model is saved at {}.".format(model_path))

def create_model():
    # To create a Keras layer - Token based text embedding trained on English Google News 130GB corpus
    model = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1"
    hub_layer = hub.KerasLayer(model, output_shape=[20], input_shape=[], 
                           dtype=tf.string, trainable=True)

    # To build the model
    model = tf.keras.Sequential()
    model.add(hub_layer)
    model.add(tf.keras.layers.Dense(16, activation='relu'))
    model.add(tf.keras.layers.Dense(2, activation='softmax'))

    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

    model.summary()

    return model

def main():
    print("==== WELCOME to the TEXT SENTIMENT ====")
    prepare_datasets()

if __name__ == "__main__":
    main()

