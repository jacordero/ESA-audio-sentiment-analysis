import os
import tensorflow as tf
import numpy as np
import pandas as pd

def load_model():
    load_dir = os.path.join(os.getcwd(), 'trained models')
    model_name = 'sentiment_analysis_anger_happy_text_model'
    if not os.path.isdir(load_dir):
        print("No directory.")
    else: 
        model_path = os.path.join(load_dir, model_name)
        print("Model Path: ===", model_path)
        if not os.path.isdir(model_path):
            print("No model")
        else:
            model = tf.keras.models.load_model(model_path)
            #model.summary()
            return model
def load_data():
    input_dir = os.path.join(os.getcwd(), 'input')
    input_file = 'text.txt'
    if not os.path.isdir(input_dir):
        print("No input directory.")
    else:
        input_path = os.path.join(input_dir, input_file)
        print("INPUT PATH: ==", input_path)
        if not os.path.isfile(input_path):
            print("No input")
        else:
            with open(input_path, 'r', encoding='latin1') as reader:
                data = reader.read().replace('\n', '')
                input_text = np.array([data], dtype='object')
                print("INPUT TEST==>", input_text)
                return input_text
            

def analyse_sentiment():
    model = load_model()
    model.summary()
    data = load_data()
    pred = model.predict(data, batch_size=512)
    print("PREDICTION ==>", pred )


def main():
    print("==== WELCOME to the TEXT SENTIMENT prediction ====")
    analyse_sentiment()

if __name__ == "__main__":
    main()