import os
import tensorflow as tf
import numpy as np

from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

class Decode_emotion(object):
    def emotion_decode(self, argument):
        method_name = 'decode_'+ str(argument)
        method = getattr(self, method_name, lambda: "INVALED CODE")
        return method()

    def decode_0(self):
        return 'neutral'

    def decode_1(self):
        return 'joy'
    
    def decode_2(self):
        return 'sadness'

    def decode_3(self):
        return 'anger'
    
    def decode_4(self):
        return 'fear'
    
    def decode_5(self):
        return 'disgust'
    
    def decode_6(self):
        return 'surprise'
    
def predict_emotion():
    model = load_model()
    decoder = Decode_emotion()

    #model.summary()
    # text to predict emotion
    text = load_text()
    # Model constants.
    max_features = 20000
    embedding_dim = 128
    sequence_length = 50

    vectorizer = TextVectorization(max_tokens=max_features, output_mode="int", output_sequence_length=sequence_length)
    text_vector = vectorizer(text)

    prediction = model.predict(text_vector, batch_size=32)
    emotion_code = np.argmax(prediction)
    print("ALL 7 EMOTION", prediction)
    print("INPUT TEXT == ", text)
    print("PREDICTED EMOTION ==>", decoder.emotion_decode(emotion_code))

def load_model():
    # to load the model
    # model is located at model directory
    model_name = "emotion_text_model"
    load_dir = os.path.dirname(os.path.normpath(os.getcwd()))
    load_dir = os.path.join(load_dir, 'models')
    if not os.path.isdir(load_dir):
        print("No directory.")
    else: 
        model_path = os.path.join(load_dir, model_name)
        print("Model Path: ===", model_path)
        if not os.path.isdir(model_path):
            print("No model")
        else:
            model = tf.keras.models.load_model(model_path)
            return model

def load_text():
    text_file_name = "empty.txt"
    # to get parent directory of current directory
    load_dir = os.path.dirname(os.path.normpath(os.getcwd()))
    load_dir = os.path.join(load_dir, 'data')
    if not os.path.isdir(load_dir):
        print("No input directory.")
    else:
        text_file_path = os.path.join(load_dir, text_file_name)
        if not os.path.isfile(text_file_path):
            print("No input")
        else:
            samples = []
            with open(text_file_path, 'r', encoding='latin1') as reader:
                #data = reader.read().replace('\n', '')
                for line in reader:
                    samples.append([line])
            input_text = np.array(samples, dtype='object')
            return input_text


def main():
    print("==== WELCOME to the TEXT EMOTION prediction ====")
    predict_emotion()

if __name__ == "__main__":
    main()