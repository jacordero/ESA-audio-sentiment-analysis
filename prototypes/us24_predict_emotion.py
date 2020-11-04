import os, sys
import tensorflow as tf
import numpy as np

from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

class Decode_emotion(object):
    def emotion_decode(self, argument):
        method_name = 'decode_'+ str(argument)
        method = getattr(self, method_name, lambda: "INVALID CODE")
        return method()

    def decode_0(self):
        return 'NEUTRAL'

    def decode_1(self):
        return 'JOY'
    
    def decode_2(self):
        return 'SADNESS'

    def decode_3(self):
        return 'ANGER'
    
    def decode_4(self):
        return 'FEAR'
    
    def decode_5(self):
        return 'DISGUST'
    
    def decode_6(self):
        return 'SURPRISE'
    
    def decode_7(self):
        return 'CALM'

def predict_emotion(input_file):
    model = load_model()
    decoder = Decode_emotion()

    #model.summary()
    # text to predict emotion
    text = load_text(input_file)
    # Model constants.
    max_features = 20000
    embedding_dim = 128
    sequence_length = 50

    vectorizer = TextVectorization(max_tokens=max_features, output_mode="int", output_sequence_length=sequence_length)
    text_vector = vectorizer(text)

    prediction = model.predict(text_vector, batch_size=32)
    emotion_code = np.argmax(prediction)
    print("SCORE FOR ALL 8 EMOTIONS", prediction)
    print("INPUT TEXT == ", text)
    print("PREDICTED EMOTION ==", decoder.emotion_decode(emotion_code))

def load_model():
    # to load the model
    # model is located at model directory
    model_name = "emotion_text_model"
    load_dir = os.path.dirname(os.path.normpath(os.getcwd()))
    load_dir = os.path.join(load_dir, 'models')
    if not os.path.isdir(load_dir):
        print("NO MODEL DIRECTORY")
    else: 
        model_path = os.path.join(load_dir, model_name)
        print("MODEL PATH ==", model_path)
        if not os.path.isdir(model_path):
            print("NO MODEL")
        else:
            model = tf.keras.models.load_model(model_path)
            return model

def load_text(input_file):
    #text_file_name = "empty.txt"
    # to get parent directory of current directory
    #load_dir = os.path.dirname(os.path.normpath(os.getcwd()))
    #load_dir = os.path.join(load_dir, 'data')
    #if not os.path.isdir(load_dir):
    #    print("No input directory.")
    #else:
        #text_file_path = os.path.join(load_dir, text_file_name)
    if not os.path.isfile(input_file):
        print("NO INPUT")
    else:
        with open(input_file, 'r', encoding='latin1') as reader:
            data = reader.read().replace('\n', ' ')
            #for line in reader:
            #    samples += line
        input_text = np.array([[data]], dtype='object')
        return input_text


def main():
    print("==== WELCOME TO THE TEXT EMOTION PREDICTION ====")
    input_file_name = str(sys.argv[1])
    predict_emotion(input_file_name)

if __name__ == "__main__":
    main()
