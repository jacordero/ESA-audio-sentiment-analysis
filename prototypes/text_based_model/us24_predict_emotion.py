import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import tensorflow as tf
tf.get_logger().setLevel('INFO')
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

def predict_emotion(input_file_name):
    model = load_model()
    decoder = Decode_emotion()

    #model.summary()
    # text to predict emotion
    text = load_text(input_file_name)
    print(text)

    prediction = model.predict(text)
    emotion_code = np.argmax(prediction)
    print("PREDICTED EMOTION ==>", decoder.emotion_decode(emotion_code))

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
            model = tf.keras.models.load_model(model_path, custom_objects={'TextVectorization':TextVectorization})
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
