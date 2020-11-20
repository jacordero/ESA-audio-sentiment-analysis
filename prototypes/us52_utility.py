import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from pydub import AudioSegment
from pydub.utils import make_chunks

__authors__ = "Raha Sadeghi, Parima Mirshafiei"
__email__ = "r.sadeghi@tue.nl; P.mirshafiei@tue.nl"
__copyright__ = "TU/e ST2019"
__version__ = "1.0"
__status__ = "Prototype"


class UtilityClass:
    """
    this class provides some functions that could be used for preprocessing, saving or sketching the model result
    """

    def __init__(self):
        pass

    @staticmethod
    def save_model(trained_model):
        """
        saving model as h5 file
        :param trained_model the trained model to be saved
        """
        model_name = 'Emotion_Voice_Detection_Model.h5'
        save_dir = os.path.join(os.getcwd(), 'saved_models')
        # Save model and weights
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        model_path = os.path.join(save_dir, model_name)
        trained_model.save(model_path)
        print('Saved trained model at %s ' % model_path)

        model_json = trained_model.to_json()
        with open("model.json", "w") as json_file:
            json_file.write(model_json)

    @staticmethod
    def plot_trained_model(model_history):
        """
        plotting the model history, depicting its accuracy, useful in finding overfitting
        :param model_history the history of the model (its accuracy in each epoch)
        """
        # Loss plotting
        plt.plot(model_history.history['loss'])
        plt.plot(model_history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig('loss.png')
        plt.close()

        # Accuracy plotting
        plt.plot(model_history.history['accuracy'])
        plt.plot(model_history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('acc')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig('accuracy.png')

    @staticmethod
    def model_summary(model, x_testcnn, y_test):
        if len(x_testcnn) > 1:
            predict_vector = model.predict(x_testcnn)
            predictions = np.argmax(predict_vector, axis=1)
            new_y_test = y_test.astype(int)
            matrix = confusion_matrix(new_y_test, predictions)
            print(classification_report(new_y_test, predictions))
            print(matrix)

    @staticmethod
    def audio_split(audio_file_path, chunk_length_in_sec=5):
        """
        splitting audio files into chunks with specific length.
        default length is 5 seconds.
        :param audio_file_path: full path to the file
        :param chunk_length_in_sec: desired length for each chunk
        """
        audio_file = AudioSegment.from_file(audio_file_path, "wav")
        chunk_length_in_sec = chunk_length_in_sec
        chunk_length_ms = chunk_length_in_sec * 1000
        chunks = make_chunks(audio_file, chunk_length_ms)

        # Export all of the individual chunks as wav files
        file_name = UtilityClass.file_name_extractor(audio_file_path)
        for i, chunk in enumerate(chunks):
            chunk_name = file_name + "_" + str(i) + ".wav"
            print("exporting", chunk_name)
            chunk.export(chunk_name, format="wav")

    @staticmethod
    def file_name_extractor(audio_file_path):
        """
        getting the file name, useful in case we want to replace a file.
        :param audio_file_path: full path to the file (audio track)
        """
        path_separator = os.path.sep
        name = (audio_file_path.split(".wav")[0]).split(path_separator)[-1]
        return name

    @staticmethod
    def audio_padding(audio_file_path, chunk_length_in_sec=7000):
        """
        this function add silence to audio tracks with length less than 5 seconds.
        the new files will be saved with the same name in the same path.
        :param audio_file_path: full path to the file (audio track)
        :param chunk_length_in_sec: desired length for each chunk
        """
        print("audio_file_path", audio_file_path)

        audio_file = AudioSegment.from_file(audio_file_path, "wav")
        duration_in_sec = len(audio_file)  # Length of audio in milli-seconds
        if chunk_length_in_sec > duration_in_sec:
            pad_ms = (chunk_length_in_sec - duration_in_sec)  # milliseconds of silence needed
            pad_ms = (chunk_length_in_sec - duration_in_sec)  # milliseconds of silence needed
        else:
            pad_ms = 0
            print("no padding needed")
        silence = AudioSegment.silent(duration=pad_ms)

        audio = AudioSegment.from_wav(audio_file_path)

        # Adding silence after the audio
        padded = audio + silence
        padded.export(audio_file_path, format='wav')
