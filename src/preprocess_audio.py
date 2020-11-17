import numpy as np
from librosa.feature import mfcc

from src.utils import Utils
from pydub.utils import make_chunks
from pydub import AudioSegment

class Preprocessor():

    @staticmethod
    def audio_split(audio_file_path, chunk_length_in_sec= 5):
        """
        splitting audio files into chunks with specific length.
        default length is 5 seconds.
        :param audio_file_path: full path to the file
        :param chunk_length_in_sec: desired length for each chunk
        """

        audio_file = AudioSegment.from_file(audio_file_path, "wav")

        chunk_length_in_sec = chunk_length_in_sec
        chunk_length_in_ms = chunk_length_in_sec * 1000
        chunks = make_chunks(audio_file, chunk_length_in_ms)

        # Export all of the individual chunks as wav files
        file_name = Utils.file_name_extractor(audio_file_path)
        
        for i, chunk in enumerate(chunks):
            chunk_name = file_name + "_" + str(i) + ".wav"
            print("exporting", chunk_name)
            chunk.export(chunk_name, format="wav")


    @staticmethod
    def audio_padding(audio_file_path, chunk_length_in_sec= 5000):
        """
        this function add silence to audio tracks with length less than 5 seconds.
        the new files will be saved with the same name in the same path.
        :param audio_file_path: full path to the file (audio track)
        :param chunk_length_in_sec: desired length for each chunk
        """
        print("audio_file_path" , audio_file_path)

        audio_file = AudioSegment.from_file(audio_file_path, "wav")
        duration_in_sec = len(audio_file)   # Length of audio in milli-seconds
        if chunk_length_in_sec > duration_in_sec:
            pad_ms = (chunk_length_in_sec - duration_in_sec)   # milliseconds of silence needed
        else:
            print("no padding needed")
            return
        silence = AudioSegment.silent(duration=pad_ms)

        audio = AudioSegment.from_wav(audio_file_path)

        padded = audio + silence  # Adding silence after the audio

        # rewriting the file with new length
        padded.export(audio_file_path, format='wav')
