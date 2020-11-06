from vosk import Model, KaldiRecognizer
import wave
import json


class AudioToTxt:
    def __init__(self):
        self.model_path = "../models/vosk-model-en-us-daanzu-20200905-lgraph"
        self.wav_file = "../data/03-02-01-01-01-01-01.wav"
        self.model = Model(self.model_path)
        self.wf = wave.open(self.wav_file, "rb")
        self.rec = KaldiRecognizer(self.model, self.wf.getframerate())

    def transcribe(self):
        output = ""
        phrase = ""
        while True:
            data = self.wf.readframes(4000)
            if len(data) == 0:
                break
            if self.rec.AcceptWaveform(data):
                res = json.loads(self.rec.Result())
                print(res.get("text"))
                phrase = phrase + res.get("text") + " "
                output = res.get("text")
            else:
                print(self.rec.PartialResult())

        return output

if __name__ == '__main__':
    audio2txt = AudioToTxt()
    print("-------------> " + audio2txt.transcribe())