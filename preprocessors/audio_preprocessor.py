import librosa
import numpy as np

class AudioPreprocessor:
    @staticmethod
    def preprocess(file):
        try:
            audio, sr = librosa.load(file, sr=16000) # load audio at 16khz
            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13) # mfcc features
            mfcc = np.expand_dims(mfcc, axis=0) # batch dim
            return mfcc
        except Exception as e:
            raise ValueError(f"Geçersiz veya bozuk bir ses dosyası: {str(e)}")