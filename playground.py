# --------------------- Playground ---------------------
import librosa, librosa.display
import numpy as np
import tensorflow as tf
from tensorflow import keras

model = keras.models.load_model('./Models/model.h5')

def preprocessMelCoeff(file_path):
    wav, sr = librosa.load(file_path, sr=16000)
    wav = librosa.util.fix_length(wav,size=80000)
    spect = librosa.feature.melspectrogram(y=wav, sr=sr, n_mels=128)
    spect = librosa.power_to_db(spect, ref=np.max)
    mfccs = librosa.feature.mfcc(S=spect, n_mfcc=13)
    return mfccs


# input_file = '/Users/josephbeasse/Desktop/deepLanguageWebsite/static/temp/recording.wav'
# input_file = '/Users/josephbeasse/Desktop/deepLanguage/newData/fr/output1.wav'
input_file = 'testrecording.wav'
mfcssInput = preprocessMelCoeff(input_file)
mfccs = tf.convert_to_tensor(mfcssInput, dtype=tf.float32)
mfccs = tf.reshape(mfccs, (1, 13, 157, 1))
prediction = model.predict(mfccs)
print(prediction)


language_index = tf.argmax(prediction, axis=1).numpy()[0]
language_mapping = {0: 'French', 1: 'Spanish', 2: 'German', 3: 'English'}
language = language_mapping[language_index]
language_probability = prediction[0, language_index]
print(f"The language of the audio file is: {language} with {language_probability * 100:.2f}% of probability.")
