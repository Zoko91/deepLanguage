# Load Dependencies
import os
import tensorflow as tf
import tensorflow_io as tfio
from tensorflow import keras
model = keras.models.load_model('Models/model.h5')


def load_wav_16k_mono(filename):
    file_contents = tf.io.read_file(filename)
    wav, sample_rate = tf.audio.decode_wav(file_contents, desired_channels=1)
    wav = tf.squeeze(wav, axis=-1)
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000)
    return wav


def preprocess(file_path, label):
     # Load audio file
    wav = load_wav_16k_mono(file_path)
    # Pad or truncate the audio file to 5 seconds
    wav = wav[:80000]
    zero_padding = tf.zeros([80000] - tf.shape(wav), dtype=tf.float32)
    # Concatenate audio with padding so that all audio clips will be of the
    wav = tf.concat([zero_padding, wav], 0)
    # Normalize the audio file
    # Do I really need to normalize ???
    wav = wav / tf.math.reduce_max(wav)
    spectrogram = tf.signal.stft(wav, frame_length=320, frame_step=32)
    # Stride of 50% (ratio between frame_length and frame_step)
    # frame_length: 320 = 20ms with 16kHz pooling rate
    # frame_step: 32 = 2ms with 16kHz pooling rate
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.expand_dims(spectrogram, axis=2)
    return spectrogram, label


input_file = 'input.wav'
input_spectrogram, _ = preprocess(input_file, 0)
input_data = tf.expand_dims(input_spectrogram, axis=0)
prediction = model.predict(input_data)
print(model)
print(prediction)
#


