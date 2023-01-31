# Load Dependencies
import os
import tensorflow as tf
import tensorflow_io as tfio
from tensorflow import keras
import matplotlib.pyplot as plt

model = keras.models.load_model('Models/model.h5')


# Load audio file
def load_wav_16k_mono(filename):
    file_contents = tf.io.read_file(filename)
    wav, sample_rate = tf.audio.decode_wav(file_contents, desired_channels=1)
    wav = tf.squeeze(wav, axis=-1)
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000)
    return wav


# Pad or truncate the audio file to 5 seconds
def preprocess(file_path):
    wav = load_wav_16k_mono(file_path)
    wav = wav[:80000]
    zero_padding = tf.zeros([80000] - tf.shape(wav), dtype=tf.float32)
    wav = tf.concat([zero_padding, wav], 0)
    wav = wav / tf.math.reduce_max(wav)
    return wav


def extract_mfccs(file_path):
    preprocessed_audio = preprocess(file_path)
    # if not tf.reduce_any(tf.math.is_finite(preprocessed_audio)):
    #     print("Detected NaN values")
    #     tf.print(file_path)
    # Get the audio data as a tensor
    audio_tensor = tf.convert_to_tensor(preprocessed_audio)
    # Reshape the audio data to 2D for the STFT function
    audio_tensor = tf.reshape(audio_tensor, (1, -1))
    # Perform STFT on the audio data
    stft = tf.signal.stft(audio_tensor, frame_length=2048, frame_step=512)
    # Get the magnitude of the complex STFT output
    magnitude = tf.abs(stft)
    # Apply a logarithm to the magnitude to get the log-magnitude
    log_magnitude = tf.math.log(magnitude + 1e-9)
    # Apply a Mel filter bank to the log-magnitude to get the Mel-frequency spectrum
    mel_spectrum = tf.signal.linear_to_mel_weight_matrix(num_mel_bins=40,
                                                         num_spectrogram_bins=magnitude.shape[-1],
                                                         sample_rate=16000,
                                                         lower_edge_hertz=20,
                                                         upper_edge_hertz=8000)
    mel_spectrum = tf.tensordot(log_magnitude, mel_spectrum, 1)
    # Perform the DCT to get the MFCCs
    mfccs = tf.signal.dct(mel_spectrum, type=2, axis=-1, norm='ortho')
    # Get the first 13 MFCCs, which are the most important for speech recognition
    mfccs = mfccs[..., :13]
    # Normalize the MFCCs
    mfccs = (mfccs - tf.math.reduce_mean(mfccs)) / tf.math.reduce_std(mfccs)
    return mfccs


input_file = 'Audios/Thomas2.wav'
inputMFCC = extract_mfccs(input_file)
inputMFCC = tf.expand_dims(inputMFCC, axis=0)
prediction = model.predict(inputMFCC)
# get the index of the predicted class
language_index = tf.argmax(prediction, axis=1).numpy()[0]
# define the mapping of class index to language
language_mapping = {0: 'French', 1: 'English', 2: 'German', 3: 'Spanish'}
# get the predicted language
language = language_mapping[language_index]
# get the probability of the predicted class
language_probability = prediction[0, language_index]
# print the predicted language and its probability
print(f"The language of the audio file is: {language} with {language_probability * 100:.2f}% of probability.")
