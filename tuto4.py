# Machine learning model for language recognition

import os
import tensorflow as tf
import tensorflow_io as tfio
import matplotlib.pyplot as plt
import numpy as np

# Increase the amount of data you're using to train the model
FR = os.path.join('Data', 'fr_wav')
EN = os.path.join('Data', 'en_wav')
DE = os.path.join('Data', 'de_wav')
ES = os.path.join('Data', 'es_wav')

fr = tf.data.Dataset.list_files(FR + '/*.wav')
en = tf.data.Dataset.list_files(EN + '/*.wav')
de = tf.data.Dataset.list_files(DE + '/*.wav')
es = tf.data.Dataset.list_files(ES + '/*.wav')

frMapped = tf.data.Dataset.zip((fr, tf.data.Dataset.from_tensor_slices(tf.ones(len(fr)))))
enMapped = tf.data.Dataset.zip((en, tf.data.Dataset.from_tensor_slices(tf.zeros(len(en)))))
deMapped = tf.data.Dataset.zip((de, tf.data.Dataset.from_tensor_slices(tf.ones(len(de)) * 2)))
esMapped = tf.data.Dataset.zip((es, tf.data.Dataset.from_tensor_slices(tf.ones(len(es)) * 3)))
data = frMapped.concatenate(enMapped)
data = data.concatenate(deMapped)
data = data.concatenate(esMapped)


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


def extract_mfccs(file_path, label):
    preprocessed_audio = preprocess(file_path)
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
    return mfccs, label


# --------------------- Plot the MFCCs ---------------------
# Extract the MFCCs
file_path = 'input.wav'
mfccs, label = extract_mfccs(file_path, 0)

# Plot the MFCCs
plt.imshow(mfccs[0], origin='lower', aspect='auto', cmap='hot')
plt.colorbar()
plt.xlabel('Time')
plt.ylabel('MFCC Coefficients')
plt.title('MFCCs of audio file')
plt.show()

# --------------------- Prepare the data ---------------------
# data = data.map(extract_mfccs)
# # Shuffle the data
# data = data.shuffle(1000)
# data = data.batch(32)
# data = data.prefetch(32)
# # Split the data into training and validation sets
# # Train has 1000 samples, test has 200 samples
# train = data.take(1000)
# test = data.skip(1000).take(200)

#
# samples, labels = train.as_numpy_iterator().next()
# print(samples.shape)  # result:  (32, 1, 153, 13)


# --------------------- Build the model ---------------------


# --------------------- Train the model ---------------------
# history = model.fit(train, epochs=10, validation_data=test, verbose=1)

