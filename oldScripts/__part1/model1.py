# Load Dependencies
import os
import tensorflow as tf
import tensorflow_io as tfio
from matplotlib import pyplot as plt
import math

# Define Paths to Files
fr_audios = os.path.join('../Data', 'fr_wav', 'output0.wav')
en_audios = os.path.join('../Data', 'en_wav', 'output0.wav')


def load_wav_16k_mono(filename):
    # Load encoded wav file
    file_contents = tf.io.read_file(filename)
    # Decode wav (tensors by channels)
    wav, sample_rate = tf.audio.decode_wav(file_contents, desired_channels=1)
    # Removes trailing axis
    wav = tf.squeeze(wav, axis=-1)
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    # Goes from 44100Hz to 16000hz - amplitude of the audio signal
    wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000)
    return wav


FR = os.path.join('../Data', 'fr_wav')
EN = os.path.join('../Data', 'en_wav')

fr = tf.data.Dataset.list_files(FR + '/*.wav')
en = tf.data.Dataset.list_files(EN + '/*.wav')

frMapped = tf.data.Dataset.zip((fr, tf.data.Dataset.from_tensor_slices(tf.ones(len(fr)))))
enMapped = tf.data.Dataset.zip((en, tf.data.Dataset.from_tensor_slices(tf.zeros(len(en)))))
data = frMapped.concatenate(enMapped)


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


# filepath, label = frMapped.shuffle(buffer_size=10000).as_numpy_iterator().next()
# spectrogram, label = preprocess(filepath, label)
# plt.figure(figsize=(30, 20))
# plt.imshow(tf.transpose(spectrogram)[0], cmap='jet')
# # Add title and labels
# plt.title(label)
# plt.xlabel('Time')
# plt.ylabel('Frequency')
# plt.show()

data = data.map(preprocess)
data = data.cache()
data = data.shuffle(buffer_size=300)
data = data.batch(16)
data = data.prefetch(8)

train = data.take(100)
test = data.skip(100).take(20)

# samples, labels = train.as_numpy_iterator().next()
# print(samples.shape)  # result:  (16, 2491, 257, 1)
print("Creating sequential model")
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(2491, 257, 1)))
model.add(tf.keras.layers.Conv2D(16, (3, 3), activation='relu'))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
model.compile('Adam', loss='BinaryCrossentropy', metrics=[tf.keras.metrics.Recall(), tf.keras.metrics.Precision()])
model.summary()

# model.summary()  # Total params:  1,288,626,865

print("Time to train the Model")
# Train the model
# --------------------------------------------------------------
history = model.fit(train, epochs=4, validation_data=test, verbose=1)
# --------------------------------------------------------------
plt.title('Loss')
plt.plot(history.history['loss'], 'r')
plt.plot(history.history['val_loss'], 'b')
plt.show()
plt.title('Precision')
plt.plot(history.history['precision'], 'r')
plt.plot(history.history['val_precision'], 'b')
plt.show()
plt.title('Recall')
plt.plot(history.history['recall'], 'r')
plt.plot(history.history['val_recall'], 'b')
plt.show()

