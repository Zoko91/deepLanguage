# Load Dependencies
import os
import tensorflow as tf
import tensorflow_io as tfio

fr_audios = os.path.join('../../oldData', 'fr_wav', 'output0.wav')
en_audios = os.path.join('../../oldData', 'en_wav', 'output0.wav')


def load_wav_16k_mono(filename):
    file_contents = tf.io.read_file(filename)
    wav, sample_rate = tf.audio.decode_wav(file_contents, desired_channels=1)
    wav = tf.squeeze(wav, axis=-1)
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000)
    return wav


FR = os.path.join('../../oldData', 'fr_wav')
EN = os.path.join('../../oldData', 'en_wav')

fr = tf.data.Dataset.list_files(FR + '/*.wav')
en = tf.data.Dataset.list_files(EN + '/*.wav')

frMapped = tf.data.Dataset.zip((fr, tf.data.Dataset.from_tensor_slices(tf.ones(len(fr)))))
enMapped = tf.data.Dataset.zip((en, tf.data.Dataset.from_tensor_slices(tf.zeros(len(en)))))
data = frMapped.concatenate(enMapped)


def preprocess(file_path, label):
    wav = load_wav_16k_mono(file_path)
    wav = wav[:80000]
    zero_padding = tf.zeros([80000] - tf.shape(wav), dtype=tf.float32)
    wav = tf.concat([zero_padding, wav], 0)
    wav = wav / tf.math.reduce_max(wav)
    stfts = tf.signal.stft(wav, frame_length=320, frame_step=32)
    spectrogram = tf.abs(stfts)
    mfccs = tf.signal.mfccs_from_log_mel_spectrograms(tf.math.log(spectrogram))
    # mfccs = tf.expand_dims(mfccs, -1)
    return mfccs, label


data = data.map(preprocess)
data = data.cache()
data = data.batch(16)
data = data.prefetch(8)
train = data.take(100)
test = data.skip(100).take(20)
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(2491, 257, 1)))
model.add(tf.keras.layers.MaxPooling2D(2, 2))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Conv2D(16, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(2, 2))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.GlobalAveragePooling2D())
model.add(tf.keras.layers.SpatialDropout2D(0.2))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
model.compile(optimizer=tf.keras.optimizers.Adam(), loss='BinaryCrossentropy', metrics=[tf.keras.metrics.Recall(), tf.keras.metrics.Precision()])

# model = tf.keras.models.Sequential()
# model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(2491, 257, 1)))
# model.add(tf.keras.layers.MaxPooling2D(2, 2))
# model.add(tf.keras.layers.BatchNormalization())
# model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))
# model.add(tf.keras.layers.MaxPooling2D(2, 2))
# model.add(tf.keras.layers.BatchNormalization())
# model.add(tf.keras.layers.Flatten())
# model.add(tf.keras.layers.Dense(64, activation='relu'))
# model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
# model.compile(optimizer=tf.keras.optimizers.Adam(), loss='BinaryCrossentropy', metrics=[tf.keras.metrics.Recall(), tf.keras.metrics.Precision()])
# model.summary()  Total params: 78,862,049


history = model.fit(train, epochs=4, validation_data=test, verbose=1)
# Print the training loss
print("Training Loss:", history.history['loss'])
# Print the validation loss
print("Validation Loss:", history.history['val_loss'])
# Print the training accuracy
print("Training Accuracy:", history.history['acc'])
# Print the validation accuracy
print("Validation Accuracy:", history.history['val_acc'])

# save the model
# model.save('model.h5')

