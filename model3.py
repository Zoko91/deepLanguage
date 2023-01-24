
# TUTO 3: new implementations of mfccs and not spectrograms

import os
import tensorflow as tf
import tensorflow_io as tfio

# Increase the amount of data you're using to train the model
FR = os.path.join('Data', 'fr_wav')
EN = os.path.join('Data', 'en_wav')
DE = os.path.join('Data', 'de_wav')
ES = os.path.join('Data', 'es_wav')

# On kaggle use the following:
# FR = os.path.join('data-fr')
# EN = os.path.join('data-en')
# DE = os.path.join('data-de')
# ES = os.path.join('data-es')

fr = tf.data.Dataset.list_files(FR + '/*.wav')
en = tf.data.Dataset.list_files(EN + '/*.wav')
de = tf.data.Dataset.list_files(DE + '/*.wav')
es = tf.data.Dataset.list_files(ES + '/*.wav')

frMapped = tf.data.Dataset.zip((fr, tf.data.Dataset.from_tensor_slices(tf.ones(len(fr)))))
enMapped = tf.data.Dataset.zip((en, tf.data.Dataset.from_tensor_slices(tf.zeros(len(en)))))
deMapped = tf.data.Dataset.zip((de, tf.data.Dataset.from_tensor_slices(tf.ones(len(de))*2)))
esMapped = tf.data.Dataset.zip((es, tf.data.Dataset.from_tensor_slices(tf.ones(len(es))*3)))
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


# Data augmentation
def data_augmentation(file_path, label):
    wav = load_wav_16k_mono(file_path)
    wav = wav[:80000]
    zero_padding = tf.zeros([80000] - tf.shape(wav), dtype=tf.float32)
    wav = tf.concat([zero_padding, wav], 0)
    wav = wav / tf.math.reduce_max(wav)
    # Add random time shifting

    stft = tf.signal.stft(wav, frame_length=2048, frame_step=256, fft_length=2048)
    spectrogram = tf.abs(stft)

    # Compute the Mel-scaled power spectrogram
    mel_spectrogram = tf.signal.linear_to_mel_weight_matrix(40, 1025, 16000)
    mel_spectrogram = tf.matmul(spectrogram, mel_spectrogram)
    mel_spectrogram = tf.math.log(mel_spectrogram + 1e-6)

    # Compute the MFCCs
    mfccs = tf.signal.mfccs_from_log_mel_spectrograms(mel_spectrogram)[..., :13]

    return mfccs, label


data = data.map(data_augmentation)
data = data.cache()
data = data.shuffle(buffer_size=4000)

# Divide the dataset into train and test sets
train = data.take(3200)
test = data.skip(3200).take(800)
