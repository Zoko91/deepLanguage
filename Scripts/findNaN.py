import os
import tensorflow as tf
import tensorflow_io as tfio

# Increase the amount of data you're using to train the model
FR = os.path.join('../Data', 'fr_wav')
EN = os.path.join('../Data', 'en_wav')
DE = os.path.join('../Data', 'de_wav')
ES = os.path.join('../Data', 'es_wav')
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


for file_path, label in data.take(len(data)):
    preprocessed_audio = preprocess(file_path)
    if not tf.reduce_any(tf.math.is_finite(preprocessed_audio)):
        print(f"Detected NaN values in file: {file_path}")


# French
# None

# English
# Detected NaN values in file: b'../Data/en_wav/output6433.wav'
# Detected NaN values in file: b'../Data/en_wav/output4429.wav'
# Detected NaN values in file: b'../Data/en_wav/output10472.wav'
# Detected NaN values in file: b'../Data/en_wav/output2219.wav'

# German
# Detected NaN values in file: b'../Data/de_wav/output5503.wav'
# Detected NaN values in file: b'../Data/de_wav/output4821.wav'

# Spanish
# None
