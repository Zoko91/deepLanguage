# --------------------- Playground ---------------------
import tensorflow as tf
import tensorflow_io as tfio
from tensorflow import keras
import os

FR = os.path.join('Audios', 'fr')
EN = os.path.join('Audios', 'en')
DE = os.path.join('Audios', 'de')
ES = os.path.join('Audios', 'es')

def get_label(file_path, language):
    # One-hot encode the language
    one_hot_encoded_label = tf.one_hot(
        [LANGUAGES.index(language)], depth=len(LANGUAGES))
    one_hot_encoded_label = tf.cast(one_hot_encoded_label, tf.int32)
    return file_path, one_hot_encoded_label


LANGUAGES = ['english', 'french', 'german', 'spanish']


fr = tf.data.Dataset.list_files(FR + '/*.wav')
fr = fr.map(lambda x: get_label(x, 'french'))

en = tf.data.Dataset.list_files(EN + '/*.wav')
en = en.map(lambda x: get_label(x, 'english'))

de = tf.data.Dataset.list_files(DE + '/*.wav')
de = de.map(lambda x: get_label(x, 'german'))

es = tf.data.Dataset.list_files(ES + '/*.wav')
es = es.map(lambda x: get_label(x, 'spanish'))

data = fr.concatenate(en)
data = data.concatenate(de)
data = data.concatenate(es)

# Load audio file
def load_wav_16k_mono(filename):
    file_contents = tf.io.read_file(filename)
    wav, sample_rate = tf.audio.decode_wav(file_contents, desired_channels=1)
    wav = tf.squeeze(wav, axis=-1)
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000)
    return wav


# Pad or truncate the audio file to 5 seconds
def preprocess(file_wav):
    zero_padding = tf.zeros([80000] - tf.shape(file_wav), dtype=tf.float32)
    wav = tf.concat([zero_padding, file_wav], 0)
    wav = wav / tf.math.reduce_max(wav)
    return wav



def extract_mfccs(file_wav, label):
    preprocessed_audio = preprocess(file_wav)
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


augmented_dataset = None
frame_length = 80000

for file_path, labels in data:
    wav = load_wav_16k_mono(file_path)
    if wav.shape[0] <= frame_length:
        augmented_data = tf.data.Dataset.from_tensors((wav,labels))
        if augmented_dataset is None:
            augmented_dataset = augmented_data
        else:
            augmented_dataset = augmented_dataset.concatenate(augmented_data)
    else:
        portions = tf.signal.frame(wav, frame_length, frame_length, axis=0,pad_end=False)
        for portion in portions:
            augmented_data = tf.data.Dataset.from_tensors((portion,labels))
            if augmented_dataset is None:
                augmented_dataset = augmented_data
            else:
                augmented_dataset = augmented_dataset.concatenate(augmented_data)

    # data_size = augmented_dataset.reduce(0, lambda state, _: state + 1)
    # print('Data size: ', data_size.numpy())  # Data size: 159

# --------------------- Prepare the data ---------------------
augmented_dataset = augmented_dataset.map(extract_mfccs)
augmented_dataset.save('Models/validationTestOnline')               # Save the data to a file
# augmented_dataset = augmented_dataset.shuffle(200)
# augmented_dataset = augmented_dataset.batch(8)
# augmented_dataset = augmented_dataset.prefetch(4)
# model = keras.models.load_model('Models/model2.h5')
#
# # # --------------------- Evaluate ---------------------
# val_loss, val_accuracy = model.evaluate(augmented_dataset)
# print("Validation Loss: ", val_loss)
# print("Validation Accuracy: ", val_accuracy)


# Data size: 159
# Validation Loss:  3.718963861465454
# Validation Accuracy:  0.43396225571632385
