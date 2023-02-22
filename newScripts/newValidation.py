# VALIDATION: 0.0
import tensorflow as tf
import tensorflow_io as tfio
from tensorflow import keras
import os
import matplotlib.pyplot as plt
import numpy as np

# FR = os.path.join('../Data', 'val_set_fr')
# EN = os.path.join('../Data', 'val_set_en')
# DE = os.path.join('../Data', 'val_set_de')
# ES = os.path.join('../Data', 'val_set_es')
#
# def get_label(file_path, language):
#     # One-hot encode the language
#     one_hot_encoded_label = tf.one_hot(
#         [LANGUAGES.index(language)], depth=len(LANGUAGES))
#     one_hot_encoded_label = tf.cast(one_hot_encoded_label, tf.int32)
#     return file_path, one_hot_encoded_label
#
#
LANGUAGES = ['english', 'french', 'german', 'spanish']
#
#
# fr = tf.data.Dataset.list_files(FR + '/*.wav')
# fr = fr.map(lambda x: get_label(x, 'french'))
#
# en = tf.data.Dataset.list_files(EN + '/*.wav')
# en = en.map(lambda x: get_label(x, 'english'))
#
# de = tf.data.Dataset.list_files(DE + '/*.wav')
# de = de.map(lambda x: get_label(x, 'german'))
#
# es = tf.data.Dataset.list_files(ES + '/*.wav')
# es = es.map(lambda x: get_label(x, 'spanish'))
#
#
# data = fr.concatenate(en)
# data = data.concatenate(de)
# data = data.concatenate(es)

# Load audio file
# def load_wav_16k_mono(filename):
#     file_contents = tf.io.read_file(filename)
#     wav, sample_rate = tf.audio.decode_wav(file_contents, desired_channels=1)
#     wav = tf.squeeze(wav, axis=-1)
#     sample_rate = tf.cast(sample_rate, dtype=tf.int64)
#     wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000)
#     return wav
#
#
# # Pad or truncate the audio file to 5 seconds
# def preprocess(file_path):
#     wav = load_wav_16k_mono(file_path)
#     wav = wav[:80000]
#     zero_padding = tf.zeros([80000] - tf.shape(wav), dtype=tf.float32)
#     wav = tf.concat([zero_padding, wav], 0)
#     wav = wav / tf.math.reduce_max(wav)
#     return wav
#
#
# def extract_mfccs(file_path, label):
#     preprocessed_audio = preprocess(file_path)
#     # if not tf.reduce_any(tf.math.is_finite(preprocessed_audio)):
#     #     print("Detected NaN values")
#     #     tf.print(file_path)
#     # Get the audio data as a tensor
#     audio_tensor = tf.convert_to_tensor(preprocessed_audio)
#     # Reshape the audio data to 2D for the STFT function
#     audio_tensor = tf.reshape(audio_tensor, (1, -1))
#     # Perform STFT on the audio data
#     stft = tf.signal.stft(audio_tensor, frame_length=2048, frame_step=512)
#     # Get the magnitude of the complex STFT output
#     magnitude = tf.abs(stft)
#     # Apply a logarithm to the magnitude to get the log-magnitude
#     log_magnitude = tf.math.log(magnitude + 1e-9)
#     # Apply a Mel filter bank to the log-magnitude to get the Mel-frequency spectrum
#     mel_spectrum = tf.signal.linear_to_mel_weight_matrix(num_mel_bins=40,
#                                                          num_spectrogram_bins=magnitude.shape[-1],
#                                                          sample_rate=16000,
#                                                          lower_edge_hertz=20,
#                                                          upper_edge_hertz=8000)
#     mel_spectrum = tf.tensordot(log_magnitude, mel_spectrum, 1)
#     # Perform the DCT to get the MFCCs
#     mfccs = tf.signal.dct(mel_spectrum, type=2, axis=-1, norm='ortho')
#     # Get the first 13 MFCCs, which are the most important for speech recognition
#     mfccs = mfccs[..., :13]
#     # Normalize the MFCCs
#     mfccs = (mfccs - tf.math.reduce_mean(mfccs)) / tf.math.reduce_std(mfccs)
#     return mfccs, label




# # --------------------- Prepare the data ---------------------
# data = data.map(extract_mfccs)
# data.save('../Models/newDataValidation')                  # Save the data to a file


# --------------------- Load data & model ---------------------
data = tf.data.Dataset.load('../Models/newDataValidation') # Load the data from a file
data = data.shuffle(4000)
data = data.batch(16)
data = data.prefetch(16)
model = keras.models.load_model('../Models/modelNew.h5')

# --------------------- Evaluate ---------------------
val_loss, val_accuracy = model.evaluate(data)
print("Validation Loss: ", val_loss)
print("Validation Accuracy: ", val_accuracy)

# --------------------- Recap ---------------------
# --------------------- Test ----------------------
# data : newData
# model : modelNew
# accuracy : 0.91
# loss : 0.25
# --------------------- Validation ----------------
# data : newDataValidation
# model : modelNew
# accuracy : 0.86
# loss : 0.39


# --------------------- Confusion Matrix ---------------------
# Define a function to get the label and prediction values for a batch of data
def get_labels_and_predictions(model, data):
    labels = []
    predictions = []
    for batch in data:
        batch_labels = batch[1]
        batch_predictions = model.predict(batch[0])
        labels.append(batch_labels)
        predictions.append(batch_predictions)
    labels = tf.concat(labels, axis=0)
    predictions = tf.concat(predictions, axis=0)
    return labels, predictions

# Get the labels and predictions for the validation dataset
val_labels, val_predictions = get_labels_and_predictions(model, data)
val_labels = tf.squeeze(val_labels, axis=1)
val_predictions = tf.squeeze(val_predictions, axis=1)

# Compute the confusion matrix
confusion_matrix = tf.math.confusion_matrix(
    tf.argmax(val_labels, axis=1), tf.argmax(val_predictions, axis=1))


confusion = tf.math.confusion_matrix(tf.argmax(val_labels, axis=1),
                                     tf.argmax(val_predictions, axis=1),
                                     num_classes=4)
print(confusion)
# tf.Tensor(
# [[913  12  46  28]
#  [ 86 794  74  46]
#  [ 87  37 863  13]
#  [ 55  34  26 885]], shape=(4, 4), dtype=int32)

# --------------------- Plot the confusion matrix---------------------
confusion = confusion.numpy()

# Calculate precision and recall
precision = np.diag(confusion) / np.sum(confusion, axis=0)
recall = np.diag(confusion) / np.sum(confusion, axis=1)

# Create plot
fig, ax = plt.subplots()
im = ax.imshow(confusion, cmap='Blues')

# Add colorbar
cbar = ax.figure.colorbar(im, ax=ax)

# Set axis labels and tick labels
ax.set_xticks(np.arange(len(LANGUAGES)))
ax.set_yticks(np.arange(len(LANGUAGES)))
ax.set_xticklabels(LANGUAGES)
ax.set_yticklabels(LANGUAGES)
ax.set_xlabel("Predicted")
ax.set_ylabel("True")

# Add text annotations for precision and recall
text_colors = ["black", "white"]
thresh = im.norm(confusion.max()) / 2.
for i in range(len(LANGUAGES)):
    for j in range(len(LANGUAGES)):
        color = text_colors[int(im.norm(confusion[i, j]) > thresh)]
        ax.text(j, i, f"{confusion[i, j]:d}\n(p={precision[j]:.2f}, r={recall[i]:.2f})", ha="center", va="center", color=color)

# Show plot
plt.show()






