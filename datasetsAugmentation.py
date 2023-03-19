# --------------------- Datasets Augmentation ---------------------

import librosa, librosa.display
import os
import random
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

FR = os.path.join('Data', 'fr')
ES = os.path.join('Data', 'es')
DE = os.path.join('Data', 'de')
EN = os.path.join('Data', 'en')

def get_label(filepath, language):
    # One-hot encode the language
    one_hot_encoded_label = np.zeros(len(LANGUAGES))
    one_hot_encoded_label[LANGUAGES.index(language)] = 1
    return filepath, one_hot_encoded_label


LANGUAGES = ['french', 'spanish', 'german', 'english']

fr = np.array([get_label(os.path.join(root, name), 'french') for root, _, files in os.walk(FR) for name in files if name.endswith('.wav')])
es = np.array([get_label(os.path.join(root, name), 'spanish') for root, _, files in os.walk(ES) for name in files if name.endswith('.wav')])
de = np.array([get_label(os.path.join(root, name), 'german') for root, _, files in os.walk(DE) for name in files if name.endswith('.wav')])
en = np.array([get_label(os.path.join(root, name), 'english') for root, _, files in os.walk(EN) for name in files if name.endswith('.wav')])

data = np.concatenate([fr, es, de, en])
np.random.shuffle(data)

# Split the data into training and testing datasets
train_data, test_data = train_test_split(data, test_size=0.25, random_state=42)
test_data, validation_data = train_test_split(test_data, test_size=0.5, random_state=42)

def preprocessMelCoeffAugmented(file_path,label):
    # Load audio file
    wav, sr = librosa.load(file_path, sr=16000) # load audio file with 16kHz sample rate

    ####################################################################################
    # Data augmentation
    # Randomly shift the audio file up to 1000 samples (about 60ms)
    shift = random.randint(-1000, 1000)
    wav = np.roll(wav, shift)
    # Add random noise
    noise = np.random.normal(0, 0.05, len(wav))  # add Gaussian noise with standard deviation of 0.05
    wav = wav + noise
    # Change pitch by a random factor between -500 and 500 cents (half-steps)
    pitch_shift = random.randint(-500, 500)
    wav = librosa.effects.pitch_shift(wav, sr, n_steps=pitch_shift / 100)
    ####################################################################################

    # Pad or truncate the audio file to 5 seconds
    wav = librosa.util.fix_length(wav,size=80000)
    # Normalize the waveform
    wav = librosa.util.normalize(wav)
    # Calculate Mel-frequency spectrogram
    spect = librosa.feature.melspectrogram(y=wav, sr=sr, n_mels=128)
    # Convert to log scale (dB). We'll use the peak power as reference.
    spect = librosa.power_to_db(spect, ref=np.max)
    # Calculate MFCCs from Mel-frequency spectrogram
    mfccs = librosa.feature.mfcc(S=spect, n_mfcc=13)
    return mfccs,label

def preprocessMelCoeff(file_path,label):
    # Load audio file
    wav, sr = librosa.load(file_path, sr=16000) # load audio file with 16kHz sample rate
    # Pad or truncate the audio file to 5 seconds
    wav = librosa.util.fix_length(wav,size=80000)
    # Normalize the waveform
    wav = librosa.util.normalize(wav)
    # Calculate Mel-frequency spectrogram
    spect = librosa.feature.melspectrogram(y=wav, sr=sr, n_mels=128)
    # Convert to log scale (dB). We'll use the peak power as reference.
    spect = librosa.power_to_db(spect, ref=np.max)
    # Calculate MFCCs from Mel-frequency spectrogram
    mfccs = librosa.feature.mfcc(S=spect, n_mfcc=13)
    return mfccs,label

# Training data:
####################################
print('Training data:')
j=0
data_processed = []
for file_path,label in train_data:
    processed_data = preprocessMelCoeffAugmented(file_path,label)
    data_processed.append(processed_data)
    j+=1
    print(j)


x_train = [data[0] for data in data_processed]
y_train = [data[1] for data in data_processed]
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset.save('./Data/Augmented/train_dataset')


# Validation data:
####################################
print('Validation data:')
j = 0
data_processed = []
for file_path, label in validation_data:
    processed_data = preprocessMelCoeff(file_path, label)
    data_processed.append(processed_data)
    j += 1
    print(j)

x_val = [data[0] for data in data_processed]
y_val = [data[1] for data in data_processed]
val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
val_dataset.save('./Data/Augmented/val_dataset')


# Testing data:
####################################
print('Testing data:')
j = 0
data_processed = []
for file_path, label in validation_data:
    processed_data = preprocessMelCoeff(file_path, label)
    data_processed.append(processed_data)
    j += 1
    print(j)

x_test = [data[0] for data in data_processed]
y_test = [data[1] for data in data_processed]
test_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
test_dataset.save('./Data/Augmented/test_dataset')






