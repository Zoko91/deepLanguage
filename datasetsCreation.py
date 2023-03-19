# --------------------- Datasets Creation ---------------------

import librosa, librosa.display
import os
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

data = np.concatenate([en, fr, de, es])

def preprocessMelCoeff(file_path,label):
    # Load audio file
    wav, sr = librosa.load(file_path, sr=16000) # load audio file with 16kHz sample rate
    # Pad or truncate the audio file to 5 seconds
    wav = librosa.util.fix_length(wav,size=80000)
    # Calculate Mel-frequency spectrogram
    spect = librosa.feature.melspectrogram(y=wav, sr=sr, n_mels=128)
    # Convert to log scale (dB). We'll use the peak power as reference.
    spect = librosa.power_to_db(spect, ref=np.max)
    # Calculate MFCCs from Mel-frequency spectrogram
    mfccs = librosa.feature.mfcc(S=spect, n_mfcc=13)
    return mfccs,label


j=0
data_processed = []
for file_path,label in data:
    processed_data = preprocessMelCoeff(file_path,label)
    data_processed.append(processed_data)
    j+=1
    print(j)

# Shuffle the data randomly
np.random.shuffle(data_processed)

# Split the data into training and testing datasets
train_data, test_data = train_test_split(data_processed, test_size=0.25, random_state=42)
test_data, validation_data = train_test_split(test_data, test_size=0.5, random_state=42)

# Split the training and testing datasets into input (x) and output (y)
x_train = [data[0] for data in train_data]
y_train = [data[1] for data in train_data]
x_test = [data[0] for data in test_data]
y_test = [data[1] for data in test_data]
x_validation = [data[0] for data in validation_data]
y_validation = [data[1] for data in validation_data]


# Create TensorFlow datasets from the input and output lists
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
validation_dataset = tf.data.Dataset.from_tensor_slices((x_validation, y_validation))


train_dataset.save('./Data/train_dataset')
test_dataset.save('./Data/test_dataset')
validation_dataset.save('./Data/validation_dataset')
