# Load Dependencies
import os
from matplotlib import pyplot as plt
import tensorflow as tf
import tensorflow_io as tfio
import mimetypes #gives file type details

# Define Paths to Files
fr_audios = os.path.join('Data', 'fr_wav', 'output0.wav')
de_audios = os.path.join('Data', 'de_wav', 'output0.wav')


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


FR = os.path.join('Data', 'fr_wav')
DE = os.path.join('Data', 'de_wav')

fr = tf.data.Dataset.list_files(FR+'/*.wav')
de = tf.data.Dataset.list_files(DE+'/*.wav')

frMapped = tf.data.Dataset.zip((fr, tf.data.Dataset.from_tensor_slices(tf.ones(len(fr)))))
deMapped = tf.data.Dataset.zip((de, tf.data.Dataset.from_tensor_slices(tf.zeros(len(de)))))
data = frMapped.concatenate(deMapped)


lengths = []
file_number = 0
file_numbers = []
for file in os.listdir(os.path.join('Data', 'fr_wav')):
    print(str(file_number)+ " - " + file)
    try:
        tensor_wave = load_wav_16k_mono(os.path.join('Data', 'fr_wav', file))
        lengths.append(len(tensor_wave))
    except Exception as e:
        print("Error details: ", e)
        file_type, encoding = mimetypes.guess_type(os.path.join('Data', 'fr_wav', file))
        file_numbers.append(file)

    file_number +=1

print("END OF LOOP --")
print(file_numbers)

