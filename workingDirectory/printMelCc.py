import librosa, librosa.display
from matplotlib import pyplot as plt
import os
import numpy as np

filepath = os.path.join('../Data', 'fr_wav', 'output134.wav')

# def preprocessMelSpect(file_path):
#     # Load audio file
#     wav, sr = librosa.load(file_path, sr=16000) # load audio file with 16kHz sample rate
#     # Pad or truncate the audio file to 5 seconds
#     wav = librosa.util.fix_length(wav, sr*5,size=80000)
#     # Calculate Mel-frequency spectrogram
#     spect = librosa.feature.melspectrogram(wav, sr=sr, n_mels=128)
#     # Convert to log scale (dB). We'll use the peak power as reference.
#     spect = librosa.power_to_db(spect, ref=np.max)
#     return spect


# spectrogram = preprocessMelSpect(filepath)
# # plot the melspectrogram
# plt.figure(figsize=(10, 4))
# librosa.display.specshow(spectrogram, x_axis='time', y_axis='mel', sr=16000, fmax=8000)
# plt.colorbar(format='%+2.0f dB')
# plt.title('Mel-frequency spectrogram')
# plt.tight_layout()
# plt.show()




def preprocessMelCoeff(file_path):
    # Load audio file
    wav, sr = librosa.load(file_path, sr=16000) # load audio file with 16kHz sample rate
    # Pad or truncate the audio file to 5 seconds
    wav = librosa.util.fix_length(wav,sr=sr,size=80000)
    # Calculate Mel-frequency spectrogram
    spect = librosa.feature.melspectrogram(wav, sr=sr, n_mels=128)
    # Convert to log scale (dB). We'll use the peak power as reference.
    spect = librosa.power_to_db(spect, ref=np.max)
    # Calculate MFCCs from Mel-frequency spectrogram
    mfccs = librosa.feature.mfcc(S=spect, n_mfcc=13)
    return mfccs



mfccs= preprocessMelCoeff(filepath)
frame_times = librosa.frames_to_time(range(mfccs.shape[1]), sr=16000)
# SHAPE : ndarray  : (13,157)

plt.figure(figsize=(10, 4))
librosa.display.specshow(mfccs, x_axis='time', sr=16000, hop_length=512, x_coords=frame_times)
plt.colorbar()
plt.title('MFCCs')
plt.tight_layout()
plt.show()


