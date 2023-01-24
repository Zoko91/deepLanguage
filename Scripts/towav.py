from pydub import AudioSegment
import os

path_to_dir = '/Users/josephbeasse/Desktop/deepLanguage/ToConvert/'
path_to_wav = '/Users/josephbeasse/Desktop/deepLanguage/AudiosToConvert/'
number_files = 0

for file in os.listdir(path_to_dir):
    if not file.startswith(".DS_Store"):
        file_name = path_to_dir+file.title()
        # Load the MP3 file using the from_mp3() method
        mp3_audio = AudioSegment.from_mp3(file_name)
        name = "output" + str(number_files) + ".wav"
        name = path_to_wav + name
        # Save the audio as a WAV file using the export() method
        mp3_audio.export(name, format="wav")
        number_files += 1
        print(number_files)
        if number_files == 11000:
            break

print(str(number_files) + " files converted")
