# Old script used to remove files that were not a .wav file

import os

path_to_dir = '/oldData/val_set_en'
for file in os.listdir(path_to_dir):
    if not file.endswith(".wav"):
        os.remove(os.path.join(path_to_dir, file))
        print("Removed")
