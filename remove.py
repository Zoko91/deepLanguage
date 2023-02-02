import os

path_to_dir = '/Users/josephbeasse/Desktop/deepLanguage/Data/val_set_en'
for file in os.listdir(path_to_dir):
    if not file.endswith(".wav"):
        os.remove(os.path.join(path_to_dir, file))
        print("Removed")
