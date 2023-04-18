# Old script used to select random files from a folder
# Selecting them and copying them in another folder

import os
import random
import shutil

# Set the source and destination directories
src_dir = '/oldData/en2/clips'
dst_dir = '/oldData/en'

# Get a list of all files in the source directory
files = os.listdir(src_dir)

# Randomly select 4282 files
selected_files = random.sample(files, 4282)

# Move the selected files to the destination directory
for file in selected_files:
    src_path = os.path.join(src_dir, file)
    dst_path = os.path.join(dst_dir, file)
    shutil.move(src_path, dst_path)
