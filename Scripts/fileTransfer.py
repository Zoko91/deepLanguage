import csv
import os
import shutil

# specify the directory containing the files to be moved
src_dir = '/Users/josephbeasse/Desktop/deepLanguage/Data/en2/clips'

# specify the directory where the files will be moved
dst_dir = '/Users/josephbeasse/Desktop/deepLanguage/Data/en'

# specify the path to the CSV file containing the file names
tsv_path = '/Data/en2/validated.tsv'

# open the CSV file and read the file names
with open(tsv_path, 'r') as f:
    reader = csv.reader(f, delimiter='\t')
    next(reader)   # skip the header
    file_names = [row[1] for row in reader]

# move each file to the destination directory
for file_name in file_names:
    src_path = os.path.join(src_dir, file_name)
    dst_path = os.path.join(dst_dir, file_name)
    shutil.move(src_path, dst_path)
