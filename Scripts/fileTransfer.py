import csv
import os
import shutil

# specify the directory containing the files to be moved
src_dir = '../newData/en/clips'

# specify the directory where the files will be moved
dst_dir = '../newData/enMP3'

# specify the path to the CSV file containing the file names
tsv_path = '../newData/en/other.tsv'

# open the CSV file and read the file names
with open(tsv_path, 'r') as f:
    reader = csv.reader(f, delimiter='\t')
    next(reader)   # skip the header
    file_names = [row[1] for row in reader]

indice = 0
# move each file to the destination directory
for file_name in file_names:
    src_path = os.path.join(src_dir, file_name)
    dst_path = os.path.join(dst_dir, file_name)
    if os.path.exists(src_path):
        shutil.move(src_path, dst_path)
        indice +=1
    if indice == 6509:
        break
    #shutil.move(src_path, dst_path)
