#mount drive
%cd ..
from google.colab import drive
drive.mount('/content/gdrive',force_remount=True)

import pandas as pd
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import os
import json
import re

# Annotations in Yolo format generated
train_csv_read = pd.read_csv('/content/gdrive/MyDrive/DL_Project/tensorflow-great-barrier-reef/train.csv')

#text file to be written into directory video_{video_id}
#text file name {video_frame}.txt
#text file content - processed annotations with each new bbox annotation in new line

root = '/content/gdrive/MyDrive/DL_Project/tensorflow-great-barrier-reef/train_images/'
for i in range(len(train_csv_read)):
  directory = 'video_' + str(train_csv_read['video_id'][i])
  filename = str(train_csv_read['video_frame'][i])+'.txt'
  re_singlequote = re.compile(r'\'')
  json_string = train_csv_read['annotations'][i]
  json_string = re_singlequote.sub('"',json_string)
  xydict = json.loads(json_string)
  yolo_string = ''
  for k in xydict:
    for key, value in k.items():
      if key == 'y' or key == 'width':
        yolo_string+=str(value)
        yolo_string+=','
      if key == 'x':
        yolo_string+=('0')
        yolo_string+=','
        yolo_string+=str(value)
        yolo_string+=','
      if key == 'height':
        yolo_string+=str(value)
        yolo_string+= '\n'
  print(directory)
  print(filename)
  print(yolo_string)
  completeName = os.path.join(root,directory,filename)
  print(completeName)
  file1 = open(completeName,'w')
  file1.write(yolo_string)
  file1.close()
  
directory = '/content/gdrive/MyDrive/DL_Project/Kavya/yolov4/darknet/build/darknet/x64/data/'
filename = 'obj.names'
content = 'starfish'
completeName = os.path.join(directory,filename)
file1 = open(completeName,'w')
file1.write(content)
file1.close()

directory = '/content/gdrive/MyDrive/DL_Project/Kavya/yolov4/darknet/build/darknet/x64/data/'
filename = 'obj.data'
content = 'classes = 1\ntrain  = /content/gdrive/MyDrive/DL_Project/Kavya/yolov4/darknet/build/darknet/x64/data/train.txt\nvalid  = /content/gdrive/MyDrive/DL_Project/Kavya/yolov4/darknet/build/darknet/x64/data/valid.txt\nnames = data/obj.names\nbackup = backup/'
completeName = os.path.join(directory,filename)
file1 = open(completeName,'w')
file1.write(content)
file1.close()

#rename all images in video_id folder as video_id_frame.jpg
root = '/content/gdrive/MyDrive/DL_Project/tensorflow-great-barrier-reef/train_images/video_0/'
for filename in os.listdir(root):
  src = root+filename
  dst = root+'video_0_'+filename
  os.rename(src,dst)

#rename all images in video_id folder as video_id_frame.jpg
root = '/content/gdrive/MyDrive/DL_Project/tensorflow-great-barrier-reef/train_images/video_1/'
for filename in os.listdir(root):
  src = root+filename
  dst = root+'video_1_'+filename
  os.rename(src,dst)

#rename all images in video_id folder as video_id_frame.jpg
root = '/content/gdrive/MyDrive/DL_Project/tensorflow-great-barrier-reef/train_images/video_2/'
for filename in os.listdir(root):
  src = root+filename
  dst = root+'video_2_'+filename
  os.rename(src,dst)

#move all files to /content/gdrive/MyDrive/DL_Project/Kavya/yolov4/darknet/build/darknet/x64/data/obj/
import shutil
parent_dir = '/content/gdrive/MyDrive/DL_Project/Kavya/yolov4/darknet/build/darknet/x64/data/'
dir = 'obj'
os.mkdir(os.path.join(parent_dir,dir))
dst = os.path.join(parent_dir,dir)
src_file_root = '/content/gdrive/MyDrive/DL_Project/tensorflow-great-barrier-reef/train_images/video_0/'
for filename in os.listdir(src_file_root):
  src = src_file_root+filename
  dst_file = dst+'/'+filename
  shutil.move(src,dst_file)

#move all files to /content/gdrive/MyDrive/DL_Project/Kavya/yolov4/darknet/build/darknet/x64/data/obj/
import shutil
dst = '/content/gdrive/MyDrive/DL_Project/Kavya/yolov4/darknet/build/darknet/x64/data/obj/'
src_file_root = '/content/gdrive/MyDrive/DL_Project/tensorflow-great-barrier-reef/train_images/video_1/'
for filename in os.listdir(src_file_root):
  src = src_file_root+filename
  dst_file = dst+filename
  shutil.move(src,dst_file)

#move all files to /content/gdrive/MyDrive/DL_Project/Kavya/yolov4/darknet/build/darknet/x64/data/obj/
import shutil
dst = '/content/gdrive/MyDrive/DL_Project/Kavya/yolov4/darknet/build/darknet/x64/data/obj/'
src_file_root = '/content/gdrive/MyDrive/DL_Project/tensorflow-great-barrier-reef/train_images/video_2/'
for filename in os.listdir(src_file_root):
  src = src_file_root+filename
  dst_file = dst+filename
  shutil.move(src,dst_file)

  
# creating train.txt and valid.txt files
import glob, os

current_dir = '/content/gdrive/MyDrive/DL_Project/Kavya/yolov4/darknet/build/darknet/x64/data/obj'

# Percentage of images to be used for the valid set
percentage_valid = 20;

# Create and/or truncate train.txt and valid.txt
file_train = open('/content/gdrive/MyDrive/DL_Project/Kavya/yolov4/darknet/build/darknet/x64/data/train.txt', 'w')
file_valid = open('/content/gdrive/MyDrive/DL_Project/Kavya/yolov4/darknet/build/darknet/x64/data/valid.txt', 'w')

# Populate train.txt and valid.txt
counter = 1
index_val = round(100 / percentage_valid)
for pathAndFilename in glob.iglob(os.path.join(current_dir, "*.jpg")):
    title, ext = os.path.splitext(os.path.basename(pathAndFilename))

    if counter == index_val:
        counter = 1
        file_valid.write("/content/gdrive/MyDrive/DL_Project/Kavya/yolov4/darknet/build/darknet/x64/data/obj" + "/" + title + '.jpg' + "\n")
    else:
        file_train.write("/content/gdrive/MyDrive/DL_Project/Kavya/yolov4/darknet/build/darknet/x64/data/obj" + "/" + title + '.jpg' + "\n")
        counter = counter + 1
