!curl -s -o logo.png https://colab.research.google.com/img/colab_favicon_256px.png

import os
import re
import cv2
import collections
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from math import sin, cos
from google.colab.patches import cv2_imshow

DATASET_DIR = '/content/drive/MyDrive/Colab Notebooks/GreatBarrierReef'

### Importing the datasets to the worksheet
df_train = pd.read_csv(os.path.join(DATASET_DIR, 'train.csv'))
df_test = pd.read_csv(os.path.join(DATASET_DIR, 'test.csv'))

### Shape(# Rows, # Columns) of the train and test datasets
print(df_train.shape)
print(df_test.shape)

### Head of the train and test datasets
print("Train Head")
print(df_train.head())
print("\n")

print("Test Head")
print(df_test.head())
print("\n")

### Number of Unique Videos
df_train["video_id"].unique()

### Number of Unique Sequences
len(df_train["sequence"].unique())

### Number of Unique Video Frames
len(df_train["video_frame"].unique())

### Number of Unique Sequence Frames
len(df_train["sequence_frame"].unique())

### Number of Unique Image_IDs
len(df_train["image_id"].unique())

### Co-ordinates of the annotations seperated
coordinates = []

for i in range(len(df_train["annotations"])):
  if df_train["annotations"][i] == []:
    coordinates.append([])
  else:
    split_annotations = df_train["annotations"][i].split("}, {")
    for annotaion_s in split_annotations:
      coordinates.append(
          [df_train["image_id"][i], re.sub(r'[{|}]', '', annotaion_s)]
          )

print(coordinates)

### Images with no COTS annotations
no_annotations = 0
for coordinate in coordinates:
  if coordinate[1] == '[]':
    no_annotations += 1

print("Number of Images with no annotations")
print(no_annotations)

print("\n")
print("Percentages of Images with no annotations")
print(round(no_annotations*100/df_train.shape[0],2))


### Annotated Images Seperated
annotated_coordinates = []

for coordinate in coordinates:
  if coordinate[1] != '[]':
    annotated_coordinates.append(coordinate)

print(annotated_coordinates)

### Annotated COTS Count
len(annotated_coordinates)

#### Taking only the Image IDs
annotated_images = []
for coordinate in annotated_coordinates:
  split_coordinates = coordinate[0].split("-")
  annotated_images.append(list(map(int, split_coordinates)))

print(annotated_images)

### The frequencies and percentages of the annotated images in each video
def getSizeOfNestedElement(listOfElem, Elem):
    ''' Get number of elements in a nested list'''
    count = 0
    # Iterate over the list
    for elem in listOfElem:
        if elem[0] == Elem:
            count += 1
    return count

totals_images = collections.Counter(df_train["video_id"])

print("Frequencies of the annotated images")
print("Video 0: " + str(totals_images[0]))
print("Video 1: " + str(totals_images[1]))
print("Video 2: " + str(totals_images[2]))

print("\n")
print("Image count in videos")
print("Video 0: " + str(getSizeOfNestedElement(annotated_images, 0)))
print("Video 1: " + str(getSizeOfNestedElement(annotated_images, 1)))
print("Video 2: " + str(getSizeOfNestedElement(annotated_images, 2)))


print("\n")
print("Percentages of the annotated images")
print("Video 0: " + str(round(
    getSizeOfNestedElement(annotated_images, 0)*100/totals_images[0])
    ) + " %")
print("Video 1: " + str(round(
    getSizeOfNestedElement(annotated_images, 1)*100/totals_images[1])
    ) + " %")
print("Video 2: " + str(round(
    getSizeOfNestedElement(annotated_images, 2)*100/totals_images[2])
    ) + " %")

### The annotated images coordinates
coordinates_list = []
for coordinate in annotated_coordinates:
  split_coordinates = coordinate[1].split(", ")
  temp_coordinate = {
      'id': str(coordinate[0]),
      'x': int(split_coordinates[0].split(": ")[1]), 
      'y': int(split_coordinates[1].split(": ")[1]), 
      'w': int(split_coordinates[2].split(": ")[1]), 
      'h': int(split_coordinates[3].split(": ")[1].replace("]", ""))
  }
  coordinates_list.append(temp_coordinate)

print(coordinates_list)

[d for d in coordinates_list if d['id'] == '0-35']

idl = '2-5778'
split_id = idl.split('-')

img_path = os.path.join(
    DATASET_DIR + '/train_images/video_' + 
    split_id[0] + "/" + split_id[1] + '.jpg'
    )
tmp_im = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
coords = [d for d in coordinates_list if d['id'] == '2-5778']

for coord in coords:
  cv2.rectangle(
      tmp_im, 
      (coord['x'], coord['y']), 
      (coord['x'] + coord['w'], coord['y'] + coord['h']),
      (255, 0, 0), 
      2
      )
cv2_imshow(tmp_im)

idl = '0-124'
split_id = idl.split('-')

img_path = os.path.join(
    DATASET_DIR + '/train_images/video_' + 
    split_id[0] + "/" + split_id[1] + '.jpg'
    )
tmp_im = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2HSV)

coords = [d for d in coordinates_list if d['id'].split("-")[0] == '0']
for coord in coords:
  cv2.rectangle(
      tmp_im, 
      (round(coord['x'] + (coord['w']/2) - 1), round(coord['y'] + (coord['h']/2) - 1)),
      (round(coord['x'] + (coord['w']/2) + 1), round(coord['y'] + (coord['h']/2) + 1)),
      (255, 0, 0), 
      2
      )
cv2_imshow(tmp_im)

idl = '1-497'
split_id = idl.split('-')

img_path = os.path.join(
    DATASET_DIR + '/train_images/video_' + 
    split_id[0] + "/" + split_id[1] + '.jpg'
    )
tmp_im = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
coords = [d for d in coordinates_list if d['id'].split("-")[0] == '1']

for coord in coords:
  cv2.rectangle(
      tmp_im, 
      (round(coord['x'] + (coord['w']/2) - 1), round(coord['y'] + (coord['h']/2) - 1)),
      (round(coord['x'] + (coord['w']/2) + 1), round(coord['y'] + (coord['h']/2) + 1)),
      (255, 0, 0), 
      2
      )
cv2_imshow(tmp_im)

idl = '2-5778'
split_id = idl.split('-')

img_path = os.path.join(
    DATASET_DIR + '/train_images/video_' + 
    split_id[0] + "/" + split_id[1] + '.jpg'
    )
tmp_im = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
coords = [d for d in coordinates_list if d['id'].split("-")[0] == '1']

for coord in coords:
  cv2.rectangle(
      tmp_im, 
      (round(coord['x'] + (coord['w']/2) - 1), round(coord['y'] + (coord['h']/2) - 1)),
      (round(coord['x'] + (coord['w']/2) + 1), round(coord['y'] + (coord['h']/2) + 1)),
      (255, 0, 0), 
      2
      )
cv2_imshow(tmp_im)

