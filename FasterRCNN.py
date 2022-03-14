#mount drive
%cd ..
from google.colab import drive
drive.mount('/content/gdrive')

!pip install opencv-python-headless==4.5.2.52

import pandas as pd
import numpy as np
import cv2
import os
import re

from collections import defaultdict
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import json
from PIL import Image

!pip install -U albumentations
import albumentations 
from albumentations.pytorch import ToTensorV2

import torch
import torchvision
from torchvision import transforms

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SequentialSampler

from matplotlib import pyplot as plt

train_df = pd.read_csv('/content/gdrive/MyDrive/DL_Project/tensorflow-great-barrier-reef/train.csv')
train_df.shape

train_df.head()

all_annot = ''
for i in range(len(train_df)):
  re_singlequote = re.compile(r'\'')
  json_string = train_df['annotations'][i]
  json_string = re_singlequote.sub('"',json_string)
  xydict = json.loads(json_string)
  if xydict == []:
    yolo_string = train_df['image_id'][i]
    yolo_string+=','
    yolo_string+= '0'
    yolo_string+=','
    yolo_string+= '0'
    yolo_string+=','
    yolo_string+= '0'
    yolo_string+=','
    yolo_string+= '0'
    yolo_string+= '\n'
    all_annot += yolo_string
    
  for k in xydict:
    for key, value in k.items():
      if key == 'x':
        yolo_string = train_df['image_id'][i]
        yolo_string+=','
        yolo_string+=str(value)
        yolo_string+=','
      if key == 'y' or key == 'width':
        yolo_string+=str(value)
        yolo_string+=','
      if key == 'height':
        yolo_string+=str(value)
        yolo_string+= '\n'
    all_annot += yolo_string

li = all_annot.splitlines()
image_id = []
x = []
y =[]
width = []
height = []
training_df = pd.DataFrame()
for a in li:
  g = a.split(',')
  image_id.append(g[0])
  x.append(int(g[1]))
  y.append(int(g[2]))
  width.append(int(g[3]))
  height.append(int(g[4]))
  
training_df['image_id'] = image_id
training_df['x'] = x
training_df['y'] = y
training_df['width'] = width
training_df['height'] = height

training_df.head(100)

training_df = training_df[training_df['width'] > 0]

training_df.head()

from torchvision.transforms.transforms import ToTensor
from numpy.ma.core import asarray
class COTSDataset(Dataset):

    def __init__(self, dataframe, root = '/content/gdrive/MyDrive/DL_Project/tensorflow-great-barrier-reef/train_images/', transforms=None):
        super().__init__()
        self.root = root
        self.image_ids = dataframe['image_id'].unique() 
        self.df = dataframe
        self.image_dir = ['video_' +a[0] for a in self.image_ids] 
        self.transforms = transforms

    def __getitem__(self, index: int):
        image_id = self.image_ids[index]
        records = self.df[self.df['image_id'] == image_id]
        im_dir = self.image_dir[index]
        #print(index)
        image = cv2.imread(root + im_dir +'/'+image_id[2:]+'.jpg', cv2.IMREAD_COLOR)
        #print(index)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        pixels = asarray(image)
        pixels = pixels.astype(np.float32)
        img = Image.open(root + im_dir +'/'+image_id[2:]+'.jpg')
        w,h = img.size
        #print(pixels.max())
        image /= pixels.max()
        #print(index)
        #print(records)
        boxes = records[['x', 'y', 'width', 'height']].values
        #print(index)
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
        #boxes = np.hstack((boxes[:, 0:2], boxes[:, 0:2] + boxes[:, 2:4]-1))

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        area = torch.as_tensor(area, dtype=torch.float32)

        # there is only one class
        labels = torch.ones((records.shape[0],), dtype=torch.int64)
        
        # suppose all instances are not crowd
        iscrowd = torch.zeros((records.shape[0],), dtype=torch.int64)
        
        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        # target['masks'] = None
        target['image_id'] = torch.tensor([index])
        target['area'] = area
        target['iscrowd'] = iscrowd
        if self.transforms:
            sample = {
                'image': image,
                'bboxes': target['boxes'],
                'labels': labels
             }
            sample = self.transforms(**sample)
            image = sample['image']
            
            target['boxes'] = torch.stack(tuple(map(torch.tensor, zip(*sample['bboxes'])))).permute(1, 0)

        return image, target, image_id

    def __len__(self) -> int:
        return self.image_ids.shape[0]

root = '/content/gdrive/MyDrive/DL_Project/tensorflow-great-barrier-reef/train_images/'
train_dataset = COTSDataset(training_df,root)
a,b,c = train_dataset.__getitem__(0)
print(a)
print(b)
print(c)

def get_train_transform():
    return albumentations.Compose([
        albumentations.Flip(0.5),
        ToTensorV2(p=1.0),
    ], bbox_params=albumentations.BboxParams(
        format='pascal_voc', label_fields=['labels']))
    #bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})

#def get_valid_transform():
#    return albumentations.Compose([
#        ToTensorV2(p=1.0)
#    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

num_classes = 2  # 1 class (COTS) + background

# get number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features

# replace the pre-trained head with a new one
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

class Averager:
    def __init__(self):
        self.current_total = 0.0
        self.iterations = 0.0

    def send(self, value):
        self.current_total += value
        self.iterations += 1

    @property
    def value(self):
        if self.iterations == 0:
            return 0
        else:
            return 1.0 * self.current_total / self.iterations

    def reset(self):
        self.current_total = 0.0
        self.iterations = 0.0

def collate_fn(batch):
    return tuple(zip(*batch))
root = '/content/gdrive/MyDrive/DL_Project/tensorflow-great-barrier-reef/train_images/'
train_dataset = COTSDataset(training_df,root,get_train_transform())
#valid_dataset = COTSDataset(valid_df, DIR_TRAIN, get_valid_transform())


# split the dataset in train and test set
indices = torch.randperm(len(train_dataset)).tolist()

train_data_loader = DataLoader(
    train_dataset,
    batch_size=1,
    shuffle=False,
    #num_workers=2,
    collate_fn=collate_fn
)

#valid_data_loader = DataLoader(
#    valid_dataset,
#    batch_size=8,
#    shuffle=False,
#    num_workers=4,
#    collate_fn=collate_fn
#)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

images, targets, image_ids = next(iter(train_data_loader))
images = list(image.to(device) for image in images)
targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

boxes = targets[0]['boxes'].cpu().numpy().astype(np.int32)
sample = images[0].permute(1,2,0).cpu().numpy()

fig, ax = plt.subplots(1, 1, figsize=(16, 8))

for box in boxes:
    cv2.rectangle(sample,
                  (box[0], box[1]),
                  (box[2], box[3]),
                  (220, 0, 0), 3)
    
ax.set_axis_off()
ax.imshow(sample)

model.to(device)
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
lr_scheduler = None

num_epochs = 2

loss_hist = Averager()
itr = 1

for epoch in range(num_epochs):
    loss_hist.reset()
    
    for images, targets, image_ids in train_data_loader:
        
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()

        loss_hist.send(loss_value)

        optimizer.zero_grad()
        losses.backward()
        torch.cuda.empty_cache()
        optimizer.step()

        if itr % 50 == 0:
            print(f"Iteration #{itr} loss: {loss_value}")

        itr += 1
    
    # update the learning rate
    if lr_scheduler is not None:
        lr_scheduler.step()

    print(f"Epoch #{epoch} loss: {loss_hist.value}")   
    
torch.save(model.state_dict(), 'fasterrcnn_resnet50_fpn.pth')
