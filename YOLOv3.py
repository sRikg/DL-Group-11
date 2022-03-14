from google.colab import drive
drive.mount('/content/gdrive')

# ONLY ONE TIME RUN-DO NOT RERUN
import shutil
shutil.move('/content/gdrive/MyDrive/DL_Project/Kavya/yolov4/darknet/cfg/yolo-obj.cfg','/content/gdrive/MyDrive/DL_Project/Kavya/yolov4/darknet/build/darknet/x64/cfg/yolo-obj.cfg')

import shutil
shutil.move('/content/gdrive/MyDrive/DL_Project/Kavya/yolov4/darknet/build/darknet/Makefile','/content/gdrive/MyDrive/DL_Project/Kavya/yolov4/darknet/Makefile')

# TRAINING BLOCK
%cd /content/gdrive/MyDrive/DL_Project/Kavya/yolov4/darknet/

%cd /content/gdrive/MyDrive/DL_Project/Kavya/yolov4/darknet/
!sed -i 's/OPENCV=0/OPENCV=1/' Makefile
!sed -i 's/GPU=0/GPU=1/' Makefile
!sed -i 's/CUDNN=0/CUDNN=1/' Makefile
!sed -i 's/CUDNN_HALF=0/CUDNN_HALF=1/' Makefile
!sed -i 's/LIBSO=0/LIBSO=1/' Makefile

#make file
!make
%ls /content/gdrive/MyDrive/DL_Project/Clarissa/yolov3-obj.cfg

# Train model
!./darknet detector train /content/gdrive/MyDrive/DL_Project/Kavya/yolov4/darknet/build/darknet/x64/data/obj.data /content/gdrive/MyDrive/DL_Project/Clarissa/yolov3-obj.cfg darknet53.conv.74 -dont_show -map

# Check Metrics
!./darknet detector map /content/gdrive/MyDrive/DL_Project/Kavya/yolov4/darknet/build/darknet/x64/data/obj.data /content/gdrive/MyDrive/DL_Project/Clarissa/yolov3-obj.cfg /content/gdrive/MyDrive/DL_Project/Kavya/yolov4/darknet/backup/yolov3-obj_last.weights
!./darknet detector map /content/gdrive/MyDrive/DL_Project/Kavya/yolov4/darknet/build/darknet/x64/data/obj.data /content/gdrive/MyDrive/DL_Project/Clarissa/yolov3-obj.cfg /content/gdrive/MyDrive/DL_Project/Kavya/yolov4/darknet/backup/yolov3-obj_1000.weights
!./darknet detector map /content/gdrive/MyDrive/DL_Project/Kavya/yolov4/darknet/build/darknet/x64/data/obj.data /content/gdrive/MyDrive/DL_Project/Clarissa/yolov3-obj.cfg /content/gdrive/MyDrive/DL_Project/Kavya/yolov4/darknet/backup/yolov3-obj_last.weights
