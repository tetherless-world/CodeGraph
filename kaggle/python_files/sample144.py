#!/usr/bin/env python
# coding: utf-8

# ## Contents
# ### YOLO v3 for image detection
# 0. <a href='#0.-Introduction'>Introduction</a>
# 1. <a href='#1.-Clone-and-Build-YOLOv3'>Clone and Build YOLOv3</a>
# 2. <a href='#2.-Data-Migration-for-YOLOv3'>Data Migration</a>
# 3. <a href='#3.-Prepare-Configuration-Files-for-Using-YOLOv3'>Prepare Configuration Files for training</a>
# 4. <a href='#4.-Training-YOLOv3'>Training model</a>
# 5. <a href='#5.-How-to-use-trainined-YOLOv3-for-test-images-(command-line)'>How to use trained model for test images (command line)</a>
# 6. <a href='#6.-Generate-Submission-Files-with-YOLOv3-Python-Wrapper'>Generate Submission Files Using YOLOv3 Python Wrapper</a>
# 7. <a href='#7.-Future-works-&-Etc'>Future works & Etc</a>

# ## 0. Introduction
# * I'll introduce super easy and quick way to train [YOLOv3](https://pjreddie.com/darknet/yolo/) on RSNA and to generate submission file (to be honest, not super easy ...!).
# 
# 
# * The purpose of this competition is 'object detection'. Generally, object detection algorithms with deep learning take a long time to train model and require a lot of gpu resources. Most individual participants use one or two gpu (... or zero). Therefore, there is a need for **algorithms that works quickly with less gpu resources.**
# 
# 
# * I tried to use Mask R-CNN, UNet, Fast R-CNN and FCN algorithms, But eventually switched to YOLOv3.
# 
# 
# * In comparison to YOLOv3, Other algorithms(Mask R-CNN, UNet, FCN, ...) which contain instance/sementic segmentation tasks are very slow, and require more gpu resources, redundant parameter tunning and post-processes. Therefore, if you try to use these algorithms, you may experience difficulties in terms of training time and gpu resources. (Please see [YOLOv3 paper](https://pjreddie.com/media/files/papers/YOLOv3.pdf) for details)
# 
# 
# * In addition, **YOLOv3 was able to obtain high score (LB: 0.141) without additional processes(data augmentation, parameter tunning, etc...)** compared to other algorithms. So I think YOLOv3 has sufficient potential for this competition.
# 
# 
# * In this notebook, I'll introduce how to simply apply YOLOv3 on RSNA data. I hope this notebook would be helpful for everyone.

# In[ ]:


import math
import os
import shutil
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import pydicom
import cv2
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# In[ ]:


random_stat = 123
np.random.seed(random_stat)

# ## 1. Clone and Build YOLOv3

# In[ ]:



# Build gpu version darknet

# -j <The # of cpu cores to use>. Chang 999 to fit your environment. Actually i used '-j 50'.

# ## 2. Data Migration for YOLOv3
# It might take a while.

# ### 2.0. Make subdirectories

# In[ ]:


DATA_DIR = "../input"

train_dcm_dir = os.path.join(DATA_DIR, "stage_1_train_images")
test_dcm_dir = os.path.join(DATA_DIR, "stage_1_test_images")

img_dir = os.path.join(os.getcwd(), "images")  # .jpg
label_dir = os.path.join(os.getcwd(), "labels")  # .txt
metadata_dir = os.path.join(os.getcwd(), "metadata") # .txt

# YOLOv3 config file directory
cfg_dir = os.path.join(os.getcwd(), "cfg")
# YOLOv3 training checkpoints will be saved here
backup_dir = os.path.join(os.getcwd(), "backup")

for directory in [img_dir, label_dir, metadata_dir, cfg_dir, backup_dir]:
    if os.path.isdir(directory):
        continue
    os.mkdir(directory)

# In[ ]:



# ### 2.1. Load stage_1_train_labels.csv

# In[ ]:


annots = pd.read_csv(os.path.join(DATA_DIR, "stage_1_train_labels.csv"))
annots.head()

# ### 2.2. Generate images and labels for training YOLOv3
# * YOLOv3 needs .txt file for each image, which contains ground truth object in the image that looks like:
# ```
# <object-class_1> <x_1> <y_1> <width_1> <height_1>
# <object-class_2> <x_2> <y_2> <width_2> <height_2>
# ```
# * <object-class\>: Since RSNA task is binary classification basically, <object-class\> is 0.
# * <x\>, <y\>: Those are float values of bbox center coordinate, divided by image width and height respectively.
# * <w\>, <h\>: Those are width and height of bbox, divided by image width and height respectively.
# 
# * So it is different from the format of label data provided by kaggle. We should change it.

# In[ ]:


def save_img_from_dcm(dcm_dir, img_dir, patient_id):
    img_fp = os.path.join(img_dir, "{}.jpg".format(patient_id))
    if os.path.exists(img_fp):
        return
    dcm_fp = os.path.join(dcm_dir, "{}.dcm".format(patient_id))
    img_1ch = pydicom.read_file(dcm_fp).pixel_array
    img_3ch = np.stack([img_1ch]*3, -1)

    img_fp = os.path.join(img_dir, "{}.jpg".format(patient_id))
    cv2.imwrite(img_fp, img_3ch)
    
def save_label_from_dcm(label_dir, patient_id, row=None):
    # rsna defualt image size
    img_size = 1024
    label_fp = os.path.join(label_dir, "{}.txt".format(patient_id))
    
    f = open(label_fp, "a")
    if row is None:
        f.close()
        return

    top_left_x = row[1]
    top_left_y = row[2]
    w = row[3]
    h = row[4]
    
    # 'r' means relative. 'c' means center.
    rx = top_left_x/img_size
    ry = top_left_y/img_size
    rw = w/img_size
    rh = h/img_size
    rcx = rx+rw/2
    rcy = ry+rh/2
    
    line = "{} {} {} {} {}\n".format(0, rcx, rcy, rw, rh)
    
    f.write(line)
    f.close()
        
def save_yolov3_data_from_rsna(dcm_dir, img_dir, label_dir, annots):
    for row in tqdm(annots.values):
        patient_id = row[0]

        img_fp = os.path.join(img_dir, "{}.jpg".format(patient_id))
        if os.path.exists(img_fp):
            save_label_from_dcm(label_dir, patient_id, row)
            continue

        target = row[5]
        # Since kaggle kernel have samll volume (5GB ?), I didn't contain files with no bbox here.
        if target == 0:
            continue
        save_label_from_dcm(label_dir, patient_id, row)
        save_img_from_dcm(dcm_dir, img_dir, patient_id)

# In[ ]:


save_yolov3_data_from_rsna(train_dcm_dir, img_dir, label_dir, annots)

# In[ ]:



# ### 2.3. Plot a sample train image and label

# In[ ]:


ex_patient_id = annots[annots.Target == 1].patientId.values[0]
ex_img_path = os.path.join(img_dir, "{}.jpg".format(ex_patient_id))
ex_label_path = os.path.join(label_dir, "{}.txt".format(ex_patient_id))

plt.imshow(cv2.imread(ex_img_path))

img_size = 1014
with open(ex_label_path, "r") as f:
    for line in f:
        print(line)
        class_id, rcx, rcy, rw, rh = list(map(float, line.strip().split()))
        x = (rcx-rw/2)*img_size
        y = (rcy-rh/2)*img_size
        w = rw*img_size
        h = rh*img_size
        plt.plot([x, x, x+w, x+w, x], [y, y+h, y+h, y, y])

# ### 2.4. Generate train/val file path list (.txt)
# * We should give the list of image paths to YOLO. two seperate list textfiles for training images and validation images.

# In[ ]:


def write_train_list(metadata_dir, img_dir, name, series):
    list_fp = os.path.join(metadata_dir, name)
    with open(list_fp, "w") as f:
        for patient_id in series:
            line = "{}\n".format(os.path.join(img_dir, "{}.jpg".format(patient_id)))
            f.write(line)

# In[ ]:


# Following lines do not contain data with no bbox
patient_id_series = annots[annots.Target == 1].patientId.drop_duplicates()

tr_series, val_series = train_test_split(patient_id_series, test_size=0.1, random_state=random_stat)
print("The # of train set: {}, The # of validation set: {}".format(tr_series.shape[0], val_series.shape[0]))

# train image path list
write_train_list(metadata_dir, img_dir, "tr_list.txt", tr_series)
# validation image path list
write_train_list(metadata_dir, img_dir, "val_list.txt", val_series)

# ### 2.5. Create test image and labels for YOLOv3

# In[ ]:


def save_yolov3_test_data(test_dcm_dir, img_dir, metadata_dir, name, series):
    list_fp = os.path.join(metadata_dir, name)
    with open(list_fp, "w") as f:
        for patient_id in series:
            save_img_from_dcm(test_dcm_dir, img_dir, patient_id)
            line = "{}\n".format(os.path.join(img_dir, "{}.jpg".format(patient_id)))
            f.write(line)

# In[ ]:


test_dcm_fps = list(set(glob.glob(os.path.join(test_dcm_dir, '*.dcm'))))
test_dcm_fps = pd.Series(test_dcm_fps).apply(lambda dcm_fp: dcm_fp.strip().split("/")[-1].replace(".dcm",""))

save_yolov3_test_data(test_dcm_dir, img_dir, metadata_dir, "te_list.txt", test_dcm_fps)

# ### 2.6. Plot a sample test Image

# In[ ]:


ex_patient_id = test_dcm_fps[0]
ex_img_path = os.path.join(img_dir, "{}.jpg".format(ex_patient_id))

plt.imshow(cv2.imread(ex_img_path))

# ## 3. Prepare Configuration Files for Using YOLOv3
# We should prepare and modify config files, and bring pre-trained weights necessary for training. This proceeds with following four steps.
# ```
#  cfg/rsna.data
#  cfg/rsna.names
#  darknet53.conv.74
#  cfg/rsna_yolov3.cfg_train
# ```

# ### - cfg/rsna.data
# This file point to RSNA data path
#   * train: Path to training image list textfile
#   * val: Path to validation image list textfile
#   * names: RSNA class name list (see <a href='#3.1.-cfg/rsna.names'>3.1</a>)
#   * backup: A directory where trained weights(checkpoints) will be stored as training progresses.

# In[ ]:


data_extention_file_path = os.path.join(cfg_dir, 'rsna.data')
with open(data_extention_file_path, 'w') as f:
    contents = """classes= 1
train  = {}
valid  = {}
names  = {}
backup = {}
    """.format(os.path.join(metadata_dir, "tr_list.txt"),
               os.path.join(metadata_dir, "val_list.txt"),
               os.path.join(cfg_dir, 'rsna.names'),
               backup_dir)
    f.write(contents)

# In[ ]:



# ### - cfg/rsna.names

# In[ ]:


# Label list of bounding box.

# ### - darknet53.conv.74  (Download Pre-trained Model)
# For training, we would download the pre-trained model weights(darknet53.conv.74) using following wget command. I recommend you to use this pre-trained weight too. Author of darknet also uses this pre-trained weights in different fields of image recognition.

# In[ ]:



# ### - cfg/rsna_yolov3.cfg_train
# * Basically, you can use darknet/cfg/yolov3.cfg files. However it won't work for RSNA. you need to edit for RSNA.
# * You can just download a cfg file I edited for RSNA with following wget command.
# 
# 
# * I refer to the following articles for editing cfg files.
#   * [YOLOv3 blog](https://pjreddie.com/darknet/yolo/)
#   * [YOLOv3 paper](https://pjreddie.com/media/files/papers/YOLOv3.pdf)
#   * [how to train yolov2 blog](https://medium.com/@manivannan_data/how-to-train-yolov2-to-detect-custom-objects-9010df784f36)
#   * [darknet github issues/236](https://github.com/pjreddie/darknet/issues/236)

# In[ ]:



# ## 4. Training YOLOv3

# ### 4.0. Command for training with Pre-trained CNN Weights (darknet53.conv.74)
# * I didn't run following command on kaggle kernel becuase of the long output.
# * If you crash with  'CUDA Error: out of memory', Solve it by Editing 'batch' and 'subdivisions' in 'cfg/rsna_yolov3.cfg_train'
# * If 'batch' and 'subdivisions' are 64 and 64 respectively, for every iteration only one image will be loaded on GPU memory. So it will use less GPU memory.

# In[ ]:


# !./darknet_gpu detector train cfg/rsna.data cfg/rsna_yolov3.cfg_train darknet53.conv.74 -i 0 | tee train_log.txt

# ### 4.1. Command for training with Multi-gpu after 1000 iteration
# 
# If you are trying to train with multi-gpu, there are three things to watch out.
# * (The # of gpus)x('learning rate' in 'cfg/rsna_yolov3.cfg_train') is the real learning rate for training
# * I don't recommend you to use multi-gpu for first 1000 iterations. with multi-gpu, training would not be stable. Use single gpu before 1000 and after 1000, continue with more gpus.
# * By the way, If the # of gpus is over 5, training is not stable.
# 
# ```
# Above things will depend on your environment. The best way to find the most appropriate method is to just give it a try :)
# ```

# In[ ]:


# !./darknet_gpu detector train cfg/rsna.data cfg/rsna_yolov3.cfg_train backup/rsna_yolov3_1000.weights -gpus 0,1,2,3 | tee train_log.txt

# ### 4.2. My Plot of Training Loss
# It's a loss graph up to about 2000 iteration. Since it tooks too long on kaggle kernel, I brought it. When learning, don't be surprised of big loss values at the beginning. Stay calm and It'll go down. Please See the following loss graph.

# In[ ]:



# In[ ]:


iters = []
losses = []
total_losses = []
with open("train_log.txt", 'r') as f:
    for i,line in enumerate(f):
        if "images" in line:
            iters.append(int(line.strip().split()[0].split(":")[0]))
            losses.append(float(line.strip().split()[2]))        
            total_losses.append(float(line.strip().split()[1].split(',')[0]))

plt.figure(figsize=(20, 5))
plt.subplot(1,2,1)
sns.lineplot(iters, total_losses, label="totla loss")
sns.lineplot(iters, losses, label="avg loss")
plt.xlabel("Iteration")
plt.ylabel("Loss")

plt.subplot(1,2,2)
sns.lineplot(iters, total_losses, label="totla loss")
sns.lineplot(iters, losses, label="avg loss")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.ylim([0, 4.05])

# ## 5. How to use trainined YOLOv3 for test images (command line)

# ### 5.0. Copy sample test image

# In[ ]:


ex_patient_id = annots[annots.Target == 1].patientId.values[2]
shutil.copy(ex_img_path, "test.jpg")
print(ex_patient_id)

# ### 5.1. Load trained model (at 15300 iteration)
# Since i uploaded the weights file (large big file) on my google drive, the command is very very long ...
# * It's a weight file at 15300 iteration, which I made submission file with. If you use this weight, you'll get a score of 0.141LB.
#   * Up to 15300 iteration, It takes about 8 hours.
#     * In .cfg file, I set 'batch' and 'subdivisions' as 64 and 8 respectively.
#     * Up to 1000 iteration from 0, it takes about 1h with **one** Tesla P100 GPU.      **(1000 iter/h)**
#     * Up to 15300 iteration from 1000, it takes about 7h with **four** Tesla P100 GPU. **(2043 iter/h)**

# In[ ]:



# In[ ]:



# ### 5.2. cfg file for test (not for training)

# In[ ]:



# In[ ]:



# In[ ]:


# ![](predictions.jpg)
plt.imshow(cv2.imread("./darknet/predictions.jpg"))

# ## 6. Generate Submission Files with YOLOv3 Python Wrapper

# ### 6.0. Download darknet python wrapper (darknet.py)
# * Basically, you can use darknet/python/darknet.py files. However it'll show error.
# * So, I edited the darknet.py. There are two main modifications.
#   * Change print statement to print function for python3
#   * Edit dynamic library('libdarknet.so') file path
# * I leaved '# ===' marks where i edited in darknet.py. For example,
# ```
# # ==============================================================================
# #lib = CDLL("/home/pjreddie/documents/darknet/libdarknet.so", RTLD_GLOBAL)
# darknet_lib_path = os.path.join(os.getcwd(), "darknet", "libdarknet.so")
# lib = CDLL(darknet_lib_path, RTLD_GLOBAL)
# # ==============================================================================
# ```

# In[ ]:



# ### 6.1. Load darknet python wrapper module

# In[ ]:


from darknet import *

# ### 6.2. Generate submission files
# * When making submission files, be aware of label format which is different in yolo.

# In[ ]:


threshold = 0.2

# In[ ]:


submit_file_path = "submission.csv"
cfg_path = os.path.join(cfg_dir, "rsna_yolov3.cfg_test")
weight_path = os.path.join(backup_dir, "rsna_yolov3_15300.weights")

test_img_list_path = os.path.join(metadata_dir, "te_list.txt")

# In[ ]:


gpu_index = 0
net = load_net(cfg_path.encode(),
               weight_path.encode(), 
               gpu_index)
meta = load_meta(data_extention_file_path.encode())

# In[ ]:


submit_dict = {"patientId": [], "PredictionString": []}

with open(test_img_list_path, "r") as test_img_list_f:
    # tqdm run up to 1000(The # of test set)
    for line in tqdm(test_img_list_f):
        patient_id = line.strip().split('/')[-1].strip().split('.')[0]

        infer_result = detect(net, meta, line.strip().encode(), thresh=threshold)

        submit_line = ""
        for e in infer_result:
            confi = e[1]
            w = e[2][2]
            h = e[2][3]
            x = e[2][0]-w/2
            y = e[2][1]-h/2
            submit_line += "{} {} {} {} {} ".format(confi, x, y, w, h)

        submit_dict["patientId"].append(patient_id)
        submit_dict["PredictionString"].append(submit_line)

pd.DataFrame(submit_dict).to_csv(submit_file_path, index=False)

# In[ ]:


# !ls -lsht

# In[ ]:



# ## 7. Future works & Etc
# 
# ### Future works (Things to try)
# * Image augmentation
# * More training
# * Utilizing the not labeled images because we got rid of not labeled images above
# 
# ### ETC
# * For a private matter, i can not proceed RSNA task after 09/27. If you have any ideas, questions and problems with this kernel after 09/27, Please leave those things anyway~! Collaborator '@John Byun' will reply to your comments.
