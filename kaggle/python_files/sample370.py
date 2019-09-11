#!/usr/bin/env python
# coding: utf-8

# # Exploratory data analysis of the human protein atlas image dataset
# update 5/10/2018: beginning of cell segmentation algorithm
# 
# update 5/10/2018: add red + blue channels stack and whole cell identification (does not give a clean result, though)
# 
# This kernel is just the beginning of a work in progress and will be updated very often.
# We will explore the dataset available for the human protein atlas image competition. Questions we would like to answer include:
# * what channels of the image contain the relevant information
# * how much can we reduce dimensionality of data while retaining important information

# In[ ]:


#import modules

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from PIL import Image
from collections import Counter

import os
print(os.listdir("../input"))

# ## What's in the data?
#  Let's import the *train.csv* data files to see what they contain. We also define a dictionary containing the map between labels of the training data (the column *target* in *train.csv*) and their biological meaning.

# In[ ]:


#import training data
train = pd.read_csv("../input/train.csv")
print(train.head())

#map of targets in a dictionary
subcell_locs = {
0:  "Nucleoplasm", 
1:  "Nuclear membrane",   
2:  "Nucleoli",   
3:  "Nucleoli fibrillar center" ,  
4:  "Nuclear speckles",
5:  "Nuclear bodies",
6:  "Endoplasmic reticulum",   
7:  "Golgi apparatus",
8:  "Peroxisomes",
9:  "Endosomes",
10:  "Lysosomes",
11:  "Intermediate filaments",   
12:  "Actin filaments",
13:  "Focal adhesion sites",   
14:  "Microtubules",
15:  "Microtubule ends",   
16:  "Cytokinetic bridge",   
17:  "Mitotic spindle",
18:  "Microtubule organizing center",  
19:  "Centrosome",
20:  "Lipid droplets",   
21:  "Plasma membrane",   
22:  "Cell junctions", 
23:  "Mitochondria",
24:  "Aggresome",
25:  "Cytosol",
26:  "Cytoplasmic bodies",   
27:  "Rods & rings" 
}

# Each image is a 4-channel image with the protein of interest in the green channel. It is the subcellular localization of this protein which is recorded in the *Target* column of the *train.csv* file. The red channel corresponds to microtubules, the blue channel to the nucleus and the yellow channel to the endoplasmid reticulum. Let's display the different channels of the image with ID == 1, since it contains several subcelullar locations for our protein of interest. Then we will overlay the green and yellow channel, as the yellow channel gives a good indication of the cell shape.

# In[ ]:


print("The image with ID == 1 has the following labels:", train.loc[1, "Target"])
print("These labels correspond to:")
for location in train.loc[1, "Target"].split():
    print("-", subcell_locs[int(location)])

#reset seaborn style
sns.reset_orig()

#get image id
im_id = train.loc[1, "Id"]

#create custom color maps
cdict1 = {'red':   ((0.0,  0.0, 0.0),
                   (1.0,  0.0, 0.0)),

         'green': ((0.0,  0.0, 0.0),
                   (0.75, 1.0, 1.0),
                   (1.0,  1.0, 1.0)),

         'blue':  ((0.0,  0.0, 0.0),
                   (1.0,  0.0, 0.0))}

cdict2 = {'red':   ((0.0,  0.0, 0.0),
                   (0.75, 1.0, 1.0),
                   (1.0,  1.0, 1.0)),

         'green': ((0.0,  0.0, 0.0),
                   (1.0,  0.0, 0.0)),

         'blue':  ((0.0,  0.0, 0.0),
                   (1.0,  0.0, 0.0))}

cdict3 = {'red':   ((0.0,  0.0, 0.0),
                   (1.0,  0.0, 0.0)),

         'green': ((0.0,  0.0, 0.0),
                   (1.0,  0.0, 0.0)),

         'blue':  ((0.0,  0.0, 0.0),
                   (0.75, 1.0, 1.0),
                   (1.0,  1.0, 1.0))}

cdict4 = {'red': ((0.0,  0.0, 0.0),
                   (0.75, 1.0, 1.0),
                   (1.0,  1.0, 1.0)),

         'green': ((0.0,  0.0, 0.0),
                   (0.75, 1.0, 1.0),
                   (1.0,  1.0, 1.0)),

         'blue':  ((0.0,  0.0, 0.0),
                   (1.0,  0.0, 0.0))}

plt.register_cmap(name='greens', data=cdict1)
plt.register_cmap(name='reds', data=cdict2)
plt.register_cmap(name='blues', data=cdict3)
plt.register_cmap(name='yellows', data=cdict4)

#get each image channel as a greyscale image (second argument 0 in imread)
green = cv2.imread('../input/train/{}_green.png'.format(im_id), 0)
red = cv2.imread('../input/train/{}_red.png'.format(im_id), 0)
blue = cv2.imread('../input/train/{}_blue.png'.format(im_id), 0)
yellow = cv2.imread('../input/train/{}_yellow.png'.format(im_id), 0)

#display each channel separately
fig, ax = plt.subplots(nrows = 2, ncols=2, figsize=(15, 15))
ax[0, 0].imshow(green, cmap="greens")
ax[0, 0].set_title("Protein of interest", fontsize=18)
ax[0, 1].imshow(red, cmap="reds")
ax[0, 1].set_title("Microtubules", fontsize=18)
ax[1, 0].imshow(blue, cmap="blues")
ax[1, 0].set_title("Nucleus", fontsize=18)
ax[1, 1].imshow(yellow, cmap="yellows")
ax[1, 1].set_title("Endoplasmic reticulum", fontsize=18)
for i in range(2):
    for j in range(2):
        ax[i, j].set_xticklabels([])
        ax[i, j].set_yticklabels([])
        ax[i, j].tick_params(left=False, bottom=False)
plt.show()

# In[ ]:


#stack nucleus and microtubules images
#create blue nucleus and red microtubule images
nuclei = cv2.merge((np.zeros((512, 512),dtype='uint8'), np.zeros((512, 512),dtype='uint8'), blue))
microtub = cv2.merge((red, np.zeros((512, 512),dtype='uint8'), np.zeros((512, 512),dtype='uint8')))

#create ROI
rows, cols, _ = nuclei.shape
roi = microtub[:rows, :cols]

#create a mask of nuclei and invert mask
nuclei_grey = cv2.cvtColor(nuclei, cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(nuclei_grey, 10, 255, cv2.THRESH_BINARY)
mask_inv = cv2.bitwise_not(mask)

#make area of nuclei in ROI black
red_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

#select only region with nuclei from blue
blue_fg = cv2.bitwise_and(nuclei, nuclei, mask=mask)

#put nuclei in ROI and modify red
dst = cv2.add(red_bg, blue_fg)
microtub[:rows, :cols] = dst

#show result image
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(microtub)
ax.set_title("Nuclei (blue) + microtubules (red)", fontsize=15)
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.tick_params(left=False, bottom=False)

# Let's see how the targets are distributed.

# In[ ]:


labels_num = [value.split() for value in train['Target']]
labels_num_flat = list(map(int, [item for sublist in labels_num for item in sublist]))
labels = ["" for _ in range(len(labels_num_flat))]
for i in range(len(labels_num_flat)):
    labels[i] = subcell_locs[labels_num_flat[i]]

fig, ax = plt.subplots(figsize=(15, 5))
pd.Series(labels).value_counts().plot('bar', fontsize=14)


# According to [Chen *et al*. 2007](https://academic.oup.com/bioinformatics/article-lookup/doi/10.1093/bioinformatics/btm206), if images are segmented into single cell regions, additional features that are not appropriate for whole fields can be calculated after *seeded watershed segmentation*. Nucleus images provide a means to identify each cell, so image segmentation may start by identification of nuclei in images.  The function `cv2.connectedComponents` provides a simple and effective means to label nuclei in images. Conversely, as shown on the following notebook cell, identification of whole cells using `cv2.connectedComponents` is not as efficient, due to the less homogeneous signal in the yellow channel of the image.

# In[ ]:


#apply threshold on the nucleus image
ret, thresh = cv2.threshold(blue, 0, 255, cv2.THRESH_BINARY)
#display threshold image
fig, ax = plt.subplots(ncols=3, figsize=(20, 20))
ax[0].imshow(thresh, cmap="Greys")
ax[0].set_title("Threshold", fontsize=15)
ax[0].set_xticklabels([])
ax[0].set_yticklabels([])
ax[0].tick_params(left=False, bottom=False)

#morphological opening to remove noise
kernel = np.ones((5,5),np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
ax[1].imshow(opening, cmap="Greys")
ax[1].set_title("Morphological opening", fontsize=15)
ax[1].set_xticklabels([])
ax[1].set_yticklabels([])
ax[1].tick_params(left=False, bottom=False)

# Marker labelling
ret, markers = cv2.connectedComponents(opening)
# Map component labels to hue val
label_hue = np.uint8(179 * markers / np.max(markers))
blank_ch = 255 * np.ones_like(label_hue)
labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
# cvt to BGR for display
labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
# set bg label to black
labeled_img[label_hue==0] = 0
ax[2].imshow(labeled_img)
ax[2].set_title("Markers", fontsize=15)
ax[2].set_xticklabels([])
ax[2].set_yticklabels([])
ax[2].tick_params(left=False, bottom=False)


# In[ ]:


#apply threshold on the endoplasmic reticulum image
ret, thresh = cv2.threshold(yellow, 4, 255, cv2.THRESH_BINARY)
#display threshold image
fig, ax = plt.subplots(ncols=4, figsize=(20, 20))
ax[0].imshow(thresh, cmap="Greys")
ax[0].set_title("Threshold", fontsize=15)
ax[0].set_xticklabels([])
ax[0].set_yticklabels([])
ax[0].tick_params(left=False, bottom=False)

#morphological opening to remove noise
kernel = np.ones((5,5),np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
ax[1].imshow(opening, cmap="Greys")
ax[1].set_title("Morphological opening", fontsize=15)
ax[1].set_xticklabels([])
ax[1].set_yticklabels([])
ax[1].tick_params(left=False, bottom=False)

#morphological closing
closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
ax[2].imshow(closing, cmap="Greys")
ax[2].set_title("Morphological closing", fontsize=15)
ax[2].set_xticklabels([])
ax[2].set_yticklabels([])
ax[2].tick_params(left=False, bottom=False)

# Marker labelling
ret, markers = cv2.connectedComponents(closing)
# Map component labels to hue val
label_hue = np.uint8(179 * markers / np.max(markers))
blank_ch = 255 * np.ones_like(label_hue)
labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
# cvt to BGR for display
labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
# set bg label to black
labeled_img[label_hue==0] = 0
ax[3].imshow(labeled_img)
ax[3].set_title("Markers", fontsize=15)
ax[3].set_xticklabels([])
ax[3].set_yticklabels([])
ax[3].tick_params(left=False, bottom=False)

# Let's try different simple thresholding methods. Description of threshold types can be found [here](https://docs.opencv.org/3.4/d7/d4d/tutorial_py_thresholding.html) and [here](https://docs.opencv.org/3.4/d7/d1b/group__imgproc__misc.html#gaa9e58d2860d4afa658ef70a9b1115576).

# In[ ]:


#apply threshold on the endoplasmic reticulum image
ret, thresh1 = cv2.threshold(yellow, 4, 255, cv2.THRESH_BINARY)
ret, thresh2 = cv2.threshold(yellow, 4, 255, cv2.THRESH_TRUNC)
ret, thresh3 = cv2.threshold(yellow, 4, 255, cv2.THRESH_TOZERO)

#display threshold images
fig, ax = plt.subplots(ncols=3, figsize=(20, 20))
ax[0].imshow(thresh1, cmap="Greys")
ax[0].set_title("Binary", fontsize=15)

ax[1].imshow(thresh2, cmap="Greys")
ax[1].set_title("Trunc", fontsize=15)

ax[2].imshow(thresh3, cmap="Greys")
ax[2].set_title("To zero", fontsize=15)

# *To zero* simple thresholding is not adapted at all for identifying cell boundaries based on the yellow channel. Even after playing with the upper and lower parameter values, no satisfactory result is obtained. *Binary* and *truncate* methods work better. Let's see how *connectedComponents* work after both thresholding methods.

# In[ ]:


fig, ax = plt.subplots(ncols=4, figsize=(20, 20))

#morphological opening to remove noise after binary thresholding
kernel = np.ones((5,5),np.uint8)
opening1 = cv2.morphologyEx(thresh1, cv2.MORPH_OPEN, kernel)
ax[0].imshow(opening1, cmap="Greys")
ax[0].set_title("Morphological opening (binary)", fontsize=15)
ax[0].set_xticklabels([])
ax[0].set_yticklabels([])
ax[0].tick_params(left=False, bottom=False)

#morphological closing after binary thresholding
closing1 = cv2.morphologyEx(opening1, cv2.MORPH_CLOSE, kernel)
ax[1].imshow(closing1, cmap="Greys")
ax[1].set_title("Morphological closing (binary)", fontsize=15)
ax[1].set_xticklabels([])
ax[1].set_yticklabels([])
ax[1].tick_params(left=False, bottom=False)

#morphological opening to remove noise after truncate thresholding
kernel = np.ones((5,5),np.uint8)
opening2 = cv2.morphologyEx(thresh2, cv2.MORPH_OPEN, kernel)
ax[2].imshow(opening2, cmap="Greys")
ax[2].set_title("Morphological opening (truncate)", fontsize=15)
ax[2].set_xticklabels([])
ax[2].set_yticklabels([])
ax[2].tick_params(left=False, bottom=False)

#morphological closing after truncate thresholding
closing2 = cv2.morphologyEx(opening2, cv2.MORPH_CLOSE, kernel)
ax[3].imshow(closing2, cmap="Greys")
ax[3].set_title("Morphological closing (truncate)", fontsize=15)
ax[3].set_xticklabels([])
ax[3].set_yticklabels([])
ax[3].tick_params(left=False, bottom=False)

fig, ax = plt.subplots(ncols=2, figsize=(10, 10))
# Marker labelling for binary thresholding
ret, markers1 = cv2.connectedComponents(closing1)
# Map component labels to hue val
label_hue1 = np.uint8(179 * markers1 / np.max(markers1))
blank_ch1 = 255 * np.ones_like(label_hue1)
labeled_img1 = cv2.merge([label_hue1, blank_ch1, blank_ch1])
# cvt to BGR for display
labeled_img1 = cv2.cvtColor(labeled_img1, cv2.COLOR_HSV2BGR)
# set bg label to black
labeled_img1[label_hue1==0] = 0
ax[0].imshow(labeled_img1)
ax[0].set_title("Markers (binary)", fontsize=15)
ax[0].set_xticklabels([])
ax[0].set_yticklabels([])
ax[0].tick_params(left=False, bottom=False)

# Marker labelling for truncate thresholding
ret, markers2 = cv2.connectedComponents(closing2)
# Map component labels to hue val
label_hue2 = np.uint8(179 * markers2 / np.max(markers2))
blank_ch2 = 255 * np.ones_like(label_hue2)
labeled_img2 = cv2.merge([label_hue2, blank_ch2, blank_ch2])
# cvt to BGR for display
labeled_img2 = cv2.cvtColor(labeled_img2, cv2.COLOR_HSV2BGR)
# set bg label to black
labeled_img2[label_hue2==0] = 0
ax[1].imshow(labeled_img2)
ax[1].set_title("Markers (truncate)", fontsize=15)
ax[1].set_xticklabels([])
ax[1].set_yticklabels([])
ax[1].tick_params(left=False, bottom=False)

# At this point it's not clear if truncate thresholding is an improvement compared to binary thresholding. Some cells are fused to each other while they should not be.
# 
# On the other hand. Adaptive thresholding methods apply a different threshold on different parts of the image, let's see how well it does on our images. See [here](https://docs.opencv.org/3.4/d7/d1b/group__imgproc__misc.html#gaa42a3e6ef26247da787bf34030ed772c) for more explanations.

# In[ ]:


#apply adaptive threshold on endoplasmic reticulum image
y_blur = cv2.medianBlur(yellow, 3)

#apply adaptive thresholding
ret,th1 = cv2.threshold(y_blur, 5,255, cv2.THRESH_BINARY)

th2 = cv2.adaptiveThreshold(y_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 3)

th3 = cv2.adaptiveThreshold(y_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 3)

#display threshold images
fig, ax = plt.subplots(ncols=3, figsize=(20, 20))
ax[0].imshow(th1, cmap="Greys")
ax[0].set_title("Binary", fontsize=15)

ax[1].imshow(th2, cmap="Greys_r")
ax[1].set_title("Adaptive: mean", fontsize=15)

ax[2].imshow(th3, cmap="Greys_r")
ax[2].set_title("Adaptive: gaussian", fontsize=15)

# In[ ]:



