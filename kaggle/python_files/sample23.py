#!/usr/bin/env python
# coding: utf-8

# # Who can save the rainforest?
# 
# In this competition we are given a **multilabel classification** problem, where we have to decide, given an image, which labels belong to it. From the evaluation section of the competition:
# 
# For each image listed in the test set, predict a space-delimited list of tags which you believe are associated with the image. There are 17 possible tags: agriculture, artisinal_mine, bare_ground, blooming, blow_down, clear, cloudy, conventional_mine, cultivation, habitation, haze, partly_cloudy, primary, road, selective_logging, slash_burn, water.
# 
# In this notebook we will:
# 
# * generate a fun bernoulli trial sample submission
# * look at the actual images to get a first impression of the data.
# 
# A standard approach to multilabel classification is to learn as many OVA (one vs all) models as there are distinct labels and then assign labels by the classifier output of each of the models, we'll get to that later.
# 
# **If you like it, please upvote this :)**
# 
# Let's dive right into the data.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from subprocess import check_output
import matplotlib.pyplot as plt
from scipy.stats import bernoulli
import seaborn as sns
print(check_output(["ls", "../input"]).decode("utf8"))
#print(check_output(["ls", "../input/train-jpg"]).decode("utf8"))

# In[ ]:


sample = pd.read_csv('../input/sample_submission.csv')
sample.head()

# In[ ]:


df = pd.read_csv('../input/train.csv')
df.head()

# So, we are given around 40.000 training images.

# In[ ]:


df.shape

# # Tag counts
# 
# First, let's count all of the tags.

# In[ ]:


all_tags = [item for sublist in list(df['tags'].apply(lambda row: row.split(" ")).values) for item in sublist]
print('total of {} non-unique tags in all training images'.format(len(all_tags)))
print('average number of labels per image {}'.format(1.0*len(all_tags)/df.shape[0]))

# Now, lets do the actual counting. We're going to use pandas dataframe groupby method for that. In total, as we found in the description above, there are 17 distinct tags.

# In[ ]:


tags_counted_and_sorted = pd.DataFrame({'tag': all_tags}).groupby('tag').size().reset_index().sort_values(0, ascending=False)
tags_counted_and_sorted.head()

# There are only a few tags, that occur very often in the data:
# 
# * primary
# * clear
# * agriculture
# * road
# * and water.

# In[ ]:


tags_counted_and_sorted.plot.barh(x='tag', y=0, figsize=(12,8))

# From this tag distribution it will most likely be relatively easy to predict the often occuring tags and comparatively very hard to get the low sampled tags correct.

# # Submission from training tag counts
# 
# Let's do something fun. We'll take the training tag distribution and sample from it as a prior for our test data. For that we will configure a bernoulli distribution for each sample with the observed training frequency and sample from that for each test image. With that we'll generate a submission without ever looking at the images. :)

# In[ ]:


tag_probas = tags_counted_and_sorted[0].values/tags_counted_and_sorted[0].values.sum()
indicators = np.hstack([bernoulli.rvs(p, 0, sample.shape[0]).reshape(sample.shape[0], 1) for p in tag_probas])
indicators = np.array(indicators)
indicators.shape

# In[ ]:


indicators[:10,:]

# In[ ]:


sorted_tags = tags_counted_and_sorted['tag'].values
all_test_tags = []
for index in range(indicators.shape[0]):
    all_test_tags.append(' '.join(list(sorted_tags[np.where(indicators[index, :] == 1)[0]])))
len(all_test_tags)

# In[ ]:


sample['tags'] = all_test_tags
sample.head()
sample.to_csv('bernoulli_submission.csv', index=False)

# Ok, enough for the fun part, lets get serious :).

# # Looking at the actual images

# In[ ]:


from glob import glob
image_paths = glob('../input/train-jpg/*.jpg')[0:1000]
image_names = list(map(lambda row: row.split("/")[-1][:-4], image_paths))
image_names[0:10]

# In[ ]:


plt.figure(figsize=(12,8))
for i in range(6):
    plt.subplot(2,3,i+1)
    plt.imshow(plt.imread(image_paths[i]))
    plt.title(str(df[df.image_name == image_names[i]].tags.values))

# It seems, that all of the images are of the same size, which would make preprocessing them much easier.

# # Image clustering
# 
# Without having to look at all of the images, a common technique is to cluster images by their native representation (pixel intensities) or some encoded version of it, e.g. by computing activations of a vision-based neural network.
# 
# For our purpose we will just use the pixel intensities and compute pairwise distances.

# In[ ]:


import cv2

n_imgs = 600

all_imgs = []

for i in range(n_imgs):
    img = plt.imread(image_paths[i])
    img = cv2.resize(img, (50, 50), cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype('float') / 255.0
    img = img.reshape(1, -1)
    all_imgs.append(img)

img_mat = np.vstack(all_imgs)
img_mat.shape

# We can see frmo the line spectrum in the clustermap, that there are a few images that are very similar to each other using the pixel intensities.

# In[ ]:


from scipy.spatial.distance import pdist, squareform

sq_dists = squareform(pdist(img_mat))
print(sq_dists.shape)
sns.clustermap(
    sq_dists,
    figsize=(12,12),
    cmap=plt.get_cmap('viridis')
)

# 

# In[ ]:



