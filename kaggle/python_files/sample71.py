#!/usr/bin/env python
# coding: utf-8

# # Don't skip the analysis!!!
# ### Updated on 21 Apr
# 
# When it comes to image data, people tend to skip the data understanding step and goes straight into using powerful models for transfer learning or feature extractions. This is alright, but wouldn't you want to understand what goes on under the hood?
# 
# In this kernel, I attempt to understand the data in a conventional data analysis approach (on the raw image metadata etc) and highlight interesting observations which can help your pre-processing pipeline and feature engineering.
# 
# What is covered & what I plan to do over the next few weeks (or whenever I have time):
# * Image Shape **(done!)**
# * Understanding Target Variable **(done!)** **<---------------------------- NEW!!!!**
# * RGB Pixel Statistics **(done!)** **<--------------------------------------- NEW!!!!**
# * Edge Detection Pixel Statistics **(done!)** **<-------------------------- NEW!!!!**
# * Intensity Histogram
# * Correlations of the above with Target Variable
# 
# ### TL;DR - Summary of Insights
# 1. Super long images exist in the dataset
# 2. Highly imbalanced dataset, over 1000 labels, with 90% images having less than 5 labels
# 3. Similar target classes (e.g. men, women, portraits, human figures)
# 4. RGB pixel statistics - normal distribution with different mean

# ## Load the necessary packages and files

# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd 
import gc
import os
import PIL

from scipy import stats
from multiprocessing import Pool
from PIL import ImageOps, ImageFilter
from tqdm import tqdm
from wordcloud import WordCloud

tqdm.pandas()

# In[ ]:


df_train = pd.read_csv('../input/train-file-with-labels-and-meta-data/weird_images_w_labels.csv')
train_path = '../input/imet-2019-fgvc6/train/'
label_df = pd.read_csv('../input/imet-2019-fgvc6/labels.csv')

print('Files loaded!')

# ### Fun Fact : PIL.Image.open() is faster than plt.imread()
# 
# For image reading, PIL averages at 400 iter/second, while plt only managed around 100 iter/second.
# 
# Something that I just found out, but I'm not too sure why it is that way. Appreciate if someone could enlighten me. 
# 
# Update 1 : I just learnt how to use multiprocessing with pandas, now it is much faster!!
# Update 2 : I moved the meta data extraction and one-hot encoding of labels to another file to another kernel

# # Long tail distribution for image width and height
# 
# Looking at the KDE plot, you can see that there is a bump at ~5000 for width and ~7000 for height.
# 
# We must have some abnormally huge images in our dataset. (Or are they?)
# 
# ![](https://media1.tenor.com/images/39bac10b152f60e65329dfff93c7bf18/tenor.gif?itemid=4780502)

# In[ ]:


plt.figure(figsize=(14,6))
plt.subplot(121)
sns.distplot(df_train['width'],kde=False, label='Width')
sns.distplot(df_train['height'], kde=False, label='Height')
plt.legend()
plt.title('Image Dimension Histogram', fontsize=15)

plt.subplot(122)
sns.kdeplot(df_train['width'], label='Width')
sns.kdeplot(df_train['height'], label='Height')
plt.legend()
plt.title('Image Dimension KDE Plot', fontsize=15)

plt.tight_layout()
plt.show()

# ### Turns out they are only huge in either width or height
# 
# Something pretty long?

# In[ ]:


df_train[['width','height']].sort_values(by='width',ascending=False).head()

# In[ ]:


df_train[['width','height']].sort_values(by='height',ascending=False).head()

# ### Lets check out these weird images.

# In[ ]:


weird_height_id = [v for v in df_train.sort_values(by='height',ascending=False).head(20)['id'].values]
weird_width_id = [v for v in df_train.sort_values(by='width',ascending=False).head(20)['id'].values]

# In[ ]:


plt.figure(figsize=(12,10))

for num, img_id in enumerate(weird_height_id):
    img = PIL.Image.open(f'{train_path}{img_id}.png')
    plt.subplot(1,20,num + 1)
    plt.imshow(img)
    plt.axis('off')
    
plt.suptitle('Images with HUGE Height', fontsize=20)
plt.show()

# In[ ]:


plt.figure(figsize=(12,10))

for num, img_id in enumerate(weird_width_id):
    img = PIL.Image.open(f'{train_path}{img_id}.png')
    plt.subplot(20,1,num + 1)
    plt.imshow(img)
    plt.axis('off')
    
plt.suptitle('Images with HUGE Width', fontsize=20)
plt.show()

# ### Turns out they are all long artifacts/ accessories
# 
# During the pre-processing steps, these images should be padded instead of being resized to a square. 
# 
# Or perhaps a custom pre-processing steps can be applied to images of vastly different size like the weird images above? I will leave that to you to explore.

# In[ ]:


img = PIL.Image.open(f'{train_path}{weird_height_id[0]}.png')

w_resized = int(img.size[0] * 300 / img.size[1])
resized = img.resize((w_resized ,300))
pad_width = 300 - w_resized
padding = (pad_width // 2, 0, pad_width-(pad_width//2), 0)
resized_w_pad = ImageOps.expand(resized, padding)

resized_wo_pad = img.resize(size=(300,300))

# In[ ]:


plt.figure(figsize=(12,8))

plt.subplot(131)
plt.imshow(img)
plt.axis('off')
plt.title('Original Image',fontsize=15)

plt.subplot(132)
plt.imshow(resized_wo_pad)
plt.axis('off')
plt.title('A bent flat head screw?',fontsize=15)

plt.subplot(133)
plt.imshow(resized_w_pad)
plt.axis('off')
plt.title('Padded Image',fontsize=15)

plt.show()

# # Target Variable - Cultures and Tags
# 
# Before we move on to subsequent analysis, let's take a look at the target variable for this challenge. There is a total of **1103 categories**, of which **398 are cultures** and **705 are tags**.
# 
# * Cultures - The arts and other manifestations of human intellectual achievement regarded collectively. **Erm, pretty abstract**
# * Tags - A label attached to someone or something for the purpose of identification or to give other information. **Tags as in, you know, Facebook tags**
# 
# Let's visualize them with a word cloud!
# 
# *Note: The font sizes are arbitary and does not correlate to the category distribution.*

# In[ ]:


label_names = label_df['attribute_name'].values

num_labels = np.zeros((df_train.shape[0],))
train_labels = np.zeros((df_train.shape[0], len(label_names)))

for row_index, row in enumerate(df_train['attribute_ids']):
    num_labels[row_index] = len(row.split())    
    for label in row.split():
        train_labels[row_index, int(label)] = 1

# In[ ]:


culture, tag, unknown = 0, 0, 0

for l in label_names:
    if l[:3] == 'cul':
        culture += 1
    elif l[:3] == 'tag':
        tag += 1
    else:
        unknown += 1
        
print(f'Culture : {culture}')
print(f'Tag     : {tag}')
print(f'Unknown : {unknown}')
print(f'Total   : {culture + tag + unknown}')

# In[ ]:


label_sum = np.sum(train_labels, axis=0)

culture_sequence = label_sum[:398].argsort()[::-1]
tag_sequence = label_sum[398:].argsort()[::-1]

culture_labels = [label_names[x][9:] for x in culture_sequence]
culture_counts = [label_sum[x] for x in culture_sequence]

tag_labels = [label_names[x + 398][5:] for x in tag_sequence]
tag_counts = [label_sum[x + 398] for x in tag_sequence]

# In[ ]:


culture_labels_dict = dict((l,1) for l in culture_labels)
tag_labels_dict = dict((l,1) for l in tag_labels)

culture_labels_dict['<CULTURE>'] = 50
tag_labels_dict['<TAG>'] = 50

culture_cloud = WordCloud(background_color='Black', colormap='Paired', width=1600, height=800, random_state=123).generate_from_frequencies(culture_labels_dict)
tag_cloud = WordCloud(background_color='Black', colormap='Paired', width=1600, height=800, random_state=123).generate_from_frequencies(tag_labels_dict)

plt.figure(figsize=(24,24))
plt.subplot(211)
plt.imshow(culture_cloud,interpolation='bilinear')
plt.axis('off')

plt.subplot(212)
plt.imshow(tag_cloud, interpolation='bilinear')
plt.axis('off')

plt.tight_layout()
plt.show()

# ### Highly Imbalanced Data
# 
# In the plots below, only the top 20 cultures and tags with the highest counts were shown. You can see that the **20th place** for both label types only accounts for **0.72% (culture)** and **1.83% (tag)** of the entire dataset. In other words, majority of the labels are very rare.
# 
# **Culture** : Western cultures are the most common culture labels. This is of no surprise, considering the dataset origins from the Metropolitan Museum of Art in New York. Some of the cultures are subsets of another culture (e.g. London - subset of British, Paris - subset of French). There is even a culture label named 'Turkish or Venice', seems like even the subject matter experts had a hard time labelling these images! 
# 
# **Tag** : Men, women and flowers have been the most interesting since the beginning of time, thus their dominance on the tag labels. Similar to the culture, there seem to be a significant amount (considering we are only viewing 20 out of 705 tags) of tags with overlapping meaning (e.g. men, women, human figures, portraits, profiles). 
# 
# To tackle the challenge effectively, perhaps it is crucial to deal with the similar labels (or those with subset-superset relationship). At this point, I can think of 2 possible methods to deal with them:
# * Using word2vec to represent the label vocab (e.g. french), and check for its cosine similarity with the other labels
# * Using collocation matrix of the train set and see if any sets of labels often appear together (well, I doubt this will work, considering that many images only have 2-5 labels, which is shown in the subsequent plot)

# In[ ]:


plt.figure(figsize=(20,15))

plt.subplot(1,2,1)
ax1 = sns.barplot(y=culture_labels[:20], x=culture_counts[:20], orient="h")
plt.title('Label Counts by Culture (Top 20)',fontsize=15)
plt.xlim((0, max(culture_counts)*1.15))
plt.yticks(fontsize=15)

for p in ax1.patches:
    ax1.annotate(f'{int(p.get_width())}\n{p.get_width() * 100 / df_train.shape[0]:.2f}%',
                (p.get_width(), p.get_y() + p.get_height() / 2.), 
                ha='left', 
                va='center', 
                fontsize=12, 
                color='black',
                xytext=(7,0), 
                textcoords='offset points')

plt.subplot(1,2,2)    
ax2 = sns.barplot(y=tag_labels[:20], x=tag_counts[:20], orient="h")
plt.title('Label Counts by Tag (Top 20)',fontsize=15)
plt.xlim((0, max(tag_counts)*1.15))
plt.yticks(fontsize=15)

for p in ax2.patches:
    ax2.annotate(f'{int(p.get_width())}\n{p.get_width() * 100 / df_train.shape[0]:.2f}%',
                (p.get_width(), p.get_y() + p.get_height() / 2.), 
                ha='left', 
                va='center', 
                fontsize=12, 
                color='black',
                xytext=(7,0), 
                textcoords='offset points')

plt.tight_layout()
plt.show()

# ### Most Images only have 2 - 5 labels.
# 
# 90% of the images have 2 to 5 labels.
# 
# And there is an image with 11 labels.

# In[ ]:


plt.figure(figsize=(20,8))

ax = sns.countplot(num_labels)
plt.xlabel('Number of Labels')
plt.title('Number of Labels per Image', fontsize=20)

for p in ax.patches:
    ax.annotate(f'{p.get_height() * 100 / df_train.shape[0]:.3f}%',
            (p.get_x() + p.get_width() / 2., p.get_height()), 
            ha='center', 
            va='center', 
            fontsize=11, 
            color='black',
            xytext=(0,7), 
            textcoords='offset points')

# ### Mother of Images - 11 labels on a Single Image
# 
# That has to be some extraordinary confusing image. Lets take a look.
# 
# ![](https://my350z.com/forum/attachments/exterior-and-interior/435557d1501884683-bensopra-fenders-and-bodykit-thread-mother-of-god-meme_zpsa0e4dcb9.png)

# In[ ]:


weird_img_index = np.nonzero(num_labels == 11)[0][0]

img_w_11_labels_path = df_train.iloc[weird_img_index,0]
img_labels = df_train.iloc[weird_img_index,1]

img = PIL.Image.open(f'{train_path}{img_w_11_labels_path}.png')

print('LABELS OF IMAGE\n', *[label_names[int(l)] for l in sorted([int(l) for l in img_labels.split()])],sep='\n')

plt.figure(figsize=(20,6))
plt.imshow(img)
plt.axis('off')
plt.show()

# The image doesn't look as weird afterall!
# 
# It looks like a piece of fabric, with multiple types of objects and animals printed on it. 
# 
# Of the 11 labels, 9 of them belong to 'tags', due to the diversity of the printed patterns. 
# 
# Once again, it seems like '**culture**' constitutes something more abstract, like an art style (which can probably be explained by the use of colour, medium etc). On the other hand, '**tag**' is more straightforward, they literally means what you see from the image (shapes, patterns). 
# 
# As such, in terms of modelling, I'm guessing that tags can be more easily predicted by CNN based models (compared to culture). I shall do an analysis on the available public kernels to verify this in the future.

# # Onwards to Pixel Statistics!
# 
# The provided images comes in RGB channels, which stands for red, green and blue channel respectively. The pixel value ranges from 0 to 255. The mean and standard deviation for each colour channel is extracted and visualized.
# 
# ### Normal distributions, with a little skewness
# 
# Surprise surprise. I wasn't expecting any normal distributions for the pixel values per channel, let alone 3 normal distributions! Perhaps it is the norm for large enough dataset? Let me know what you think about this, I'm not sure if I'm missing any stuff here.

# In[ ]:


pal = ['red', 'green', 'blue']

plt.figure(figsize=(20,6))
plt.subplot(1,2,1)
sns.kdeplot(df_train['r_mean'], color=pal[0])
sns.kdeplot(df_train['g_mean'], color=pal[1])
sns.kdeplot(df_train['b_mean'], color=pal[2])
plt.ylabel('Density')
plt.xlabel('Mean Pixel Value')
plt.title('KDE Plot - Mean Pixel Value by Channel', fontsize=15)

plt.subplot(1,2,2)
sns.kdeplot(df_train['r_std'], color=pal[0])
sns.kdeplot(df_train['g_std'], color=pal[1])
sns.kdeplot(df_train['b_std'], color=pal[2])
plt.ylabel('Density')
plt.xlabel('Standard Deviation of Pixel Value')
plt.title('KDE Plot - Stdev Pixel Value by Channel', fontsize=15)

plt.show()

# ### Let's checkest the REDDEST, GREENEST and BLUEST image.

# In[ ]:


reddest = (df_train['r_mean'] - df_train['g_mean'] - df_train['b_mean'] + 255*2)
greenest = (df_train['g_mean'] - df_train['r_mean'] - df_train['b_mean'] + 255*2)
bluest = (df_train['b_mean'] - df_train['g_mean'] - df_train['r_mean'] + 255*2)

reddest_img_path = df_train.iloc[reddest.idxmax(),0]
greenest_img_path = df_train.iloc[greenest.idxmax(),0]
bluest_img_path = df_train.iloc[bluest.idxmax(),0]

# In[ ]:


reddest_im = PIL.Image.open(f'{train_path}{reddest_img_path}.png')
greenest_im = PIL.Image.open(f'{train_path}{greenest_img_path}.png')
bluest_im = PIL.Image.open(f'{train_path}{bluest_img_path}.png')

plt.figure(figsize=(20,6))
plt.subplot(131)
plt.imshow(reddest_im)
plt.axis('off')
plt.title('REDDEST glass-like object')

plt.subplot(132)
plt.imshow(greenest_im)
plt.axis('off')
plt.title('GREENEST piece of ancient writing')

plt.subplot(133)
plt.imshow(bluest_im)
plt.axis('off')
plt.title('BLUEST dark sky with some flags')

plt.show()

# Pretty interesting, but other than the green piece of ancient writing, the other 2 images are intensely red or blue simply because of their background colour. 
# 
# Don't think this is very useful for prediction, but we shall see.

# ### Edge Characteristics
# 
# In traditional vision domain (pre-CNN era), edge detection of an image is important in helping to identify the shapes or patterns. This can serve as an alternative image representation that we train our models on. But for now, let's check out if its statistics are of any use.
# 
# Some samples of the edge version of the images can be seen below. You can see the edges (in white) can approximate the outlines and distinct geometric features of the object in the image. Note that the edge quality can be further improved (for instance, do a gausian blur before applying edge detection, to remove noisy features).

# In[ ]:


plt.figure(figsize=(20,4))

random_image_paths = df_train['id'].sample(n=3, random_state=123).values

for index, path in enumerate(random_image_paths):
    im = PIL.Image.open(f'{train_path}{path}.png')
    plt.subplot(1,6, index*2 + 1)
    plt.imshow(im)
    plt.axis('off')
    plt.title('Original')

    plt.subplot(1,6, index*2 + 2)
    plt.imshow(im.filter(ImageFilter.FIND_EDGES))
    plt.axis('off')
    plt.title('Edge Only')

plt.show()

# The KDE plots show that the mean and standard deviations of edge pixel values in all 3 channels are almost the same. Thus we will only use a single channel to visualize the 3 sample images (lowest mean edge pixel value, median and highest).

# In[ ]:


pal = ['red', 'green', 'blue']

plt.figure(figsize=(20,6))
plt.subplot(1,2,1)
sns.kdeplot(df_train['r_edge_mean'], color=pal[0])
sns.kdeplot(df_train['g_edge_mean'], color=pal[1])
sns.kdeplot(df_train['b_edge_mean'], color=pal[2])
plt.ylabel('Density')
plt.xlabel('Mean Pixel Value')
plt.title('KDE Plot - Mean Pixel Value (Edge) by Channel', fontsize=15)

plt.subplot(1,2,2)
sns.kdeplot(df_train['r_edge_std'], color=pal[0])
sns.kdeplot(df_train['g_edge_std'], color=pal[1])
sns.kdeplot(df_train['b_edge_std'], color=pal[2])
plt.ylabel('Density')
plt.xlabel('Standard Deviation of Pixel Value')
plt.title('KDE Plot - Stdev Pixel Value (Edge) by Channel', fontsize=15)

plt.show()

# In[ ]:


edge_min, edge_median, edge_max = df_train['r_edge_mean'].min(), df_train['r_edge_mean'].median(), df_train['r_edge_mean'].max()

low_mean_edge = df_train.loc[df_train['r_edge_mean'] == edge_min, 'id'].values[0]
med_mean_edge = df_train.loc[df_train['r_edge_mean'] == edge_median, 'id'].values[0]
high_mean_edge = df_train.loc[df_train['r_edge_mean'] == edge_max, 'id'].values[0]

low_mean_edge_im = PIL.Image.open(f'{train_path}{low_mean_edge}.png')
med_mean_edge_im = PIL.Image.open(f'{train_path}{med_mean_edge}.png')
high_mean_edge_im = PIL.Image.open(f'{train_path}{high_mean_edge}.png')

plt.figure(figsize=(20,16))
plt.subplot(231)
plt.imshow(low_mean_edge_im)
plt.axis('off')
plt.title(f'Mean Edge = {edge_min:.2f} (Raw)')

plt.subplot(232)
plt.imshow(med_mean_edge_im)
plt.axis('off')
plt.title(f'Mean Edge = {edge_median:.2f} (Raw)')

plt.subplot(233)
plt.imshow(high_mean_edge_im)
plt.axis('off')
plt.title(f'Mean Edge = {edge_max:.2f} (Raw)')

plt.subplot(234)
plt.imshow(low_mean_edge_im.filter(ImageFilter.FIND_EDGES))
plt.axis('off')
plt.title(f'Mean Edge = {edge_min:.2f} (Edge)')

plt.subplot(235)
plt.imshow(med_mean_edge_im.filter(ImageFilter.FIND_EDGES))
plt.axis('off')
plt.title(f'Mean Edge = {edge_median:.2f} (Edge)')

plt.subplot(236)
plt.imshow(high_mean_edge_im.filter(ImageFilter.FIND_EDGES))
plt.axis('off')
plt.title(f'Mean Edge = {edge_max:.2f} (Edge)')

plt.show()

# ### Spooky!
# 
# The image with the highest mean edge pixel value (right most) seems like some sort of optical illusion.

# # To be continued!
# 
# ### Do give me an upvote if you find it insightful/ interesting! :)

# In[ ]:



