#!/usr/bin/env python
# coding: utf-8

# Let's explore the various data files and get a feel for the images and the data. I'll use OpenCV to read the images and hvplot for the charts. Hvplot is cool in that it's similar to Pandas plotting but with a Bokeh backend to make interactive charts.
# 
# 
# ### Files
# There are several files provided here. The Data tab for the competition provides a nice overview of each file you see below.  Note that the train images and the image urls aren't part of the kernel files. The Data tab has more information on how to get them.

# In[ ]:


import os
from glob import glob
import numpy as np
import pandas as pd
from bq_helper import BigQueryHelper
from dask import bag, diagnostics 
from urllib import request
import cv2
import missingno as msno
import hvplot.pandas  # custom install
from matplotlib import pyplot as plt

files = glob(os.path.join('../input', '*.csv'))
for f in files:
    df = pd.read_csv(f, nrows=5)
    display(f, df) 

# ### Images
# Let's use the data to annotate some train images and see what we're trying to do. You'll see there is some complexity here, both with the labels and the boxes.

# In[ ]:


train = pd.read_csv('../input/train_human_labels.csv', usecols=['ImageID', 'LabelName'])

descrips =  pd.read_csv('../input/class-descriptions.csv', names=['LabelName', 'Description'])
train = train.merge(descrips, how='left', on='LabelName')
train.head(9)

open_images = BigQueryHelper(active_project="bigquery-public-data", dataset_name="open_images")
query = """
            SELECT image_id, original_url 
            FROM `bigquery-public-data.open_images.images` 
            WHERE image_id IN UNNEST(['0199bc3e1db115d0',
                                      '4fa8054781a4c382',
                                      '51c5d8d5d9cd87ca',
                                      '9ec02b5c0315fcd1',
                                      'b37f763ae67d0888',
                                      'ddcb4b7478e9917b'])
        """
urls = open_images.query_to_pandas_safe(query)

boxes = pd.read_csv('../input/train_bounding_boxes.csv')
boxes = boxes[boxes.ImageID.isin(urls.image_id.tolist())].sort_values('ImageID')

# In[ ]:


imlist = urls.image_id.tolist()
files = ['../input/openimages-support/ims/{}.jpg'.format(i) for i in imlist]
fig, ax = plt.subplots() 
fig.set_size_inches((15,15))
ax.set_axis_off()
for n, (file, image) in enumerate(zip(files, imlist)):
    a = fig.add_subplot(2, 3, n + 1)
    req = request.urlopen(urls.original_url[n])
    arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
    img = cv2.imdecode(arr, 1) 
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h0, w0 = img.shape[:2]
    scale = 1024/max(h0, w0)
    h, w = int(round(h0*scale)), int(round(w0*scale))
    img = cv2.resize(img, (h, w), interpolation = cv2.INTER_AREA)
    imboxes = boxes[boxes.ImageID == image]
    for idx,row in imboxes.iterrows():
        img = cv2.rectangle(img, (int(row.XMin*h), int(row.YMin*w)),\
                    (int(row.XMax*h), int(row.YMax*w)), (0,255,0), 4)
    label = train.loc[train.ImageID == image, 'Description'].str.cat(sep = ', ')
    plt.title(label, fontsize=10)
    plt.axis('off')
    plt.imshow(img)

# Looking at these images we can see that getting all of the right labels might be quite challenging! Here's part of a graphic from the Open Images website showing the hierarchy of labels for the bounding box set (which is only 600 labels). You can find the full graphic of labels and a corresponding json file on the site.
# 
# ![Circle](https://storage.googleapis.com/openimages/web/images/v2-bbox_labels_vis_screenshot.png)
# 
# 

# 
# The images are clearly of different dimensions. We can use the test images to explore since they're already in the container. I'll use Dask to parallelize the operation and speed things up. 
# 
# There are several different image sizes with a max of 1024 pixels for each dimension, as mentioned by the host. 

# In[ ]:


# get image dimensions
def get_dims(file):
    img = cv2.imread(file)
    h,w = img.shape[:2]
    return h,w

# parallelize
filepath = '../input/stage_1_test_images/'
filelist = [filepath + f for f in os.listdir(filepath)]
dimsbag = bag.from_sequence(filelist).map(get_dims)
with diagnostics.ProgressBar():
    dims = dimsbag.compute()
    
dim_df = pd.DataFrame(dims, columns=['height', 'width'])
sizes = dim_df.groupby(['height', 'width']).size().reset_index().rename(columns={0:'count'})
sizes.hvplot.scatter(x='height', y='width', size='count', xlim=(0,1200), ylim=(0,1200), grid=True, xticks=2, 
        yticks=2, height=500, width=600).options(scaling_factor=0.1, line_alpha=1, fill_alpha=0)

# ### Labels
# Let's end with a deeper look at labels. I'll use the human labels to get an idea of how many images and labels we have. Note that even though these are images from the Bounding Box set in Open Images, the labels in our scope are far more diverse.

# In[ ]:


print('{} images with {} unique labels'.format(train.ImageID.nunique(), train.LabelName.nunique()))
train.head(9)

# Here's a look at label frequencies for the top 48 labels. The top 3 are Person, Clothing, and Human Face. There's a long tail on the distribution, meaning that most categories are infrequent. Note that these are the translated labels, not the coded labels, and might not match 1-1.

# In[ ]:


### not usually necessary, but helps with flaky notebook plotting
import holoviews as hv
hv.extension('bokeh')
###

dcounts = train.Description.value_counts(normalize=True)
dcounts_df = pd.DataFrame({'label': dcounts.index.tolist(), 'pct_of_images': dcounts})
dcounts_df.reset_index(drop=True, inplace=True)
dcounts_df[0:48].hvplot.bar(x='label', y='pct_of_images', invert=True, flip_yaxis=True, 
                            height=600, width=600, ylim=(0,0.12))

# Let's take a closer look at how many labels are in each image. It looks like some images have a lot of labels! Most have ten or fewer though as seen in the histogram. Note, you can zoom in on the histogram to get a closer look at specific ranges.

# In[ ]:


images = train.groupby('ImageID').count()
images.columns = ['LabelCount', 'DescriptionCount']
display(images.sort_values('LabelCount', ascending=False).head(10))
images.hvplot.hist('LabelCount', bins=50, height=400, width=600)

# Finally, to look at correlation, we can use hierarchical clustering to see how often labels appear together. I'm using a shortcut here so these results may only be approximate. Here are the top 48 most frequent labels.

# In[ ]:


trainmain = train[train.Description.isin(dcounts_df.loc[0:48, 'label'])]
trainpiv = trainmain.pivot_table(index='ImageID', columns='Description', aggfunc='size')
msno.dendrogram(trainpiv, inline=True)

# That's all for now, good luck!
