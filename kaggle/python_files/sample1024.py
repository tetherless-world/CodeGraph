#!/usr/bin/env python
# coding: utf-8

# 
# # Ideas for Generating Image Features and Measuring Image Quality
# 
# <br>
# 
# ![](https://i.imgur.com/84TEdoa.png)
# 
# <br>
# 
# [Avito](https://www.kaggle.com/c/avito-demand-prediction) is Russia's largest Advertisment firm. The quality of the advertisement image significantly affects the demand volume on an item. For both advertisers and Avito, it is important to use authentic high quality images. In this kernel, I have implemented some ideas which can be used to create new features related to images. These features are an indicatory factors about the Image Quality. Following is the list of feature ideas:  
# 
# 
# ### 1. Dullness : Is the Image Very Dull ?   
#     
#    1.1 Image Dullness Score
# 
# ### 2. Whiteness : Is the Image Very White ?  
#    2.1 Image Whiteness Score  
#     
# ### 3. Uniformity : Is the Image too Uniform ?
#    3.1 Average Pixel Width
# 
# ### 4. Colors : What are the top colors used in the Image ? 
#    4.1 Dominant Color of the Image   
#    4.2 Average Color of the Image
# 
# ### 5. Dimensions : Is the Image too Large or too Small ?  
#    5.1 Width of the Image    
#    5.2 Height of the Image   
#    5.3 Size of the Image    
# 
# ### 6. Blurrness : Is the Image Too Blurry ?   
#    6.1 Width of the Image      
# 
# <br>
# 

# In[1]:


from collections import defaultdict
from scipy.stats import itemfreq
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
from skimage import feature
from PIL import Image as IMG
import numpy as np
import pandas as pd 
import operator
import cv2
import os 

from IPython.core.display import HTML 
from IPython.display import Image

images_path = '../input/sampleavitoimages/sample_avito_images/'
imgs = os.listdir(images_path)

features = pd.DataFrame()
features['image'] = imgs

# ## 1. Is the image Very Dull 
# 
# ### Feature 1 : Dullness
# 
# Dull Images may not be good for the advirtisment purposes. The analysis of prominent colors present in the images can indicate a lot about if the image is dull or not. In the following cell, I have added a code to measure the dullness score of the image which can be used as one of the feature in the model. 
# 
# 

# In[2]:


def color_analysis(img):
    # obtain the color palatte of the image 
    palatte = defaultdict(int)
    for pixel in img.getdata():
        palatte[pixel] += 1
    
    # sort the colors present in the image 
    sorted_x = sorted(palatte.items(), key=operator.itemgetter(1), reverse = True)
    light_shade, dark_shade, shade_count, pixel_limit = 0, 0, 0, 25
    for i, x in enumerate(sorted_x[:pixel_limit]):
        if all(xx <= 20 for xx in x[0][:3]): ## dull : too much darkness 
            dark_shade += x[1]
        if all(xx >= 240 for xx in x[0][:3]): ## bright : too much whiteness 
            light_shade += x[1]
        shade_count += x[1]
        
    light_percent = round((float(light_shade)/shade_count)*100, 2)
    dark_percent = round((float(dark_shade)/shade_count)*100, 2)
    return light_percent, dark_percent

# Lets compute the dull score for the sample images from Avito's dataset 

# In[3]:


def perform_color_analysis(img, flag):
    path = images_path + img 
    im = IMG.open(path) #.convert("RGB")
    
    # cut the images into two halves as complete average may give bias results
    size = im.size
    halves = (size[0]/2, size[1]/2)
    im1 = im.crop((0, 0, size[0], halves[1]))
    im2 = im.crop((0, halves[1], size[0], size[1]))

    try:
        light_percent1, dark_percent1 = color_analysis(im1)
        light_percent2, dark_percent2 = color_analysis(im2)
    except Exception as e:
        return None

    light_percent = (light_percent1 + light_percent2)/2 
    dark_percent = (dark_percent1 + dark_percent2)/2 
    
    if flag == 'black':
        return dark_percent
    elif flag == 'white':
        return light_percent
    else:
        return None

# In[4]:


features['dullness'] = features['image'].apply(lambda x : perform_color_analysis(x, 'black'))
topdull = features.sort_values('dullness', ascending = False)
topdull.head(5)

# Lets plot some of the images with very high dullness

# In[5]:


for j,x in topdull.head(2).iterrows():
    path = images_path + x['image']
    html = "<h4>Image : "+x['image']+" &nbsp;&nbsp;&nbsp; (Dullness : " + str(x['dullness']) +")</h4>"
    display(HTML(html))
    display(IMG.open(path).resize((300,300), IMG.ANTIALIAS))

# ## 2. Is the Image too bright or white 
# 
# ### Feature 2 : Image Whiteness
# 
# Some images can be too white or too bright which might not be good for the advertisement purposes. Using the samy type of color analysis, we can check if the images are too white. 

# In[6]:


features['whiteness'] = features['image'].apply(lambda x : perform_color_analysis(x, 'white'))
topdull = features.sort_values('whiteness', ascending = False)
topdull.head(5)

# Lets plot some of the images having high whiteness score

# In[7]:


for j,x in topdull.head(2).iterrows():
    path = images_path + x['image']
    html = "<h4>Image : "+x['image']+" &nbsp;&nbsp;&nbsp; (Whiteness : " + str(x['whiteness']) +")</h4>"
    display(HTML(html))
    display(IMG.open(path).resize((300,300), IMG.ANTIALIAS))

# ## 3. Uniform Images (with no pixel variations)
# 
# ### Feature 3 - Average Pixel Width (using edge detection)
# 
# Some images may contain no pixel variation and are entirely uniform. Average Pixel Width is a measure which indicates the amount of edges present in the image. If this number comes out to be very low, then the image is most likely a uniform image and may not represent right content. 
# 
# To compute this measure, I am using skimage's Canny Detection

# In[8]:


im1 = IMG.open(images_path+'59.png')
im2 = im1.convert(mode='L')
im = np.asarray(im2)

edges1 = feature.canny(im, sigma=1)
edges2 = feature.canny(im, sigma=3)

# display results
fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(8, 3), sharex=True, sharey=True)

ax1.imshow(im, cmap=plt.cm.gray)
ax1.axis('off')
ax1.set_title('noisy image', fontsize=20)

ax2.imshow(edges1, cmap=plt.cm.gray)
ax2.axis('off')
ax2.set_title('Canny filter, $\sigma=1$', fontsize=20)

ax3.imshow(edges2, cmap=plt.cm.gray)
ax3.axis('off')
ax3.set_title('Canny filter, $\sigma=3$', fontsize=20)

fig.tight_layout()

plt.show()

# In[9]:


def average_pixel_width(img):
    path = images_path + img 
    im = IMG.open(path)    
    im_array = np.asarray(im.convert(mode='L'))
    edges_sigma1 = feature.canny(im_array, sigma=3)
    apw = (float(np.sum(edges_sigma1)) / (im.size[0]*im.size[1]))
    return apw*100

# In[10]:


features['average_pixel_width'] = features['image'].apply(average_pixel_width)
tempdf = features.sort_values('average_pixel_width').head()
tempdf 

# Lets plot some images having very low average pixel width

# In[11]:


for j,x in tempdf.head(6).iterrows():
    path = images_path + x['image']
    html = "<h4>Image : "+x['image']+" &nbsp;&nbsp;&nbsp; (Average Pixel Width : " + str(x['average_pixel_width']) +")</h4>"
    display(HTML(html))
    display(IMG.open(path).resize((300,300), IMG.ANTIALIAS))

# Above images are most likely nosie and have low average pixel width values.
# 
# ## 4. What are the key colors used in the image ?
# 
# Colors used in the images play a significant role in garnering the attraction from users. Additional features related to colors such as Dominant and Average colors can be created. 
# 
# ### Feature 4.1 - Dominant Color

# In[12]:


def get_dominant_color(img):
    path = images_path + img 
    img = cv2.imread(path)
    arr = np.float32(img)
    pixels = arr.reshape((-1, 3))

    n_colors = 5
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
    flags = cv2.KMEANS_RANDOM_CENTERS
    _, labels, centroids = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)

    palette = np.uint8(centroids)
    quantized = palette[labels.flatten()]
    quantized = quantized.reshape(img.shape)

    dominant_color = palette[np.argmax(itemfreq(labels)[:, -1])]
    return dominant_color

features['dominant_color'] = features['image'].apply(get_dominant_color)
features.head(10)

# Lets split the dominant color's RGB values to separate features 
# 
# - Feature 4.1.1 dominant_red value
# - Feature 4.1.2 dominant_green value
# - Feature 4.1.3 dominant_blue value

# In[13]:


features['dominant_red'] = features['dominant_color'].apply(lambda x: x[0]) / 255
features['dominant_green'] = features['dominant_color'].apply(lambda x: x[1]) / 255
features['dominant_blue'] = features['dominant_color'].apply(lambda x: x[2]) / 255
features[['dominant_red', 'dominant_green', 'dominant_blue']].head(5)

# ### Feature 4.2 Average Color

# In[14]:


def get_average_color(img):
    path = images_path + img 
    img = cv2.imread(path)
    average_color = [img[:, :, i].mean() for i in range(img.shape[-1])]
    return average_color

features['average_color'] = features['image'].apply(get_average_color)
features.head(10)

# In[15]:


features['average_red'] = features['average_color'].apply(lambda x: x[0]) / 255
features['average_green'] = features['average_color'].apply(lambda x: x[1]) / 255
features['average_blue'] = features['average_color'].apply(lambda x: x[2]) / 255
features[['average_red', 'average_green', 'average_blue']].head(5)

# ## 5. Dimensions of the Image 
# 
# Too Big Images or Too Small Images might not be very good for generating good attraction. Users may skip viewing a very large or very small sized image. Hence for advertisers it is important to set precise dimensions and size of the image. Hence we can create additional features. 
# 
# - Image width
# - Image height
# - Image size

# In[16]:


def getSize(filename):
    filename = images_path + filename
    st = os.stat(filename)
    return st.st_size

def getDimensions(filename):
    filename = images_path + filename
    img_size = IMG.open(filename).size
    return img_size 

# In[17]:


features['image_size'] = features['image'].apply(getSize)
features['temp_size'] = features['image'].apply(getDimensions)
features['width'] = features['temp_size'].apply(lambda x : x[0])
features['height'] = features['temp_size'].apply(lambda x : x[1])
features = features.drop(['temp_size', 'average_color', 'dominant_color'], axis=1)
features.head()

# ## 6. Is the image too Blurry 
# 
# ### Feature 6 - Image Blurrness
# 
# To measure the image blurrness, I refered to the following paper: "Diatom Autofocusing in Brightfield Microscopy: A Comparative Study". 
# 
# In this paper the author Pech-Pacheco et al. has provided variance of the Laplacian Filter which can be used to measure if the image blurryness score.
# 
# In this technique, the single channel of an image is convolved  with the the laplacian filter. If the specified value is less than a threshold value, then image is blurry otherwise not.  
# 
# ![](https://www.pyimagesearch.com/wp-content/uploads/2015/09/detecting_blur_laplacian.png)
# 
# -  Paper Link : http://optica.csic.es/papers/icpr2k.pdf  
# -  Reference : https://www.pyimagesearch.com/2015/09/07/blur-detection-with-opencv/
# 
# 
# 
# 
# 

# In[18]:


def get_blurrness_score(image):
    path =  images_path + image 
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = cv2.Laplacian(image, cv2.CV_64F).var()
    return fm

# In[19]:


features['blurrness'] = features['image'].apply(get_blurrness_score)
features[['image','blurrness']].head(5)

# In[20]:


tempdf = features.sort_values('blurrness')
for y,x in tempdf.head(5).iterrows():
    path = images_path + x['image']
    html = "<h4>Image : "+x['image']+" &nbsp;&nbsp;&nbsp; (Blurrness : " + str(x['blurrness']) +")</h4>"
    display(HTML(html))
    display(IMG.open(path).resize((300,300), IMG.ANTIALIAS))

# ### Other Ideas about features from Image
# 
# - No of objects detected 
# - Total Number of Color Present 
# - No. of shapes detected 
# - Amount of Text Present in the image 
# 
# Other great kernels on Image Feature Extraction:
# 
# 1. https://www.kaggle.com/wesamelshamy/ad-image-recognition-and-quality-scoring by wesamelshamy  
# 2. https://www.kaggle.com/peterhurford/image-feature-engineering by peterhurford  
