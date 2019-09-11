#!/usr/bin/env python
# coding: utf-8

# # [Update] Image Classification Confidence and Deal Probability
# Most publicly shared models show that the `image_top_1` feature has one of the strongest signals in predicting deal probability.  Take a look at [SRK], [kxx] and [Bojan Tunguz][bojan] among many others.  However, this is a black box feature to us.  We don't know how it was built to use it as a guide for building similar features.  I took a stab at trying image based features and found one that might work as well.
# 
# If you viewed this kernel before, skip to [the last section below](#Correlation-with-Deal-Probability) for analysis of correlation between image classification confidence and deal probability.
# 
# # Image Quality
# In this competition, the quality of the advertisement image significantly affects the demand volume on an item.  Let's extract the dataset image features and see if we can use it to help predict demand.  I found out that the image classification confidence highly correlates with the deal probability.  However, we have to keep in mind that:
# 
# - Not all advertisements have images.
# - Advertisements with images tend to have a higher deal probability.
# 
# Some code and sections of this notebook were adapted from:
# - https://www.kaggle.com/classtag/lightgbm-with-mean-encode-feature-0-233
# - https://keras.io/applications/#classify-imagenet-classes-with-resnet50
# 
# # Image Classification with Deep Learning
# [Keras] provides pre-trained deep learning models that will save us days annotating, and training our models.  We will just load one that suits our needs and use it to classify our images.  The assumption here is that the image classification accuracy score will reflect how clear it is for a human to identify it, and affect the chance of buying it.
# 
# Different exploratory data analyses [[1], [2], [3]] in this competition show high correlation between the `image_top_1` feature and our target `deal_probability`.  So, we are proceeding in the right direction.
# 
# We will start by preparing our workspace and copying the large pretrained model files to where Keras can find them.
# 
# [keras]: https://keras.io/applications/#resnet50
# [OpenCV]: https://docs.opencv.org/3.4.1/d2/d58/tutorial_table_of_content_dnn.html
# [1]: https://www.kaggle.com/shivamb/indepth-analysis-visualisations-avito-updated
# [2]: https://www.kaggle.com/classtag/lightgbm-with-mean-encode-feature-0-233
# [3]: https://www.kaggle.com/sudalairajkumar/simple-exploration-baseline-notebook-avito
# [SRK]: https://www.kaggle.com/sudalairajkumar/simple-exploration-baseline-notebook-avito
# [kxx]: https://www.kaggle.com/kailex/xgb-text2vec-tfidf-0-2240
# [bojan]: https://www.kaggle.com/tunguz/bow-meta-text-and-dense-features-lb-0-2241

# In[ ]:


"""Copy Keras pre-trained model files to work directory from:
https://www.kaggle.com/gaborfodor/keras-pretrained-models

Code from: https://www.kaggle.com/classtag/extract-avito-image-features-via-keras-vgg16/notebook
"""
import os

cache_dir = os.path.expanduser(os.path.join('~', '.keras'))
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)

# Create symbolic links for trained models.
# Thanks to Lem Lordje Ko for the idea
# https://www.kaggle.com/lemonkoala/pretrained-keras-models-symlinked-not-copied
models_symlink = os.path.join(cache_dir, 'models')
if not os.path.exists(models_symlink):
    os.symlink('/kaggle/input/keras-pretrained-models/', models_symlink)

images_dir = os.path.expanduser(os.path.join('~', 'avito_images'))
if not os.path.exists(images_dir):
    os.makedirs(images_dir)

# Due to Kaggle's disk space restrictions, we will only extract a few images to classify here.  Keep in mind that the pretrained models take almost 650 MB disk space.

# In[ ]:


"""Extract images from Avito's advertisement image zip archive.

Code adapted from: https://www.kaggle.com/classtag/extract-avito-image-features-via-keras-vgg16/notebook
"""
import zipfile

NUM_IMAGES_TO_EXTRACT = 1000

with zipfile.ZipFile('../input/avito-demand-prediction/train_jpg.zip', 'r') as train_zip:
    files_in_zip = sorted(train_zip.namelist())
    for idx, file in enumerate(files_in_zip[:NUM_IMAGES_TO_EXTRACT]):
        if file.endswith('.jpg'):
            train_zip.extract(file, path=file.split('/')[3])


# # ResNet50 *vs* InceptionV3 *vs* Xception
# Let's compare the performance of three pretrained deep learning models implmented for [Keras].
# 
# The models, [ResNet50], [InceptionV3] and [Xception], are all pre-trained on the [ImageNet] dataset.  Here we initialize them and plot a few images from our Avito's image set and the probability of their top classifications.
# 
# **[ImageNet]** is a research project to develop a large image dataset with annotations, such as standard labels and descriptions.  The dataset has been used in the annual [ILSVRC] image classification challenge.  A few of the winners published their pretrained models with the research community, and we are going to use some of them here.
# 
# [resnet50]: https://keras.io/applications/#resnet50
# [VGG16]: https://keras.io/applications/#vgg16
# [Xception]: https://keras.io/applications/#xception
# [InceptionV3]: https://keras.io/applications/#inceptionv3
# [Keras]: https://keras.io/applications/
# [ImageNet]: http://www.image-net.org/
# [ILSVRC]: http://image-net.org/challenges/LSVRC/2017/index

# In[ ]:


import os

import numpy as np
import pandas as pd
from keras.preprocessing import image
import keras.applications.resnet50 as resnet50
import keras.applications.xception as xception
import keras.applications.inception_v3 as inception_v3
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})


# In[ ]:


resnet_model = resnet50.ResNet50(weights='imagenet')
inception_model = inception_v3.InceptionV3(weights='imagenet')
xception_model = xception.Xception(weights='imagenet')

# In[ ]:


def image_classify(model, pak, img, top_n=3):
    """Classify image and return top matches."""
    target_size = (224, 224)
    if img.size != target_size:
        img = img.resize(target_size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = pak.preprocess_input(x)
    preds = model.predict(x)
    return pak.decode_predictions(preds, top=top_n)[0]


def plot_preds(img, preds_arr):
    """Plot image and its prediction."""
    sns.set_color_codes('pastel')
    f, axarr = plt.subplots(1, len(preds_arr) + 1, figsize=(20, 5))
    axarr[0].imshow(img)
    axarr[0].axis('off')
    for i in range(len(preds_arr)):
        _, x_label, y_label = zip(*(preds_arr[i][1]))
        plt.subplot(1, len(preds_arr) + 1, i + 2)
        ax = sns.barplot(np.array(y_label), np.array(x_label))
        plt.xlim(0, 1)
        ax.set()
        plt.xlabel(preds_arr[i][0])
    plt.show()


def classify_and_plot(image_path):
    """Classify an image with different models.
    Plot it and its predicitons.
    """
    img = Image.open(image_path)
    resnet_preds = image_classify(resnet_model, resnet50, img)
    xception_preds = image_classify(xception_model, xception, img)
    inception_preds = image_classify(inception_model, inception_v3, img)
    preds_arr = [('Resnet50', resnet_preds), ('xception', xception_preds), ('Inception', inception_preds)]
    plot_preds(img, preds_arr)

# In[ ]:


image_files = [x.path for x in os.scandir(images_dir)]

# In[ ]:


classify_and_plot(image_files[10])

# In[ ]:


classify_and_plot(image_files[11])

# In[ ]:


classify_and_plot(image_files[12])

# In[ ]:


classify_and_plot(image_files[13])

# In[ ]:


classify_and_plot(image_files[14])

# In[ ]:


classify_and_plot(image_files[15])

# In[ ]:


classify_and_plot(image_files[16])

# # Correlation with Deal Probability
# From the few examples shown above, it seems like the classification confidence of the top class may work as a proxy for image quality.  We can say that if the image is ambiguous to the classification neural network, it will be ambiguous to a human being and unattractive as a product.
# 
# To verfify this, we can measure the correlation betwen the top class' classification score and the deal probability.  To have a reference to measure against, we can compare it to the correlation between the **advertisement's title or description lengths with deal probability**.  Analysis by [Bojan Tunguz][bojan] and others showed that these two features are strong predictors of deal probability.
# 
# [bojan]: https://www.kaggle.com/tunguz/bow-meta-text-and-dense-features-lb-0-2241

# In[ ]:


def classify_inception(image_path):
    """Classify image and return top match."""
    img = Image.open(image_path)
    target_size = (224, 224)
    if img.size != target_size:
        img = img.resize(target_size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = inception_v3.preprocess_input(x)
    preds = inception_model.predict(x)
    return inception_v3.decode_predictions(preds, top=1)[0][0]

def image_id_from_path(path):
    return path.split('/')[3].split('.')[0]

# In[ ]:


train = pd.read_csv('../input/avito-demand-prediction/train.csv')

# In[ ]:


train['desc_len'] = train['description'].str.len()
train['title_len'] = train['title'].str.len()

# In[ ]:


plt.figure(figsize=(10, 10))
inception_conf = [[image_id_from_path(x), classify_inception(x)[2]] for x in image_files]
confidence = pd.DataFrame(inception_conf, columns=['image', 'image_confidence'])
df = confidence.merge(train, on='image')
corr = df[['image', 'image_confidence', 'deal_probability', 'desc_len', 'title_len']].corr()
sns.heatmap(corr, annot=True)
plt.xticks(rotation=30)
plt.yticks(rotation=30)
plt.title('Correlation Between Deal Probability and Strong Model Predictors')
plt.show()

# # Conclusion
# The correlation between `deal_probability` and `image_confidence` shown in the figure above is similar to or stronger than that with `title_len` (title length) or `desc_len` (description length).
# 
# This analysis was done on a relatively small sample of 1,000 images due to disk space restrictions here.  I'm looking forward to seeing someone run this on the full image data set and add it to the list of features we're experimenting with.
