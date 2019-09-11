#!/usr/bin/env python
# coding: utf-8

# **If you like this Kernel enough to fork it, please remember to upvote as well!**

# This is the first notebook from version 3 of the fast.ai course! For more information you can visit https://www.fast.ai/.

# # Lesson 1 - What's your pet

# In this lesson we will build our first image classifier from scratch, and see if we can achieve world-class results. Let's dive in!
# 
# Every notebook starts with the following three lines; they ensure that any edits to libraries you make are reloaded here automatically, and also that any charts or images displayed are shown in this notebook.

# In[ ]:



# We import all the necessary packages. We are going to work with the [fastai V1 library](http://www.fast.ai/2018/10/02/fastai-ai/) which sits on top of [Pytorch 1.0](https://hackernoon.com/pytorch-1-0-468332ba5163). The fastai library provides many useful functions that enable us to quickly and easily build neural networks and train our models.

# In[ ]:


from fastai import *
from fastai.vision import *

# If you're using a computer with an unusually small GPU, you may get an out of memory error when running this notebook. If this happens, click Kernel->Restart, uncomment the 2nd line below to use a smaller *batch size* (you'll learn all about what this means during the course), and try again.

# In[ ]:


bs = 64
# bs = 16   # uncomment this line if you run out of memory even after clicking Kernel->Restart

# ## Looking at the data

# We are going to use the [Oxford-IIIT Pet Dataset](http://www.robots.ox.ac.uk/~vgg/data/pets/) by [O. M. Parkhi et al., 2012](http://www.robots.ox.ac.uk/~vgg/publications/2012/parkhi12a/parkhi12a.pdf) which features 12 cat breeds and 25 dogs breeds. Our model will need to learn to differentiate between these 37 distinct categories. According to their paper, the best accuracy they could get in 2012 was 59.21%, using a complex model that was specific to pet detection, with separate "Image", "Head", and "Body" models for the pet photos. Let's see how accurate we can be using deep learning!
# 
# We are going to use the `untar_data` function to which we must pass a URL as an argument and which will download and extract the data.

# In[ ]:


help(untar_data)

# In[ ]:


path = untar_data(URLs.PETS); path

# In[ ]:


path.ls()

# In[ ]:


path_anno = path/'annotations'
path_img = path/'images'

# The first thing we do when we approach a problem is to take a look at the data. We _always_ need to understand very well what the problem is and what the data looks like before we can figure out how to solve it. Taking a look at the data means understanding how the data directories are structured, what the labels are and what some sample images look like.
# 
# The main difference between the handling of image classification datasets is the way labels are stored. In this particular dataset, labels are stored in the filenames themselves. We will need to extract them to be able to classify the images into the correct categories. Fortunately, the fastai library has a handy function made exactly for this, `ImageDataBunch.from_name_re` gets the labels from the filenames using a [regular expression](https://docs.python.org/3.6/library/re.html).

# In[ ]:


fnames = get_image_files(path_img)
fnames[:5]

# In[ ]:


np.random.seed(2)
pat = re.compile(r'/([^/]+)_\d+.jpg$')

# In[ ]:


data = ImageDataBunch.from_name_re(path_img, fnames, pat, ds_tfms=get_transforms(), size=224, bs=bs, num_workers=0
                                  ).normalize(imagenet_stats)

# In[ ]:


data.show_batch(rows=3, figsize=(7,6))

# In[ ]:


print(data.classes)
len(data.classes),data.c

# ## Training: resnet34

# Now we will start training our model. We will use a [convolutional neural network](http://cs231n.github.io/convolutional-networks/) backbone and a fully connected head with a single hidden layer as a classifier. Don't know what these things mean? Not to worry, we will dive deeper in the coming lessons. For the moment you need to know that we are building a model which will take images as input and will output the predicted probability for each of the categories (in this case, it will have 37 outputs).
# 
# We will train for 4 epochs (4 cycles through all our data).

# In[ ]:


learn = create_cnn(data, models.resnet34, metrics=error_rate)

# In[ ]:


learn.fit_one_cycle(4)

# In[ ]:


learn.save('stage-1')

# ## Results

# Let's see what results we have got. 
# 
# We will first see which were the categories that the model most confused with one another. We will try to see if what the model predicted was reasonable or not. In this case the mistakes look reasonable (none of the mistakes seems obviously naive). This is an indicator that our classifier is working correctly. 
# 
# Furthermore, when we plot the confusion matrix, we can see that the distribution is heavily skewed: the model makes the same mistakes over and over again but it rarely confuses other categories. This suggests that it just finds it difficult to distinguish some specific categories between each other; this is normal behaviour.

# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)

losses,idxs = interp.top_losses()

len(data.valid_ds)==len(losses)==len(idxs)

# In[ ]:


interp.plot_top_losses(9, figsize=(15,11))

# In[ ]:


doc(interp.plot_top_losses)

# In[ ]:


interp.plot_confusion_matrix(figsize=(12,12), dpi=60)

# In[ ]:


interp.most_confused(min_val=2)

# ## Unfreezing, fine-tuning, and learning rates

# Since our model is working as we expect it to, we will *unfreeze* our model and train some more.

# In[ ]:


learn.unfreeze()

# In[ ]:


learn.fit_one_cycle(1)

# In[ ]:


learn.load('stage-1');

# In[ ]:


learn.lr_find()

# In[ ]:


learn.recorder.plot()

# In[ ]:


learn.unfreeze()
learn.fit_one_cycle(2, max_lr=slice(1e-6,1e-4))

# That's a pretty accurate model!

# ## Training: resnet50

# Now we will train in the same way as before but with one caveat: instead of using resnet34 as our backbone we will use resnet50 (resnet34 is a 34 layer residual network while resnet50 has 50 layers. It will be explained later in the course and you can learn the details in the [resnet paper](https://arxiv.org/pdf/1512.03385.pdf)).
# 
# Basically, resnet50 usually performs better because it is a deeper network with more parameters. Let's see if we can achieve a higher performance here. To help it along, let's us use larger images too, since that way the network can see more detail. We reduce the batch size a bit since otherwise this larger network will require more GPU memory.

# In[ ]:


data = ImageDataBunch.from_name_re(path_img, fnames, pat, ds_tfms=get_transforms(),
                                   size=299, bs=bs//2, num_workers=0).normalize(imagenet_stats)

# In[ ]:


learn = create_cnn(data, models.resnet50, metrics=error_rate)

# In[ ]:


learn.lr_find()
learn.recorder.plot()

# In[ ]:


learn.fit_one_cycle(8)

# In[ ]:


learn.save('stage-1-50')

# It's astonishing that it's possible to recognize pet breeds so accurately! Let's see if full fine-tuning helps:

# In[ ]:


learn.unfreeze()
learn.fit_one_cycle(3, max_lr=slice(1e-6,1e-4))

# If it doesn't, you can always go back to your previous model.

# In[ ]:


learn.load('stage-1-50');

# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)

# In[ ]:


interp.most_confused(min_val=2)

# ## Other data formats

# In[ ]:


path = untar_data(URLs.MNIST_SAMPLE); path

# In[ ]:


tfms = get_transforms(do_flip=False)
data = ImageDataBunch.from_folder(path, ds_tfms=tfms, size=26, num_workers=0)

# In[ ]:


data.show_batch(rows=3, figsize=(5,5))

# In[ ]:


learn = create_cnn(data, models.resnet18, metrics=accuracy)
learn.fit(2)

# In[ ]:


df = pd.read_csv(path/'labels.csv')
df.head()

# In[ ]:


data = ImageDataBunch.from_csv(path, ds_tfms=tfms, size=28, num_workers=0)

# In[ ]:


data.show_batch(rows=3, figsize=(5,5))
data.classes

# In[ ]:


data = ImageDataBunch.from_df(path, df, ds_tfms=tfms, size=24, num_workers=0)
data.classes

# In[ ]:


fn_paths = [path/name for name in df['name']]; fn_paths[:2]

# In[ ]:


pat = r"/(\d)/\d+\.png$"
data = ImageDataBunch.from_name_re(path, fn_paths, pat=pat, ds_tfms=tfms, size=24, num_workers=0)
data.classes

# In[ ]:


data = ImageDataBunch.from_name_func(path, fn_paths, ds_tfms=tfms, size=24,
        label_func = lambda x: '3' if '/3/' in str(x) else '7', num_workers=0)
data.classes

# In[ ]:


labels = [('3' if '/3/' in str(x) else '7') for x in fn_paths]
labels[:5]

# In[ ]:


data = ImageDataBunch.from_lists(path, fn_paths, labels=labels, ds_tfms=tfms, size=24, num_workers=0)
data.classes
