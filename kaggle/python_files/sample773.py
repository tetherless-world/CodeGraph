#!/usr/bin/env python
# coding: utf-8

# ## Overview
# Resnet34 is commonly used as an encoder for U-net and SSD, boosting the model performance and training time since you do not need to train the model from scratch. However, in particular cases it makes sense to do fine-tuning of Resnet34 model before using it as a decoder for object localization or image segmentation. In this competition the size of ship masks is much smaller than the size of images that leads to quite unbalanced training with ~1 positive pixel per 1000 negative ones. If images with no ships are used, instead of ~1:1000 you will end up with ~1:10000 unbalance, which is quite tough. Moreover, the training time is ~4 times longer since you need to process more images in each epoch. So, it is reasonable to drop empty images and focus only on ones with ships. Meanwhile, since the current dataset is quite different from ImageNet, the empty images are quite helpful in fine-tuning your encoder on a pseudo task - ship detection. Moreover, when the training of your U-net or SSD model is completed, you can run the model on images without ships, add false positives (~4000 in my case) as negative example to you training set, and train the model for several additional epochs. Finally, a good model focused on a single task, ship detection, can boost the final score when you stack up it with U-net or SSD. If you predict a ship for an empty image you will get automatically zero score for it, and since PLB has ~85% of empty images, prediction of empty images is quite important.
# 
# In this notebook I want to share how to pretrain Resnet34 (or higher end models) on a ship detection task. After training of the head layers of the model on 256x256 rescaled images for one epoch the accuracy has reached 93.7%. The following fine-tuning of entire model for 2 more epochs with learning rate annealing boosted the accuracy to ~97%. If the training is continued for several epochs with a new data set composed of images of 384x384 resolution, the accuracy could be boosted to ~98%. Unfortunately, continuing training the model on full resolution, 768x768, images leaded to reduction of the accuracy that is likely attributed to insufficient model capacity.

# In[ ]:


from fastai.conv_learner import *
from fastai.dataset import *

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

# ### Data

# In[ ]:


PATH = './'
TRAIN = '../input/train/'
TEST = '../input/test/'
SEGMENTATION = '../input/train_ship_segmentations.csv'
exclude_list = ['6384c3e78.jpg','13703f040.jpg', '14715c06d.jpg',  '33e0ff2d5.jpg',
                '4d4e09f2a.jpg', '877691df8.jpg', '8b909bb20.jpg', 'a8d99130e.jpg', 
                'ad55c3143.jpg', 'c8260c541.jpg', 'd6c7f17c7.jpg', 'dc3e7c901.jpg',
                'e44dffe88.jpg', 'ef87bad36.jpg', 'f083256d8.jpg'] #corrupted image

# In[ ]:


nw = 4   #number of workers for data loader
arch = resnet34 #specify target architecture

# In[ ]:


train_names = [f for f in os.listdir(TRAIN)]
test_names = [f for f in os.listdir(TEST)]
for el in exclude_list:
    if(el in train_names): train_names.remove(el)
    if(el in test_names): test_names.remove(el)
#5% of data in the validation set is sufficient for model evaluation
tr_n, val_n = train_test_split(train_names, test_size=0.05, random_state=42)

# In[ ]:


class pdFilesDataset(FilesDataset):
    def __init__(self, fnames, path, transform):
        self.segmentation_df = pd.read_csv(SEGMENTATION).set_index('ImageId')
        super().__init__(fnames, transform, path)
    
    def get_x(self, i):
        img = open_image(os.path.join(self.path, self.fnames[i]))
        if self.sz == 768: return img 
        else: return cv2.resize(img, (self.sz, self.sz))
    
    def get_y(self, i):
        if(self.path == TEST): return 0
        masks = self.segmentation_df.loc[self.fnames[i]]['EncodedPixels']
        if(type(masks) == float): return 0 #NAN - no ship 
        else: return 1
    
    def get_c(self): return 2 #number of classes

# In[ ]:


def get_data(sz,bs):
    #data augmentation
    aug_tfms = [RandomRotate(20, tfm_y=TfmType.NO),
                RandomDihedral(tfm_y=TfmType.NO),
                RandomLighting(0.05, 0.05, tfm_y=TfmType.NO)]
    tfms = tfms_from_model(arch, sz, crop_type=CropType.NO, tfm_y=TfmType.NO, 
                aug_tfms=aug_tfms)
    ds = ImageData.get_ds(pdFilesDataset, (tr_n[:-(len(tr_n)%bs)],TRAIN), 
                (val_n,TRAIN), tfms, test=(test_names,TEST))
    md = ImageData(PATH, ds, bs, num_workers=nw, classes=None)
    md.is_multi = False
    return md

# ### Model

# In[ ]:


sz = 256 #image size
bs = 64  #batch size

md = get_data(sz,bs)
learn = ConvLearner.pretrained(arch, md, ps=0.5) #dropout 50%
learn.opt_fn = optim.Adam

# I begin with finding the optimal learning rate. The following function runs training with different lr and records the loss. Increase of the loss indicates onset of divergence of training. The optimal lr lies in the vicinity of the minimum of the curve but before the onset of divergence. Based on the following plot, for the current setup the divergence starts at ~0.01, and the recommended learning rate is ~0.002.

# In[ ]:


learn.lr_find()
learn.sched.plot()

# Training the head part of the model with constant learning rate for one epoch. 

# In[ ]:


learn.fit(2e-3, 1)

# Unfreeze the model and train it with differential learning rate. The lr of the head part is still 2e-3, while the middle layers of the model a trained with 5e-4 lr, and the base is trained with even smaller lr, 1e-4, since low level detector do not vary much from one image data set to another.

# In[ ]:


learn.unfreeze()
lr=np.array([1e-4,5e-4,2e-3])

# In[ ]:


learn.fit(lr, 1, cycle_len=2, use_clr=(20,8))

# The training has been run with learning rate annealing. Periodic lr increase followed by slow decrease drives the system out of steep minima (when lr is high) towards broader ones (which are explored when lr decreases) that enhances the ability of the model to generalize and reduces overfitting. Due to time limit, only once cycle has been run. But ideally several cycles must be run with gradual increase of the image size, etc. 256x256, 384x384, 768x768, to reach better performance of the model.

# In[ ]:


learn.sched.plot_lr()

# In[ ]:


learn.save('Resnet34_lable_256_1')

# ### Prediction

# In[ ]:


log_preds,y = learn.predict_with_targs(is_test=True)
probs = np.exp(log_preds)[:,1]
pred = (probs > 0.5).astype(int)

# In[ ]:


df = pd.DataFrame({'id':test_names, 'p_ship':probs})
df.to_csv('ship_detection.csv', header=True, index=False)

# ### Training on high resolution images

# Since each epoch on higher resolution images, like 384x384 or 768x768, takes quite long time, training the model from scratch on these images is quite inefficient. Fortunately, modern convolutional nets support input images of arbitrary resolution. To decrease the training time, one can start training the model on low resolution images first and continue training on higher resolution images for only a few epochs. In addition, a model pretrained on low resolution images first generalizes better since a pixel information is less available and high order features are tended to be used.

# In[ ]:


sz = 384 #image size
bs = 32  #batch size

md = get_data(sz,bs)
learn = ConvLearner.pretrained(arch, md, ps=0.5) #dropout 50%
learn.opt_fn = optim.Adam
learn.unfreeze()
lr=np.array([1e-4,5e-4,2e-3])

# In[ ]:


learn.load('Resnet34_lable_256_1')

# Due to the kernal run time limit, the following code does not have enough time to be executed. I hope it is possible to rerun this kernel with disabled cells from "Model" part and included data from the previous commit to complete training of higher resolution images. Each epoch on images 383x384 takes about 1.5 hours. Probably, some paths, like "learn.models_path", may need to be changed.

# In[ ]:


#learn.fit(lr/2, 1, cycle_len=2, use_clr=(20,8)) #lr is smaller since bs is only 32
#learn.save('Resnet34_lable_384_1')
