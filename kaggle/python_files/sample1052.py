#!/usr/bin/env python
# coding: utf-8

# Hi, this is my second kernel. I tried out the fruits dataset with fastai.vision library. And the results are pretty good with 99.96% accuracy.

# In[ ]:


from fastai.vision import *
from fastai.metrics import error_rate

# Current dataset directory is read-only; we might need to modify some training set data.
# Moving to /tmp gives us Write permission on the folder.

# In[ ]:


cp -r /kaggle/input/fruits-360_dataset/ /tmp

# In[ ]:


path= '/tmp/fruits-360_dataset/fruits-360/'

# Creating our databunch object. Fortunately this dataset uses imagenet style, so our factory method will suffice.

# In[ ]:


data=ImageDataBunch.from_folder(path,train='Training',valid='Test',size=224,bs=64).normalize(imagenet_stats)
data.show_batch(rows=3,figsize=(5,5))

# Now let's create our learner. I'm using resnet34 for now with accuracy as metric.
# Resnet34 trains faster so let's go with that.

# In[ ]:


learner34=cnn_learner(data,models.resnet34,metrics=accuracy)

# In[ ]:


learner34.fit_one_cycle(4)

# 99% Accuracy using resnet34 is great. We might be able to increase our accuracy even more with resent50.
# Let's try that!

# In[ ]:


learner50=cnn_learner(data,models.resnet50,metrics=accuracy)

# In[ ]:


learner50.fit_one_cycle(2)

# Resnet50 performs even better. Let's try fine tuning our learning rate now.

# In[ ]:


learner50.lr_find()
learner50.recorder.plot()

# It seems learning rate range is good between 5e-6 and 8e-5.
# Let's unfreeze the model try that with 2 epochs to train, we don't want to overfit.

# In[ ]:


learner50.unfreeze()
learner50.fit_one_cycle(2,max_lr=slice(5e-6,8e-5))

# Results are quite good.
# 
# Now let's see which cases we failed.

# In[ ]:


interp=ClassificationInterpretation.from_learner(learner50)
interp.plot_top_losses(9,figsize=(12,12))

# In[ ]:


interp.most_confused(min_val=1)

# In[ ]:


learner50.export('fruitsmodel.pkl')

# Thank you for sticking around.
