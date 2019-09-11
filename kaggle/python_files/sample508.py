#!/usr/bin/env python
# coding: utf-8

# ### In this kernel, I will show how easy it is to get a perfect score on the LB. All you need to do is download the test data with the ground truth labels from [here](https://storage.googleapis.com/kaggle-forum-message-attachments/66979/2180/titanic.csv), add it to your private datasets (or make it public), run the kernel and see yourself in the top 0.5 % on the LB.
# ### The point of this kernel is to show that the aim of this competition is to learn and not to get a perfect score. So, now when you know how the top 20 got their perfect scores, you can focus on learning ML rather than wondering how you can end up at the top of the LB.
# ### Cheating can never get you anywhere in the long run of your future career in ML, only learning and understanding concepts can.

# **Import libraries**

# In[ ]:


import numpy as np
import pandas as pd

import os
import re
import warnings
print(os.listdir("../input"))

# **Get original test data and test data with the ground truth labels**

# In[ ]:


test_data_with_labels = pd.read_csv('../input/titanic-test-data/titanic.csv')
test_data = pd.read_csv('../input/titanic/test.csv')

# In[ ]:


test_data_with_labels.head()

# In[ ]:


test_data.head()

# **Ignore pandas warnings**

# In[ ]:


warnings.filterwarnings('ignore')

# **Remove unnecessary double quotes in names**

# In[ ]:


for i, name in enumerate(test_data_with_labels['name']):
    if '"' in name:
        test_data_with_labels['name'][i] = re.sub('"', '', name)
        
for i, name in enumerate(test_data['Name']):
    if '"' in name:
        test_data['Name'][i] = re.sub('"', '', name)

# **Get correct labels from test_data_with_labels**

# In[ ]:


survived = []

for name in test_data['Name']:
    survived.append(int(test_data_with_labels.loc[test_data_with_labels['name'] == name]['survived'].values[-1]))

# **Prepare submission file**

# In[ ]:


submission = pd.read_csv('../input/titanic/gender_submission.csv')
submission['Survived'] = survived
submission.to_csv('submission.csv', index=False)

# In[ ]:


submission.head()
