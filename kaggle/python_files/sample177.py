#!/usr/bin/env python
# coding: utf-8

# # Motivation
# ![Owl](https://i.kym-cdn.com/photos/images/newsfeed/000/572/078/d6d.jpg)

# 

# In[ ]:


import pandas as pd
import numpy as np
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
import ast
sns.set_palette(sns.color_palette('copper', 20))
from datetime import date, timedelta

start = dt.datetime.now()

# # Step 0 - Data Understanding

# In[ ]:


owls = pd.read_csv('../input/train_simplified/owl.csv')
owls = owls[owls.recognized]
owls['timestamp'] = pd.to_datetime(owls.timestamp)
owls = owls.sort_values(by='timestamp', ascending=False)[-100:]
owls['drawing'] = owls['drawing'].apply(ast.literal_eval)

owls.head()

# In[ ]:


n = 10
fig, axs = plt.subplots(nrows=n, ncols=n, sharex=True, sharey=True, figsize=(16, 10))
for i, drawing in enumerate(owls.drawing):
    ax = axs[i // n, i % n]
    for x, y in drawing:
        ax.plot(x, -np.array(y), lw=3)
    ax.axis('off')
fig.savefig('owls.png', dpi=200)
plt.show();

# # Step 1 - Create baseline model

# In[ ]:


submission = pd.read_csv('../input/sample_submission.csv')
submission.shape

submission['word'] = 'owl owl owl'
submission.to_csv('submission_owl.csv', index=False)
submission.head()

# # Step 2 - Train a Neural Network
# 
# ...

# In[ ]:


target = [
    'owl owl owl',
    'owl owl owl',
    'owl owl owl',
    'owl owl owl',
    'owl owl owl',
    'owl owl owl',
    'owl owl owl',
    'owl owl owl',
    'owl owl owl',
    'owl owl owl',
    'owl owl owl',
    'owl owl owl',
    'owl owl owl',
    'owl owl owl',
    'owl owl owl',
]

# # Step 3 - Profit

# In[ ]:


end = dt.datetime.now()
print('Latest run {}.\nTotal time {}s'.format(end, (end - start).seconds))
