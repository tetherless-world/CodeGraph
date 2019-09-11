#!/usr/bin/env python
# coding: utf-8

# # Galactic vs Extragalactic Objects
# 
# The astronomical transients that appear in this challenge can be separated into two distinct groups: ones that are in our Milky Way galaxy (galactic) and ones that are outside of our galaxy (extragalactic). As described in the data note, all of the galactic objects have been assigned a host galaxy photometric redshift of 0. We can use this information to immediately classify every object as either galactic or extragalactic and remove a lot of potential options from the classification. Doing so results in matching the naive benchmark.
# 
# We find that all of the classes are either uniquely galactic or extragalactic except for class 99 which represents the unknown objects that aren't in the training set.

# ## Load the data
# 
# For this notebook, we'll only need the metadata.

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# In[ ]:



# In[ ]:


meta_data = pd.read_csv('../input/training_set_metadata.csv')
test_meta_data = pd.read_csv('../input/test_set_metadata.csv')

# Map the classes to the range 0-14. We manually add in the 99 class that doesn't show up in the training data.

# In[ ]:


targets = np.hstack([np.unique(meta_data['target']), [99]])
target_map = {j:i for i, j in enumerate(targets)}
target_ids = [target_map[i] for i in meta_data['target']]
meta_data['target_id'] = target_ids

# Let's look at which classes show up in galactic vs extragalactic hosts. We can use the hostgal_specz key which is 0 for galactic objects.

# In[ ]:


galactic_cut = meta_data['hostgal_specz'] == 0
plt.figure(figsize=(10, 8))
plt.hist(meta_data[galactic_cut]['target_id'], 15, (0, 15), alpha=0.5, label='Galactic')
plt.hist(meta_data[~galactic_cut]['target_id'], 15, (0, 15), alpha=0.5, label='Extragalactic')
plt.xticks(np.arange(15)+0.5, targets)
plt.gca().set_yscale("log")
plt.xlabel('Class')
plt.ylabel('Counts')
plt.xlim(0, 15)
plt.legend();

# There is no overlap at all between the galactic and extragalactic objects in the training set. Class 99 isn't represented in the training set at all. Let's make a classifier that checks if an object is galactic or extragalactic and then assigns a flat probability to each class in that group. We'll include class 99 in both the galactic and extragalactic groups.

# In[ ]:


# Build the flat probability arrays for both the galactic and extragalactic groups
galactic_cut = meta_data['hostgal_specz'] == 0
galactic_data = meta_data[galactic_cut]
extragalactic_data = meta_data[~galactic_cut]

galactic_classes = np.unique(galactic_data['target_id'])
extragalactic_classes = np.unique(extragalactic_data['target_id'])

# Add class 99 (id=14) to both groups.
galactic_classes = np.append(galactic_classes, 14)
extragalactic_classes = np.append(extragalactic_classes, 14)

galactic_probabilities = np.zeros(15)
galactic_probabilities[galactic_classes] = 1. / len(galactic_classes)
extragalactic_probabilities = np.zeros(15)
extragalactic_probabilities[extragalactic_classes] = 1. / len(extragalactic_classes)

# Apply this prediction to the data. We simply choose which of the two probability arrays to use based off of whether the object is galactic or extragalactic.

# In[ ]:


# Apply this prediction to a table
import tqdm
def do_prediction(table):
    probs = []
    for index, row in tqdm.tqdm(table.iterrows(), total=len(table)):
        if row['hostgal_photoz'] == 0:
            prob = galactic_probabilities
        else:
            prob = extragalactic_probabilities
        probs.append(prob)
    return np.array(probs)

pred = do_prediction(meta_data)
test_pred = do_prediction(test_meta_data)

# Now write the prediction out and submit it. This notebook gets a score of 2.158 which matches the naive benchmark.

# In[ ]:


test_df = pd.DataFrame(index=test_meta_data['object_id'], data=test_pred, columns=['class_%d' % i for i in targets])
test_df.to_csv('./naive_benchmark.csv')
