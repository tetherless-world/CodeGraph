#!/usr/bin/env python
# coding: utf-8

# ### Rationale
# I found the explanation for the scoring metric on this competition a little confusing, and I wanted to create a  guide for those who are just entering or haven't made it too far yet. The metric used for this competition is defined as **the mean average precision at different intersection over union (IoU) thresholds**.  
# 
# This tells us there are a few different steps to getting the score reported on the leaderboard. For each image...
# 1. For each submitted nuclei "prediction", calculate the Intersection of Union metric with each "ground truth" mask in the image.
# 2. Calculate whether this mask fits at a range of IoU thresholds.
# 3. At each threshold, calculate the precision across all your submitted masks. 
# 4. Average the precision across thresholds.
# 
# Across the dataset...
# 1. Calculate the mean of the average precision for each image.

# ### Picking a test image
# I'm going to pick a sample image from the training dataset, load the masks, then create a "mock predict" set of masks from it by moving and dilating each individual nucleus mask. Here's the dataset:

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
import pandas as pd
import imageio
from pathlib import Path

# Get image
im_id = '01d44a26f6680c42ba94c9bc6339228579a95d0e2695b149b7cc0c9592b21baf'
im_dir = Path('../input/stage1_train/{}'.format(im_id))
im_path = im_dir / 'images' / '{}.png'.format(im_id)
im = imageio.imread(im_path.as_posix())

# Get masks
targ_masks = []
for mask_path in im_dir.glob('masks/*.png'):
    targ = imageio.imread(mask_path.as_posix())
    targ_masks.append(targ)
targ_masks = np.stack(targ_masks)

# Make messed up masks
pred_masks = np.zeros(targ_masks.shape)
for ind, orig_mask in enumerate(targ_masks):
    aug_mask = ndimage.rotate(orig_mask, ind*1.5, 
                              mode='constant', reshape=False, order=0)
    pred_masks[ind] = ndimage.binary_dilation(aug_mask, iterations=1)

# Plot the objects
fig, axes = plt.subplots(1,3, figsize=(16,9))
axes[0].imshow(im)
axes[1].imshow(targ_masks.sum(axis=0),cmap='hot')
axes[2].imshow(pred_masks.sum(axis=0), cmap='hot')

labels = ['Original', '"GroundTruth" Masks', '"Predicted" Masks']
for ind, ax in enumerate(axes):
    ax.set_title(labels[ind], fontsize=18)
    ax.axis('off')

# ### Intersection Over Union (for a single Prediction-GroundTruth comparison)
# 
# > The IoU of a proposed set of object pixels and a set of true object pixels is calculated as:
# $$
# IoU(A,B)=\frac{A∩B}{A∪B}
# $$
# 
# Let's take one of the nuclei masks from our GroundTruth and Predicted volumes. Their intersections and unions look like this:

# In[2]:


A = targ_masks[3]
B = pred_masks[3]
intersection = np.logical_and(A, B)
union = np.logical_or(A, B)

fig, axes = plt.subplots(1,4, figsize=(16,9))
axes[0].imshow(A, cmap='hot')
axes[0].annotate('npixels = {}'.format(np.sum(A>0)), 
                 xy=(114, 245), color='white', fontsize=16)
axes[1].imshow(B, cmap='hot')
axes[1].annotate('npixels = {}'.format(np.sum(B>0)), 
                 xy=(114, 245), color='white', fontsize=16)

axes[2].imshow(intersection, cmap='hot')
axes[2].annotate('npixels = {}'.format(np.sum(intersection>0)), 
                 xy=(114, 245), color='white', fontsize=16)

axes[3].imshow(union, cmap='hot')
axes[3].annotate('npixels = {}'.format(np.sum(union>0)), 
                 xy=(114, 245), color='white', fontsize=16)

labels = ['GroundTruth', 'Predicted', 'Intersection', 'Union']
for ind, ax in enumerate(axes):
    ax.set_title(labels[ind], fontsize=18)
    ax.axis('off')

# Notice how the intersection will always be less than or equal to the size of the GroundTruth object, and the Union will always be greater than or equal to that size.
# 
# So, for this set of masks, the IoU metric is calculated as:
# $$
# IoU(A,B)=\frac{A∩B}{A∪B} = \frac{564}{849} = 0.664
# $$
# 

# ### Thresholding the IoU value (for a single GroundTruth-Prediction comparison)
# Next, we sweep over a range of IoU thresholds to get a vector for each mask comparison.  The threshold values range from 0.5 to 0.95 with a step size of 0.05: `(0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95)`. 
# 
# In other words, at a threshold of 0.5, a predicted object is considered a "hit" if its intersection over union with a ground truth object is greater than 0.5.

# In[3]:


def get_iou_vector(A, B, n):
    intersection = np.logical_and(A, B)
    union = np.logical_or(A, B)
    iou = np.sum(intersection > 0) / np.sum(union > 0)
    s = pd.Series(name=n)
    for thresh in np.arange(0.5,1,0.05):
        s[thresh] = iou > thresh
    return s

print('Does this IoU hit at each threshold?')
print(get_iou_vector(A, B, 'GT-P'))

# Now, for each prediction mask (P), we'll get a comparison with every ground truth mask (GT). In most cases, this will be zero since nuclei shouldn't overlap, but this also allows flexibility in matching up each mask to each potential nucleus.

# In[4]:


df = pd.DataFrame()
for ind, gt_mask in enumerate(targ_masks):
    s = get_iou_vector(pred_masks[3], gt_mask, 'P3-GT{}'.format(ind))
    df = df.append(s)
print('Performance of Predicted Mask 3 vs. each Ground Truth mask across IoU thresholds')
print(df)

# ### Single-threshold precision for a single image
# 
# Now, in our example, we've created 7 prediction masks ($P_i$) to compare with 7 ground truth masks ($GT_j$). At each threshold, we will have a $7*7$ matrix showing whether there was a hit with that object. The precision value is based on the number of true positives (TP), false negatives (FN), and false positives (FP) in this "hit matrix".
# 
# $$
# Precision(t) = \frac{TP(t)}{TP(t)+FP(t)+FN(t)}
# $$
# 
# * TP: a single predicted object matches a ground truth object with an IoU above the threshold
# * FP: a predicted object had no associated ground truth object. 
# * FN: a ground truth object had no associated predicted object. 

# In[5]:


iou_vol = np.zeros([10, 7, 7])
for ii, pm in enumerate(pred_masks):
    for jj, gt in enumerate(targ_masks):
        s = get_iou_vector(pm, gt, 'P{}-GT{}'.format(ii,jj))
        iou_vol[:,ii,jj] = s.values

mask_labels = ['P{}'.format(x) for x in range(7)]
truth_labels = ['GT{}'.format(x) for x in range(7)]

hits50 = iou_vol[0]
hits75 = iou_vol[4]

fig, axes = plt.subplots(1,2, figsize=(10,9))

axes[0].imshow(hits50, cmap='hot')
axes[0].set_xticks(range(7))
axes[0].set_xticklabels(truth_labels, rotation=45, ha='right', fontsize=16)
axes[0].tick_params(left=False, bottom=False)
axes[0].set_yticks(range(7))
axes[0].set_yticklabels(mask_labels, fontsize=16)
axes[0].tick_params(left=False, bottom=False)
axes[0].set_title('Hit Matrix at $thresh=0.50$', fontsize=18)

axes[1].imshow(hits75, cmap='hot')
axes[1].set_xticks(range(7))
axes[1].set_xticklabels(truth_labels, rotation=45, ha='right', fontsize=16)
axes[1].tick_params(left=False, bottom=False)
axes[1].tick_params(left=False, bottom=False, labelleft=False)
axes[1].set_title('Hit Matrix at $thresh=0.75$', fontsize=18)

for ax in axes:
    # Minor ticks and turn grid on
    ax.set_xticks(np.arange(-.5, 7, 1), minor=True);
    ax.set_yticks(np.arange(-.5, 7, 1), minor=True);
    ax.grid(which='minor', color='w', linestyle='-', linewidth=1)

plt.tight_layout()
plt.show()

# In the above matrix...
# * The number of **true positives** is equal to the number of predictions with a "hit" on a true object.
# * The number of **false positives** is equal to the number of predictions that don't hit anything.
# * The number of **false negatives** is equal to the number of "ground truth" objects that aren't hit.

# In[6]:


def iou_thresh_precision(iou_mat):
    tp = np.sum( iou_mat.sum(axis=1) > 0  )
    fp = np.sum( iou_mat.sum(axis=1) == 0 )
    fn = np.sum( iou_mat.sum(axis=0) == 0 )
    p = tp / (tp + fp + fn)
    return (tp, fp, fn, p)

for thresh, hits in [[0.5, hits50], [0.75, hits75]]:
    tp, fp, fn, p = iou_thresh_precision(hits)
    print('At a threshold of {:0.2f}...\n\tTP = {}\n\tFP = {}\n\tFN = {}\n\tp = {:0.3f}'.format(
                thresh, tp, fp, fn, p))

# ### Multi-threshold precision for a single image
# 
# > The average precision of a single image is then calculated as the mean of the above precision values at each IoU threshold:
# $$
# Avg.\ Precision = \frac{1}{n_{thresh}}  \sum_{t=1}^nprecision(t)
# $$
# 
# Here, we simply take the average of the precision values at each threshold to get our mean precision for the image.

# In[7]:


print('Precision values at each threshold:')
ps = []
for thresh, iou_mat in zip(np.arange(0.5, 1, 0.05), iou_vol):
    _,_,_,p = iou_thresh_precision(iou_mat)
    print('\tt({:0.2f}) = {:0.3f}'.format(thresh, p))
    ps.append(p)
print('Mean precision for image is: {:0.3f}'.format(np.mean(ps)))

# 
# ### Mean average precision for the dataset
# 
# >Lastly, the score returned by the competition metric is the mean taken over the individual average precisions of each image in the test dataset.
# 
# Therefore, the leaderboard metric will simply be the mean of the precisions across all the images.
# 
# Hope you found this helpful -- I know it helped me to work through it!
