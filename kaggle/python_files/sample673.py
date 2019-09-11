#!/usr/bin/env python
# coding: utf-8

# # TrackML Particle Tracking Challenge
# 
# >A dataset consisting of a simulation of a typical full Silicon LHC detector will be made available, listing for each event the measured 3D points (x, y , z) coordinates, an event being the recording of the collision of two bunches of protons. The ground truth is provided as separate file, indicating which hits correspond to the same particle.
# 
# >For each collision, about 10.000 space tracks (helicoidal trajectories originating approximately from the center of the detector), will leave about 10 precise 3D points. The core pattern recognition tracking task is to associate the 100.000 3D points into tracks. Current studies show that traditional algorithms suffer from a combinatorial explosion of the CPU time.
# There is a strong potential for application of Machine Learning techniques to this tracking issue. The problem can be related to representation learning, to combinatorial optimization, to clustering (associate together the hits which were deposited by the same particle), and even to time series prediction. An essential question is to efficiently exploit the a priori knowledge about geometrical constraints (structural priors) [1].
# 
# >Specifically, in this competition, you’re challenged to build an algorithm that quickly reconstructs particle tracks from 3D points left in the silicon detectors [2].
# 
# [1] https://sites.google.com/site/trackmlparticle/dataset  
# [2] https://www.kaggle.com/c/trackml-particle-identification
# 
# To slake some visualization hunger, here's a plot of a few particle trajectories. We're trying to find these from only a few points left behind.
# ![tracks](https://i.imgur.com/c0aGYdt.pngt)

# # Table of Contents
# 1. [Introduction](#intro)
# 2. [Importing and Loading](#import)
# 3. [Hits](#hits)
# 4. [Cells](#cells)
# 5. [Particles](#particles)
# 6. [Truth](#truth)

# # <a name="intro">Introduction</a>

# ## "Physics"
# Disclaimer 1: I'm an astroparticle physicist and not a HEP physicist. I'm also not a software engineer nor a data scientist.  
# Disclaimer 2: I commit early and commit often. I think this messes with Kaggle and I often loose bits of work. It's annoying. So if sentences just end, etc. that's not me.
# 
# Okay, here's the gist: accelerated and guided by supercooled magnets, a group of protons from the left and a group of protons from the right collide in the center of this LHC ([Large Hadron Collider](https://en.wikipedia.org/wiki/Large_Hadron_Collider), a particle accelerator) detector simulation (a la [ATLAS](https://en.wikipedia.org/wiki/ATLAS_experiment), [CMS](https://en.wikipedia.org/wiki/Compact_Muon_Solenoid) who are both two of many experiments attached to the LHC). This collision, with a lot of initial kinetic energy, transforms that energy into a lot of non-proton particles. These particles fly away from the center of the detector and leave their tracks behind via charge as they hit (/pass through) silicon plates. These plates are the actual inividual detectors, placed and layered in a such a way to allow track reconstruction (and ultimately particle identification, but that's beyond the scope of this challenge). However, we don't know the actual track line -- we only know where the particles hit. Like a bullet moving through a book.
# 
# I won't get into the traditional methods for such track reconstruction, but this challenge wants us to use ML techniques instead to reconstruct (accurately and quickly) these tracks from only detector hit positions and some other information like charge and momentum that are magically given to us.
# 
# ## Data
# There are 8850 __events__ in the training data. Each event creates over 10k particles summing over 100k detector hits. That means each particle passes through an average of 10 silicon plates.
# * __Hits__ are given as x, y, z positions and the detector groups they interacted with: volume, layer, module in increasing granularity (volume big, module small). 
# * For each hit, we're also also given the __cells__ position and charge that did the detecting. These are the smallest unit of detector group, even finer than module. 
# * For each event we're also given __particle__ information x, y, z momentums, and charge, and number of detectors hit from that particle.  
# * Finally, we're given __truth__, telling us for certain which hits belong to which particles. Maybe this is kind of like our "y", and the other features are kind of like our "X".
# 
# This is all explained in more detail below, later.
# 
# I'll first go through some data visualization to get a sense of what's up, then eventually make my may into some physics. It's this physics that holds the key to any real modelling.

# # <a name="import">Importing and Loading</a>

# ## Import standard libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits import mplot3d
import seaborn as sns

# ## Download data and/or import TrackML utility library
# 
# * For personal use, I downloaded `train_sample.zip` and `detectors.zip`. `train_sample.zip` unzipped into the folder `train_100_events`.  To install the [TrackML utility library](https://github.com/LAL/trackml-library) via pip, type 
# `pip install --user git+https://github.com/LAL/trackml-library.git`
# into your console. `ls *-hits*` shows that event id begins at 000001000 and ends at 000001099.
# 
# * For Kernel use, see below to install the utility library.  Under the Kernel Data tab, it's noted that  "This file is an ZIP archive. ZIP archive files will be uncompressed and their contents available at the root folder when this dataset is used in Kernels."
# I couldn't  access ```train_sample``` ("`Exception: No file matches '../input/train_sample/event000001000-hits.csv*'`") so I'll access ```train_1``` instead. 
# Note that despite the [example given by TrackML](https://www.kaggle.com/c/trackml-particle-identification/data), event ids below 000001000 do not seem to exist (in `train_1`).
# 
# Add a custom package:
# 1. Click the [<] button at top-right of Kernel screen
# 2. Click Settings
# 3. Enter "LAL/trackml-library", e.g., into "GitHub user/repo" space at the bottom
# 4. Click the (->) button to the left of that
# 5. Restart Kernel by clicking the circular refresh/recycle-y button at the bottom-right of the screen, in the Console
# 6. Custom libraries will now import when imported

# In[2]:


from trackml.dataset import load_event, load_dataset
from trackml.randomize import shuffle_hits
from trackml.score import score_event

# ## Load one event for this EDA

# In[3]:


# One event of 8850
event_id = 'event000001000'
# "All methods either take or return pandas.DataFrame objects"
hits, cells, particles, truth = load_event('../input/train_1/'+event_id)

# __hits__  
# The hits file contains the following values for each hit/entry:  
# * __hit_id__: numerical identifier of the hit inside the event.
# * __x, y, z__: measured x, y, z position (in millimeter) of the hit in global coordinates.
# * __volume_id__: numerical identifier of the detector group.
# * __layer_id__: numerical identifier of the detector layer inside the group.
# * __module_id__: numerical identifier of the detector module inside the layer.  
# 
# The volume/layer/module id could in principle be deduced from x, y, z. They are given here to simplify detector-specific data handling.
# 
# __cells__  
# The cells file contains the constituent active detector cells that comprise each hit. The cells can be used to refine the hit to track association. A cell is the smallest granularity inside each detector module, much like a pixel on a screen, except that depending on the volume_id a cell can be a square or a long rectangle. It is identified by two channel identifiers that are unique within each detector module and encode the position, much like column/row numbers of a matrix. A cell can provide signal information that the detector module has recorded in addition to the position. Depending on the detector type only one of the channel identifiers is valid, e.g. for the strip detectors, and the value might have different resolution.  
# * __hit_id__: numerical identifier of the hit as defined in the hits file.
# * __ch0, ch1__: channel identifier/coordinates unique within one module.
# * __value__: signal value information, e.g. how much charge a particle has deposited.
# 
# __particles__  
# The particles files contains the following values for each particle/entry:  
# * __particle_id__: numerical identifier of the particle inside the event.
# * __vx, vy, vz__: initial position or vertex (in millimeters) in global coordinates.
# * __px, py, pz__: initial momentum (in GeV/c) along each global axis.
# * __q__: particle charge (as multiple of the absolute electron charge).
# * __nhits__: number of hits generated by this particle.
# 
# __truth__  
# The truth file contains the mapping between hits and generating particles and the true particle state at each measured hit. Each entry maps one hit to one particle.  
# * __hit_id__: numerical identifier of the hit as defined in the hits file.
# * __particle_id__: numerical identifier of the generating particle as defined in the particles file. A value of 0 means that the hit did not originate from a reconstructible particle, but e.g. from detector noise.
# * __tx, ty, tz__ true intersection point in global coordinates (in millimeters) between the particle trajectory and the sensitive surface.
# * __tpx, tpy, tpz__ true particle momentum (in GeV/c) in the global coordinate system at the intersection point. The corresponding vector is tangent to the particle trajectory at the intersection point.
# * __weight__ per-hit weight used for the scoring metric; total sum of weights within one event equals to one.
# 
# 
# Now let's get to it...

# # <a name="hits">Hits</a>
# The hits file contains the following values for each hit/entry:  
# * __hit_id__: numerical identifier of the hit inside the event.
# * __x, y, z__: measured x, y, z position (in millimeter) of the hit in global coordinates.
# * __volume_id__: numerical identifier of the detector group.
# * __layer_id__: numerical identifier of the detector layer inside the group.
# * __module_id__: numerical identifier of the detector module inside the layer.  
# 
# The volume/layer/module id could in principle be deduced from x, y, z. They are given here to simplify detector-specific data handling.

# In[4]:


hits.head()

# In[5]:


hits.tail()

# In[6]:


hits.describe()

# Look at that. The mean of hit.x,y,z is only a few millimeters from the center of the detector. But the std is very high. Lots of spread in hit location here.

# ## What's the spatial distribution of hits within the detector for this event?

# I think first and foremost we need to understand the geometry of our detector. For reference, our hit geometry _should_ inspire a detector that looks like the one below. Spoilers: it does. Each event was created by a bundle of protons smashing into eachother as they meet from opposite ends of this "tube".
# 
# ![LHC](https://storage.googleapis.com/kaggle-media/competitions/CERN/cern_graphic.png)

# In[7]:


#plt.figure(figsize=(10,10))
#plt.scatter(hits.x,hits.y, s=1)
#plt.show()
# Essentially the same plot as above but includes univariate plots and Pearson corr coeff
radialview = sns.jointplot(hits.x, hits.y, size=10, s=1)
radialview.set_axis_labels('x (mm)', 'y (mm)')
plt.show()

# That solid core in the middle is probably more concentric detectors. Let's zoom in and find out.

# In[8]:


radialview = sns.jointplot( hits[hits.x.abs()<200].x, hits[hits.y.abs()<200].y, size=10, s=1)
radialview.set_axis_labels('x (mm)', 'y (mm)')
plt.show()

# Indeed, the inner detectors are just more concentric rings.
# 
# That scattering we see in between the rings are just events from the vertical detectors. Below shows these caps removed, and again shows the concentric nature of the inner detector.

# In[9]:


nocap = hits[hits.z.abs()<200]
radialview = sns.jointplot( nocap[nocap.x.abs()<200].x, nocap[nocap.y.abs()<200].y, size=10, s=1)
radialview.set_axis_labels('x (mm)', 'y (mm)')
plt.show()

# Notice those clipped detectors at the corners? Notice how they seem to be layered? That's because they are small flat rectangles shingled into this pattern. Let's zoom in even more to the center-most ones.

# In[10]:


radialview = sns.jointplot( nocap[nocap.x.abs()<50].x, nocap[nocap.y.abs()<50].y, size=10, s=1)
radialview.set_axis_labels('x (mm)', 'y (mm)')
plt.show()

# We can look at the sideview of the full detector, too.

# In[11]:


# See here for why these two plots are not in subplots:
# https://stackoverflow.com/a/35044845
# There appears to be radial symmetry, so x ~= y for this visualization
'''
fig = plt.figure()
sns.jointplot(hits.z, hits.y, s=1)
axialview.set_axis_labels('z (mm)', 'y (mm)')
axialview.fig.set_figwidth(24)
axialview.fig.set_figheight(8)
'''
plt.figure(figsize=(24,8))
axialview = plt.scatter(hits.z, hits.y, s=1) 
plt.xlabel('z (mm)')
plt.ylabel('y (mm)')
plt.show()

# For completeness, below is a 3D plot of a random sample effectively combining both of the above. Note though I could not get it to plot with proportional axes without adding a couple of points to force full axes.

# In[12]:


plt.figure(figsize=(15,15))
ax = plt.axes(projection='3d')
sample = hits.sample(30000)
ax.scatter(sample.z, sample.x, sample.y, s=5, alpha=0.5)
ax.set_xlabel('z (mm)')
ax.set_ylabel('x (mm)')
ax.set_zlabel('y (mm)')
# These two added to widen the 3D space
ax.scatter(3000,3000,3000, s=0)
ax.scatter(-3000,-3000,-3000, s=0)
plt.show()

# Normally one should bin these events with position resolution in mind. But it appears resolution is effectively perfect, or " Depending on the detector type only one of the channel identifiers is valid, e.g. for the strip detectors, and the value might have different resolution." This is something I need to think on.

# ## Where are the individual detector groups located?

# Let's plot each detector volume as a different color:

# In[24]:


# I am sure there's a nice pandas-specific method, but alas
volumes = hits.volume_id.unique()

#plt.figure(figsize=(15,15))
#for volume in volumes:
#    v = hits[hits.volume_id == volume]
#    #sns.jointplot(v.x, v.y, size=10, s=1)
#    plt.scatter(v.x, v.y, s=10, label='Volume '+str(volume), alpha=0.5)
#plt.title('Detector Volumes, Radial View')
#plt.xlabel('x (mm)')
#plt.ylabel('y (mm)')
#plt.legend()
#plt.show()

# In[23]:


#plt.figure(figsize=(24,8))
#for volume in volumes:
#    v = hits[hits.volume_id == volume]
#    plt.scatter(v.z, v.y, s=10, label='Volume '+str(volume), alpha=0.5)
#plt.title('Detector Volumes, Axial View')
#plt.xlabel('z (mm)')
#plt.ylabel('y (mm)')
#plt.legend()
#plt.show()

# Let's ~~also~~ only plot these in 3D:

# In[15]:


plt.figure(figsize=(20,20))
ax = plt.axes(projection='3d')
for volume in volumes:
    v = sample[sample.volume_id == volume]
    ax.scatter(v.z, v.x, v.y, s=5, label='Volume '+str(volume), alpha=0.5)
ax.set_xlabel('z (mm)')
ax.set_ylabel('x (mm)')
ax.set_zlabel('y (mm)')
ax.legend()
# These two added to widen the 3D space
ax.scatter(3000,3000,3000, s=0)
ax.scatter(-3000,-3000,-3000, s=0)
plt.show()

# We can also look at the layers:

# In[21]:


layers = hits.layer_id.unique()

#plt.figure(figsize=(15,15))
#for layer in layers:
#    l = hits[hits.layer_id == layer]
#    plt.scatter(l.x, l.y, s=10, label='Layer '+str(layer), alpha=0.5)
#plt.title('Detector Layers, Radial View')
#plt.xlabel('x (mm)')
#plt.ylabel('y (mm)')
#plt.legend()
#plt.show()

# In[22]:


#plt.figure(figsize=(24,8))
#for layer in layers:
#    l = hits[hits.layer_id == layer]
#    plt.scatter(l.z, l.y, s=10, label='Layer '+str(layer), alpha=0.5)
#plt.title('Detector Layers, Axial View')
#plt.xlabel('z (mm)')
#plt.ylabel('y (mm)')
#plt.legend()
#plt.show()

# In[18]:


plt.figure(figsize=(20,20))
ax = plt.axes(projection='3d')
for layer in layers:
    l = sample[sample.layer_id == layer]
    ax.scatter(l.z, l.x, l.y, s=5, label='Layer '+str(layer), alpha=0.5)
ax.set_xlabel('z (mm)')
ax.set_ylabel('x (mm)')
ax.set_zlabel('y (mm)')
ax.legend()
# These two added to widen the 3D space
ax.scatter(3000,3000,3000, s=0)
ax.scatter(-3000,-3000,-3000, s=0)
plt.show()

# Beautiful graphs! I'll avoid plotting modules because there are way too many.

# ## More Detector Group Inquiry

# Remember: modules make up layers, and layers make up volumes, i.e., module_id is a subdirectory of layer_id, and layer_id is a subdirectory of volume_id. I've jumped the gun here, but cells are the smallest unit of resolution and thus a subdirectory of module_id. 
# We could look at the populations of all of these, too.

# In[26]:


plt.figure(figsize=(30,10))
plt.subplot(1,5,1)
sns.distplot(hits.volume_id)
plt.subplot(1,5,2)
sns.distplot(hits.layer_id)
plt.subplot(1,5,3)
sns.distplot(hits.module_id)
plt.subplot(1,5,4)
sns.distplot(cells.ch0)
plt.subplot(1,5,5)
sns.distplot(cells.ch1)
plt.show()

# If I were to guess I's say low-id layer, modules, cells are closer to the center. Maybe we can do the obvious: plot hits by their radius.

# In[27]:


plt.figure(figsize=(30,10))
radius2 = np.sqrt(hits.x**2 + hits.y**2)
plt.subplot(1,3,1)
sns.distplot(radius2, axlabel='sqrt(x^2 + y^2)')
plt.subplot(1,3,2)
radius3 = np.sqrt(hits.x**2 + hits.y**2 + hits.z**2)
sns.distplot(radius3, axlabel='sqrt(x^2 + y^2 + z^2)')
plt.subplot(1,3,3)
sns.distplot(hits.z**2, axlabel='z')
plt.show()

# And now that we see the general distribution of events are indeed proportional to radius, let's at least plot groups by radius.

# In[28]:


plt.figure(figsize=(30,10))
plt.subplot(1,3,1)
plt.scatter(hits.volume_id, radius2)
plt.xlabel('volume_id')
plt.ylabel('radius')
plt.subplot(1,3,2)
plt.scatter(hits.layer_id, radius2)
plt.xlabel('layer_id')
plt.ylabel('radius')
plt.subplot(1,3,3)
plt.scatter(hits.module_id, radius2)
plt.xlabel('module_id')
plt.ylabel('radius')
plt.show()

# From these:
# * Volumes are named for left, center, right of detector center
# * Layers are named for their radius from the detector center, like onion layers
# * Modules are named for their rotation about the detector center
# 
# We can see this visually in the 3D plots above, too.

# Just checking out group distrubution here:

# In[29]:


hits.volume_id.value_counts()

# In[30]:


hits.layer_id.value_counts()

# In[31]:


hits.module_id.value_counts().head()

# In[32]:


#groups = hits.groupby(['volume_id', 'layer_id', 'module_id'])

# ## Hit Feature Correlations
# 
# Plotting each feature (x, y, z, volume, layer, module) against each other. ~~I've dropped hits_id because that is a scientist-given variable and not physics-relevant.~~

# In[33]:


# Pairplotting 120k hits takes too long, so let's look at a random sampling
#sample = hits.drop('hit_id', axis=1).sample(3000)
sample = hits.sample(3000)
# I've also decided to color-code them by group again. Makes sense to me.
sns.pairplot(sample, hue='volume_id', size=8)
plt.show()

# Qualitatively, there definitely exists correlations between hit features. We can extract the useable ones fairly easily (I think -- I may have done something like this a while ago) but I'll move on for now.
# 
# We can also plot a correlation heatmap. Now _here_ I've dropped hits_id.

# In[34]:


plt.figure(figsize=(10, 10))
hitscorr = hits.drop('hit_id', axis=1).corr()
sns.heatmap(hitscorr, cmap='coolwarm', square=True)
plt.title('Hits Correlation Heatmap')
plt.show()

# I suppose it does make sense that module_id is correlated with layer_id, and layer_id is correlated with volume_id.

# # <a name="cells">Cells</a>
# The cells file contains the constituent active detector cells that comprise each hit. The cells can be used to refine the hit to track association. A cell is the smallest granularity inside each detector module, much like a pixel on a screen, except that depending on the volume_id a cell can be a square or a long rectangle. It is identified by two channel identifiers that are unique within each detector module and encode the position, much like column/row numbers of a matrix. A cell can provide signal information that the detector module has recorded in addition to the position. Depending on the detector type only one of the channel identifiers is valid, e.g. for the strip detectors, and the value might have different resolution.  
# * __hit_id__: numerical identifier of the hit as defined in the hits file.
# * __ch0, ch1__: channel identifier/coordinates unique within one module.
# * __value__: signal value information, e.g. how much charge a particle has deposited.

# In[35]:


cells.head()

# In[36]:


cells.tail()

# In[37]:


cells.describe()

# In[38]:


plt.figure(figsize=(10, 10))
cellscorr = cells.drop('hit_id', axis=1).corr()
sns.heatmap(cellscorr, cmap='coolwarm', square=True)
plt.title('Cells Correlation Heatmap')
plt.show()

# # <a name="particles">Particles</a>
# The particles files contains the following values for each particle/entry:  
# * __particle_id__: numerical identifier of the particle inside the event.
# * __vx, vy, vz__: initial position or vertex (in millimeters) in global coordinates.
# * __px, py, pz__: initial momentum (in GeV/c) along each global axis.
# * __q__: particle charge (as multiple of the absolute electron charge).
# * __nhits__: number of hits generated by this particle.

# In[39]:


particles.head()

# In[40]:


particles.tail()

# In[41]:


particles.describe()

# Some things to noe here:
# * The particle vertex does not necessarily have to be the center of the detector. Perhaps that means it is a secondary particle created later inside the detector, either from a collison or a decay.
# * Charge _q_ is always +1 or -1, not some larger integer. No ions here.
# * nhits can be as low as 0. How? Not sure yet.
# 
# Edit: Thanks to Nadans, nhits can be 0 when they hit no cell (obvously). What was not obvious was why particles that hit nothing would be kept in this simulation. After all, a particle that does not interact is not observed -- though could be inferenced. Anyway, Nadans noted that a particle moving along the beamline has a nonzero chance of not interacting at al, missing all the detectorsl. Let's see if we can prove this.

# In[52]:


nhits0 = particles[particles.nhits ==0]
nhits0_2 = nhits0[nhits0.particle_id == 4503805785800704]
nhits0[['px','py','pz']].describe()

# In[74]:


pt_mean = np.mean(np.sqrt(particles.px**2 + particles.py**2))
pt_mean_unc = np.std(np.sqrt(particles.px**2 + particles.py**2))
pt0_mean = np.mean(np.sqrt(nhits0.px**2 + nhits0.py**2))
pt0_mean_unc = np.std(np.sqrt(nhits0.px**2 + nhits0.py**2))
print('Mean transverse momentum:')
print('    full:', pt_mean, ' +-', pt_mean_unc, 'GeV/c')
print(' nhits=0:', pt0_mean, '+-', pt0_mean_unc)

# From the describe(), the mean z-momentum is actually 10x _less for particles with nhits=0 than the full particles dataset. Compare 0.017 +- 4.2 GeV/c here to 0.1 +- 7.8 Gev/c.  
# From the printout, the mean t-momentum is again smaller, though not as much.  
# Perhaps particles with less total monetum just interact less.
# Are these points significant enough to conclude Nadans hypotheses? Not really. We'll leave this for now.

# Let's plot a couple of histograms:

# In[75]:


plt.figure(figsize=(15,10))
plt.subplot(1,2,1)
particles.q.hist(bins=3)
plt.xlabel('Charge (e)')
plt.ylabel('Counts')
plt.subplot(1,2,2)
particles.nhits.hist(bins=particles.nhits.max())
plt.xlabel('nhits')
plt.show()

# There appears to be about a 20% differenence betweeen positively and negatively charged particles, i.e. there are more positively charged particles. And for nhits, the majority are either 0 or centered around some gaussian at 12. We should eventualy be able to convolve nhits and momentum.

# Back to nhits and momentum... I'd say that nhits is proportional to total momentum p (particle has enough energy (but not too much) to move through enough detectors before stopping). Is this true? Or, assuming a particle does not stop or decay, then it would be dependent on where inside the detector this particle was and where it was going.

# In[76]:


plt.figure(figsize=(10,10))
p = np.sqrt(particles.px**2 + particles.py**2 + particles.pz**2)
plt.scatter(particles.nhits, p)
plt.yscale('log')
plt.xlabel('nhits')
plt.ylabel('Momentum (GeV/c)')
plt.show()

# Wow, I see no real correlation here... For now.
# 
# We can look at a histogram of momentum (note the log~~-log~~ axes):

# In[77]:


plt.figure(figsize=(15,8))
plt.subplot(1,2,1)
#plt.xscale('log')
plt.hist(np.sqrt(particles.px**2 + particles.py**2), bins=100, log=True)
plt.xlabel('Transverse momentum (GeV/c)')
plt.ylabel('Counts')
plt.subplot(1,2,2)
#plt.xscale('log')
plt.hist(particles.pz.abs(), bins=100, log=True)
plt.xlabel('Z momentum (GeV/C)')
plt.show()

# Right now I'm not sure what this tells us about momentum other than most particles have "low" momentum and few have "high" momentum. What about these momentums wrt eachother?

# In[78]:


plt.figure(figsize=(10,10))
plt.scatter(np.sqrt(particles.px**2 + particles.py**2), particles.pz, s=1)
plt.xscale('log')
#plt.yscale('log')
plt.xlabel('Transverse momentum (GeV/c)')
plt.ylabel('Z momentum (GeV/C)')
#plt.zlabel('Counts')
plt.show()

# Man, that one _pz_=500 particle is messing with our plot. Clip it and other outliers:

# In[82]:


p = particles[particles.pz < 200]

plt.figure(figsize=(10,10))
plt.scatter(np.sqrt(p.px**2 + p.py**2), p.pz, s=5, alpha=0.5)
plt.plot([0.1,0.1],[p.pz.min(),p.pz.max()], c='g') # 0.1 instead of 0 because log plot.
plt.plot([0.1,np.sqrt(p.px**2 + p.py**2).max()],[0.1,0.1], c='r', linestyle='--')
plt.xscale('log')
#plt.yscale('log')
plt.xlabel('Transverse momentum (GeV/c)')
plt.ylabel('Z momentum (GeV/C)')
plt.show()

# Particles that lie about the green line were moving in a trajectory perfectly parallel to the beamline. Particles that lie about the red dashed line were moving in a trajectory perfectly transverse to the beamline. From this we can see that move particles were travelling in some sort of cone shape, psuedocoincidentally shown as a cone here. Here I am using the fact that momentum is of course a magnitude and a direction.

# I'm also curious about where these particles start (vertex) and what their momentum is. Is there correlation?

# In[115]:


plt.figure(figsize=(20,10))
plt.subplot(1,2,1)
plt.scatter(particles.vx, particles.vy, s=1)
plt.xlabel('px')
plt.ylabel('py')
plt.subplot(1,2,2)
plt.scatter(particles.vz, particles.vy, s=1)
plt.xlabel('pz')
plt.show()

# In[103]:


plt.figure(figsize=(20,10))
plt.subplot(1,2,1)
plt.scatter(np.sqrt(particles.vx**2+particles.vy**2), np.sqrt(particles.px**2+particles.py**2), s=1)
plt.xlabel('Vertex radius, 2D (mm)')
plt.ylabel('Transverse momentum (GeV/c)')
plt.xscale('log')
plt.yscale('log')
plt.subplot(1,2,2)
plt.scatter(np.sqrt(particles.vx**2+particles.vy**2+particles.vz**2), np.sqrt(particles.px**2+particles.py**2+particles.pz**2), s=1)
plt.xlabel('Vertex radius, 3D (mm)')
plt.ylabel('Total momentum (GeV/c)')
plt.xscale('log')
plt.yscale('log')
plt.show()

# It seems particles of ~all momentums exist at all points. Interesting, though, that particle vertices seem to coincide with detector strips, i.e. they are _created_ not only at the beam but after interacting with the detectors. See below, each particle vertex is unique and not just 1 of 10 hits, e.g..

# In[99]:


len(particles), len(particles.particle_id.unique())

# In[83]:


plt.figure(figsize=(10, 10))
particlescorr = particles.drop('particle_id', axis=1).corr()
sns.heatmap(particlescorr, cmap='coolwarm', square=True)
plt.title('Particles Correlation Heatmap')
plt.show()

# # <a name="truth">Truth</a>
# The truth file contains the mapping between hits and generating particles and the true particle state at each measured hit. Each entry maps one hit to one particle.  
# * __hit_id__: numerical identifier of the hit as defined in the hits file.
# * __particle_id__: numerical identifier of the generating particle as defined in the particles file. A value of 0 means that the hit did not originate from a reconstructible particle, but e.g. from detector noise.
# * __tx, ty, tz__ true intersection point in global coordinates (in millimeters) between the particle trajectory and the sensitive surface.
# * __tpx, tpy, tpz__ true particle momentum (in GeV/c) in the global coordinate system at the intersection point. The corresponding vector is tangent to the particle trajectory at the intersection point.
# * __weight__ per-hit weight used for the scoring metric; total sum of weights within one event equals to one.

# In[84]:


truth.head()

# In[85]:


truth.tail()

# In[86]:


# Just checking out one example
#truth[truth.particle_id == 22525763437723648]

# In[87]:


# Number of unique particles
len(truth.particle_id.unique())

# ## Let's plot some particle tracks

# In[88]:


# Get every 100th particle
tracks = truth.particle_id.unique()[1::100]

plt.figure(figsize=(15,15))
ax = plt.axes(projection='3d')
for track in tracks:
    t = truth[truth.particle_id == track]
    ax.plot3D(t.tz, t.tx, t.ty)
ax.set_xlabel('z (mm)')
ax.set_ylabel('x (mm)')
ax.set_zlabel('y (mm)')
# These two added to widen the 3D space
ax.scatter(3000,3000,3000, s=0)
ax.scatter(-3000,-3000,-3000, s=0)
plt.show()

# We can actually see that indeed many particles do not start at the detector center, and originate somewhere else. We can also see that trajectories are more (visibly) helical the less z-momentum they have.

# In[ ]:



