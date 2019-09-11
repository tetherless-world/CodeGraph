#!/usr/bin/env python
# coding: utf-8

# # Which proteins come together?
# 
# Take a look at this wonderful image of an animal cell provided by LadyofHats (Mariana Ruiz) in the public domain via Wikimedia Commons. We can see some of our target proteins and may conclude that some of them are likely to occur together. In our images all of them should be present but only one of them are stained in the green channel. Now let's assume that staining is sometimes not that easy and it happens that multible organelles are likely to be stained together. If this is true we could expect groupings - some targets may be likely to be one-hot at the same time over a broader range of image samples in our data set.
# 
# As the target distribution and dependencies are an entry point to setup an objective or loss function, it could be worth it to dive into dive with me into target group analysis using a latent variable model. 
# 
# **If you like my kernel** you can make be very happy with an **upvote and/or comment** ;-)! The motivation I gain out of your feedback pushes me to share my ideas instead of hiding them. Thank you!

# ![Animal cell organelles](https://upload.wikimedia.org/wikipedia/commons/4/48/Animal_cell_structure_en.svg)

# ## What can you find within this kernel?
# 
# 1. A motivation why there is a hint that target groups exist in our data. 
# 2. A short explanation what is a bernoulli mixture model, how it learns and its implementation.
# 3. An overview of target groups found by clustering with this model. 
# 4. Some explorations related to certainty of cluster assignment and target combination anomalies.
# 5. A conclusion 
# 
# Let's go! :-)

# # Preliminary Work
# 
# As usual - loading packages:

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns
sns.set()
import matplotlib.pyplot as plt
from PIL import Image
from scipy.misc import imread

import tensorflow as tf


import os
print(os.listdir("../input"))

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# and target information data given by train.csv:

# In[ ]:


train_labels = pd.read_csv("../input/train.csv")

# After extracting the labels per id with zero-hot-encoding:

# In[ ]:


label_names = {
    0:  "Nucleoplasmn",  
    1:  "Nuclear membrane",   
    2:  "Nucleoli",   
    3:  "Nucleoli fibrillar center",   
    4:  "Nuclear speckles",
    5:  "Nuclear bodies",   
    6:  "Endoplasmic reticulum",   
    7:  "Golgi apparatus",   
    8:  "Peroxisomes",   
    9:  "Endosomes",   
    10:  "Lysosomes", 
    11:  "Intermediate filaments",   
    12:  "Actin filaments",   
    13:  "Focal adhesion sites",   
    14:  "Microtubules",   
    15:  "Microtubule ends",   
    16:  "Cytokinetic bridge",   
    17:  "Mitotic spindle",   
    18:  "Microtubule organizing center",   
    19:  "Centrosome",   
    20:  "Lipid droplets",   
    21:  "Plasma membrane",   
    22:  "Cell junctions",   
    23:  "Mitochondria",   
    24:  "Aggresome",   
    25:  "Cytosol",   
    26:  "Cytoplasmic bodies",   
    27:  "Rods & rings"
}

reverse_train_labels = dict((v,k) for k,v in label_names.items())

def fill_targets(row):
    row.Target = np.array(row.Target.split(" ")).astype(np.int)
    for num in row.Target:
        name = label_names[int(num)]
        row.loc[name] = 1
    return row

# In[ ]:


for key in label_names.keys():
    train_labels[label_names[key]] = 0

train_labels = train_labels.apply(fill_targets, axis=1)

# In[ ]:


train_labels.head()

# We are ready to start! :-D

# # Why do proteins come together?

# In[ ]:


target_counts = train_labels.drop(["Id", "Target"],axis=1).sum(axis=0).sort_values(ascending=False)
plt.figure(figsize=(15,15))
sns.barplot(y=target_counts.index.values, x=target_counts.values, order=target_counts.index);

# This is already known! We have some very seldom targets like rods & rings and very common ones like nucleoplasmn, cytosol and plasma membrane. But wouldn't it be nice to know if some proteins are likely to come together? We have already seen that lysosomes, endosomes and endoplasmatic reticulum have target correlations:
# 
# 

# In[ ]:


train_labels["number_of_targets"] = train_labels.drop(["Id", "Target"],axis=1).sum(axis=1)

def find_counts(special_target, labels):
    counts = labels[labels[special_target] == 1].drop(
        ["Id", "Target", "number_of_targets"],axis=1
    ).sum(axis=0)
    counts = counts[counts > 0]
    counts = counts.sort_values()
    return counts

lyso_endo_counts = find_counts("Lysosomes", train_labels)

plt.figure(figsize=(15,5))
sns.barplot(x=lyso_endo_counts.index.values, y=lyso_endo_counts.values, palette="Blues");

# **You can see that lysosomes and endosomes always come together and that it is likely that the endoplasmatic reticulum is present as well**. I have read about lysosomes that their membrane is produced by ribosomes that are located in the rough endoplasmatic reticulum and transferred to the golgi apparatus afterwards. Later they are shipped to endosomes, small bubbles full of stuff, and seem to fuse with them to digest all the stuff inside. What if staining is done with molecules that are found in all three participants: the rough ER, endosomes and lysosomes? Then we will always see that they often come together. 
# 
# But isn't it nice to know? If our model is sure that lysosomes are present, we can automatically say endosomes is hot as well. Great! Hence instead of predicting both target classes we can reduce to both to one single lyso-endo class.

# In[ ]:


count_perc = np.round(100 * train_labels["number_of_targets"].value_counts() / train_labels.shape[0], 2)
plt.figure(figsize=(20,5))
sns.barplot(x=count_perc.index.values, y=count_perc.values, palette="Reds")
plt.xlabel("Number of targets per image")
plt.ylabel("% of data");

# Here you can see that most of the target proteins per image come alone or as a pair. Hopefully we can find nice pair-structures in our data. We will see... 

# In[ ]:


targets = train_labels.drop(["Id", "Target", "number_of_targets"], axis=1)

# # How can we uncover hidden protein groups?
# 
# The presence of target proteins is given by a binary, discrete value:
# 
# * 0 for absence
# * 1 for presence 
# 
# As I like to find groupings or clusters of these **discrete quantities** I liked to use a discrete clustering algorithm. As I'm currently on the way of my learning path to discover mixture models the choice using a Bernoulli mixture model was an easy one.  What's the idea behind it?
# 
# *If you are not interested in mixture models and math etc. skip this chapter and jump to results analysis ;-) *
# 
# Mixture models assume that the data we **observed $X$ was generated by some latent variable $Z$.** In our case this could be the **staining process** of microscope preparates used to obtain the 4 channel images. Perhaps this process is limited and this limitation has caused our target groupings. You have already seen that we were able to reduce the number of target classes by fusing endosomes and lysosomes. Consequently we can say that there are fewer configurations of the latent variable than target classes. 
# 
# 
# ### Model description
# 
# These configurations can be seen as $K$ components of the mixture model. We don't know how many of them are actually there and we will have to estimate them during the analysis. Each component tries to explain one target group we are seeking for. And for each sample $x_{n}$ of our $N$ data spots there exists a related latent or hidden variable $z_{n}$ that holds 1 for the component $k$ that generated $x_{n}$ and 0 for all others. Imagine you would already know them, then we could describe the probability density our data as follows:
# 
# $$ P(X) = \sum_{Z} P(X, Z|\theta |\theta) = \sum_{Z} P(Z|\theta) \cdot P(X|Z, \theta)$$
# 
# If you are not very familiar with probabilities: To obtain this equation you need to know the sum and product rules of probabilites. We are now summing over all $K$ configurations of the latent variable Z, hence over $K$ assumed clusters. This formulation is very general. Now image you take a look at one of the components, on one single cluster. How is the target protein data distributed in that group?
# 
# <a title="Classical Numismatic Group, Inc. http://www.cngcoins.com [GFDL (http://www.gnu.org/copyleft/fdl.html), CC-BY-SA-3.0 (http://creativecommons.org/licenses/by-sa/3.0/) or CC BY-SA 2.5 (https://creativecommons.org/licenses/by-sa/2.5)]" href="https://commons.wikimedia.org/wiki/File:Ephesos_620-600_BC.jpg"><img width="256" alt="Ephesos 620-600 BC" src="https://upload.wikimedia.org/wikipedia/commons/4/4f/Ephesos_620-600_BC.jpg"></a>
# 
# Each target protein $x_{d}$ itself is binary. If we like to describe the distribution of nucleoplasmn it is like one we would obtain by tossing a coin. And for each of the $D=28$ target protein we could flip such a coin, with zero on one side and one on the other. If we assume that  the target proteins are independent within one component $k$ than we can write:
# 
# $$ p(x_{n}|\mu_{k}) = \prod_{d=1}^{D} \mu_{k,d}^{x_{n,d}} (1-\mu_{k,d})^{(1-x_{n,d})}$$
# 
# Whereas $\mu_{k,d}$ stands for the probability to observe the current target protein $x_{n,d}$. This sounds difficult first but let's try to understand it with lysosomes and endosomes again: If we know that they belong to group component $k=2$, for example. Then the probability to observe lysosomes and endosomes within this group could perhaps be $\mu_{2,lyso} = \mu_{2,endo} =  0.99$ for both and $\mu_{2,ER} = 0.4$ for endoplasmatic reticulum. For all other proteins the probability to observe them should be very low, for example $\mu_{2, k} = 0.01.$ If we now have a sample that fits to this target combination it would yield a high value for $p(x_{n}| \mu_{2})$. Hence it's probable that it belongs to that group.
# 
# Ok, we are close to fullfill our model decription. We try to explain our data by the probability density $P(X)$. Imagine that all samples were independently drawn from this distribution. With this assumption we can split into a product over all $N$ samples. 
# 
# $$ P(X) = \sum_{Z} P(X, Z|\theta) = \sum_{Z} P(Z|\theta) \cdot P(X|Z, \theta) = \prod_{n} \sum_{z} p(z|\pi) \cdot p(x|z, \mu)$$
# 
# As there exists one true component $z_{n,k}$ for each sample, we can say that $z_{n}$ is one-hot-encoded and can be described by a multinomial distribution:
# 
# $$p(z|\pi) = \prod_{k=1}^{K} \pi_{k}^{z_{k}}$$
# 
# The same holds for the conditional probability $p(x|z, \mu)$:
# 
# $$p(x|z, \mu) = \prod_{k=1}^{K} p(x|\mu_{k})^{z_{k}}$$
# 
# and finally we obtain: 
# 
# $$ P(X) = \prod_{n} \sum_{z} p(z|\pi) \cdot p(x|z, \mu)  = \prod_{n} \sum_{k}^{K} \pi_{k} \cdot p(x_{n}|\mu_{k}) = \prod_{n} \sum_{k=1}^{K} \pi_{k} \cdot \prod_{d=1}^{D} \mu_{k,d}^{x_{n,d}} (1-\mu_{k,d})^{(1-x_{n,d})} $$
# 
# With this model we like to describe our target data. And to make this model fit to what we observe we will maximize this probability density $p(X)$ with respect to our model parameters $\pi$ and $\mu$. This maximization is usually performed by taking the log first as it often makes it simpler to take derivarives. But in our case we will stuck....
# 
# $$ \ln p(X|\mu, \pi) = \sum_{n=1}^{N} \ln \left( \sum_{k=1}^{K} \pi_{k} \cdot p(x_{n}|\mu_{k}) \right) $$ 
# 
# Do you see the problem? The sum over all $K$ components prevents the log to act on $\pi_{k} \cdot p(x_{n}|\mu_{k})$ and consequently we can't make the bernoullis tractable for taking derivatives. :-(  

# ### Expectation maximization
# 
# $$ \ln p(X,Z|\mu, \pi) = \sum_{n=1}^{N} \sum_{k=1}^{K} \gamma (z_{n,k}) \left( \ln \pi_{k} + \sum_{d=1}^{D} x_{n,d} \ln \mu_{k,d} + (1- x_{n,d}) \ln (1-\mu_{k,d}) \right)$$
# 
# Now the hero comes: Instead of maximizing the log-likelihood we are going to act as if we already know our parameters $\pi_{k}$ and $\mu_{k}$ by initializing them randomly.  
# 
# #### E-Step: responsibilites
# 
# Then we compute how responsible each component of our mixture $k$ was to generate the data spot $x_{n}$ and its targets. This is like a soft assigment to cluster components. The one cluster with the highest responsibility would be the winner for cluster assignment. 
# 
# $$ \gamma( z_{n,k} ) = \frac{\pi_{k} p(x_{n}|\mu_{k})} {\sum_{j=1}^{K} \pi_{j} p(x_{n}|\mu_{j})} $$
# 
# #### M-Step: maximization
# 
# Now we know how responsible each component is for generating data point $x_{n}$ and given this information we can recalculate the parameters we initialized randomly at the starting point.
# 
# $$ N_{k} = \sum_{n=1}^{N} \gamma(z_{n,k})$$
# 
# $$\mu_{k} = \frac{1}{N_{k}} \sum_{n=1}^{N} \gamma(z_{n,k}) x_{n}$$
# 
# $$\pi_{k} = \frac{N_{k}}{N} $$
# 
# You may wonder where do these nice equations come from?! To understand it, you need to know more about expectation maximization and I don't like to blow up this kernel even more with math. So if you like, you should dive deeper into this topic by reading books or watching videos in the orbit out there. ;-)
# 
# Performing E- and M-Step iteratively one step after another we will end up with a nice fit of our model to the target protein data.
# 

# ### Implementation of the model

# In[ ]:


from scipy.special import logsumexp

class BernoulliMixture:
    
    def __init__(self, n_components, max_iter, tol=1e-3):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
    
    def fit(self,x):
        self.x = x
        self.init_params()
        log_bernoullis = self.get_log_bernoullis(self.x)
        self.old_logL = self.get_log_likelihood(log_bernoullis)
        for step in range(self.max_iter):
            if step > 0:
                self.old_logL = self.logL
            # E-Step
            self.gamma = self.get_responsibilities(log_bernoullis)
            self.remember_params()
            # M-Step
            self.get_Neff()
            self.get_mu()
            self.get_pi()
            # Compute new log_likelihood:
            log_bernoullis = self.get_log_bernoullis(self.x)
            self.logL = self.get_log_likelihood(log_bernoullis)
            if np.isnan(self.logL):
                self.reset_params()
                print(self.logL)
                break

    def reset_params(self):
        self.mu = self.old_mu.copy()
        self.pi = self.old_pi.copy()
        self.gamma = self.old_gamma.copy()
        self.get_Neff()
        log_bernoullis = self.get_log_bernoullis(self.x)
        self.logL = self.get_log_likelihood(log_bernoullis)
        
    def remember_params(self):
        self.old_mu = self.mu.copy()
        self.old_pi = self.pi.copy()
        self.old_gamma = self.gamma.copy()
    
    def init_params(self):
        self.n_samples = self.x.shape[0]
        self.n_features = self.x.shape[1]
        #self.gamma = np.zeros(shape=(self.n_samples, self.n_components))
        self.pi = 1/self.n_components * np.ones(self.n_components)
        self.mu = np.random.RandomState(seed=0).uniform(low=0.25, high=0.75, size=(self.n_components, self.n_features))
        self.normalize_mu()
    
    def normalize_mu(self):
        sum_over_features = np.sum(self.mu, axis=1)
        for k in range(self.n_components):
            self.mu[k,:] /= sum_over_features[k]
            
    def get_responsibilities(self, log_bernoullis):
        gamma = np.zeros(shape=(log_bernoullis.shape[0], self.n_components))
        Z =  logsumexp(np.log(self.pi[None,:]) + log_bernoullis, axis=1)
        for k in range(self.n_components):
            gamma[:, k] = np.exp(np.log(self.pi[k]) + log_bernoullis[:,k] - Z)
        return gamma
        
    def get_log_bernoullis(self, x):
        log_bernoullis = self.get_save_single(x, self.mu)
        log_bernoullis += self.get_save_single(1-x, 1-self.mu)
        return log_bernoullis
    
    def get_save_single(self, x, mu):
        mu_place = np.where(np.max(mu, axis=0) <= 1e-15, 1e-15, mu)
        return np.tensordot(x, np.log(mu_place), (1,1))
        
    def get_Neff(self):
        self.Neff = np.sum(self.gamma, axis=0)
    
    def get_mu(self):
        self.mu = np.einsum('ik,id -> kd', self.gamma, self.x) / self.Neff[:,None] 
        
    def get_pi(self):
        self.pi = self.Neff / self.n_samples
    
    def predict(self, x):
        log_bernoullis = self.get_log_bernoullis(x)
        gamma = self.get_responsibilities(log_bernoullis)
        return np.argmax(gamma, axis=1)
        
    def get_sample_log_likelihood(self, log_bernoullis):
        return logsumexp(np.log(self.pi[None,:]) + log_bernoullis, axis=1)
    
    def get_log_likelihood(self, log_bernoullis):
        return np.mean(self.get_sample_log_likelihood(log_bernoullis))
        
    def score(self, x):
        log_bernoullis = self.get_log_bernoullis(x)
        return self.get_log_likelihood(log_bernoullis)
    
    def score_samples(self, x):
        log_bernoullis = self.get_log_bernoullis(x)
        return self.get_sample_log_likelihood(log_bernoullis)

# ### Estimating the number of components
# 
# I don't know how many target groups are there in advance. But in contrast to hard clustering algorithms like k-means we can use a test set to tune the number of components to choose as a hyperparameter. 

# In[ ]:


from sklearn.model_selection import train_test_split

X = targets.values
x_train, x_test = train_test_split(X, shuffle=True, random_state=0)

# Let's try out some values:

# In[ ]:


components_to_test = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

# And fit multiple models... 

# In[ ]:


scores = []


for n in range(len(components_to_test)):
    if n > 0:
        old_score = score
    model = BernoulliMixture(components_to_test[n], 200)
    model.fit(x_train)
    score = model.score(x_test)
    scores.append(score)
    if n > 0: 
        if score < old_score:
            estimated_components = components_to_test[n-1]
            break
        

# In the end, we obtain that these number of components was nice for train and test:

# In[ ]:


estimated_components

# To obtain results to work with, let's refit out model with this estimated number of components:

# In[ ]:


model = BernoulliMixture(estimated_components, 200)
model.fit(X)

# In[ ]:


results = targets.copy()
results["cluster"] = np.argmax(model.gamma, axis=1)

# ## What kind of target groups are found?

# ### How are specific proteins distributed over all clusters in percent?

# In[ ]:


grouped_targets = results.groupby("cluster").sum() / results.drop("cluster", axis=1).sum(axis=0) * 100
grouped_targets = grouped_targets.apply(np.round).astype(np.int32)

plt.figure(figsize=(20,15))
sns.heatmap(grouped_targets, cmap="Blues", annot=True, fmt="g", cbar=False);
plt.title("How are specific proteins distributed over clusters in percent?");

# ### Take-Away
# 
# * This looks great! You can see that several clusters only hold one specific target protein!
# * For each target protein you can see the percentage of its occurences that are placed into specific clusters.
# * One example: 97 % of Mitochondria target proteins are located in cluster 18. Only a few percents are hold by cluster 12 and 8. There is one percent missing to fill up to 100 % but this is caused by rounding errors and should not worry you.
# * We find that **a lot of cellular components are roughly hold by their own clusters**: 
#     * Nuclear speckles
#     * Endoplasmatic reticulum, Lysosomes and Endosomes
#     * Golgi apparatus
#     * Intermediate filaments
#     * Microtubules
#     * Lipid Droplets
#     * Centrosomes
#     * Plasma membrane
#     * Mitochondria
# * In addition we find **meaningful combinations of cellular components**:
#    * Focal adhesion sides come together with actin fliaments and are spread over 3 clusters that are very similar up to absence or presence of plasma membrane, nuclear membrane and cell junctions. 
#     * Aggresomes come alone or sometimes with cytoplasmic bodies. This makes sense: If bodies consist of viral capsids then there could be a lot of garbage or clutted proteins as well either due to immune response or to the viral building process. 
#     * Cellular devision often comes in groups of either microtubules & mitotic spindle and cytokinetic bridge or as microtubule organizing center & mitotic spindle and cytokinetic bridge.
#     * As biologist one may see much more!
# * **Some components are spread over several clusters** like nucleoplasmn.
#  

# ### Give clusters a name
# 
# Naming clusters will make it easier to understand the patterns. I like to do this given the information of the blue map that show us in which cluster one can find a specific target protein. This information does not suffer under the target imbalance problem and show nice couplings between targets.  

# In[ ]:


cluster_names = {
    0: "Actin filaments & Focal adhesion sites",
    1: "Aggresomes",
    2: "Microtubules, Mitotic spindle, Cytokinetic Bridge",
    3: "RodsRings, Microtubule ends, Nuclear bodies",
    4: "Some nuclear membranes",
    5: "various - RodsRings",
    6: "various - Mitotic spindle, Organizing center, Cytosol",
    7: "Nuclear bodies & Aggresomes",
    8: "Nuclear speckles",
    9: "Nuclear membrane & Actin filaments & Focal adhesion sites",
    10: "Endoplasmatic reticulum & Endosomes & Lysosomes",
    11: "Low dense 1",
    12: "Mitotic spindle, Organizing center",
    13: "Plasma membrane & various",
    14: "Nucleoli fibrillar center & Peroxisomes",
    15: "Low dense 2",
    16: "Nucleoli fibrillar center & Cytoplasmic bodies",
    17: "Nucleoli & Microtubule ends & Peroxisomes & Rods Rings",
    18: "Mitochondria & Lipid droplets & RodsRings & Nucleoli",
    19: "Low dense 3",
    20: "Golgi apparatus",
    21: "Intermediate filaments",
    22: "Centrosome",
    23: "Cytoplasmic bodies & Aggresomes",
    24: "Lipid droplets & Peroxisomes & Cell junctions"
}

# ### How many targets are hot within one cluster?
# 
# Let's go one step further. We already know that cluster 10 holds all samples of endoplasmatic reticulum, lysosomes and endosomes. But we don't know how present each target is given only the targets of one cluster. 

# In[ ]:


cluster_size = results.groupby("cluster").Nucleoplasmn.count()
cluster_composition = results.groupby("cluster").sum().apply(lambda l: l/cluster_size, axis=0) * 100
cluster_composition = cluster_composition.apply(np.round).astype(np.int)

cluster_composition = cluster_composition.reset_index()
cluster_composition.cluster = cluster_composition.cluster.apply(lambda l: cluster_names[l])
cluster_composition = cluster_composition.set_index("cluster")

plt.figure(figsize=(20,20))
sns.heatmap(cluster_composition, cmap="Oranges", annot=True, fmt="g", cbar=False);
plt.title("How present alias hot are specific targets within one cluster?");
plt.ylabel("");

# ### Take-Away
# 
# * That's great! This may yields insights what kind of targets we will always find per cluster. Take a look at the endoplasmatic reticulum, lysosome, endosome cluster. You can see that almost all samples of that cluster hold the endoplasmatic reticulum with 1. In contrast due to the seldomness of lysosomes and endosomes the hotness of them is very low. A lot of samples do not have them even though all targets of lysosomes and endosomes can be find within that cluster!!!
# * Well you might ask yourself now, why this map helps you: Take a look at the aggresomes for example. It's a seldom target but once you have found the cluster 1 you can definitely say that all of its samples have an aggresome present. :-) Hence if seldom targets have their own nice, little cluster and you have found it, you are done with its targets. 

# ### Take away
# 
# * This map seems to be more difficult to understand than the one before.
# * It's highly influenced by the frequency of target proteins and **yields more insights of the imbalance of target proteins per cluster**. Let's try to collect important insights! :-)
# * First of all it comes out very clear that **nucleoplasmn and cytosol are the dominating targets** that influence the composition of many clusters. 
# * Now let's come to our example of endoplasmatic reticulum, lysosomes and endosomes: You can see that the cluster that hold these target proteins is mainly occupied by endoplasmatic reticulum followed by cytosol. Lysosomes and endosomes in contrast only appear with very small percentages of 3 and 2 % due to their seldomness. **Hence even though they are all located in this cluster, they are seldom in this cluster as well and not only in the overall data!**
# * This situation is even more worse for Rods and Rings and Microtubule ends. Due to their seldomness they are overwhelmed by more common classes like nucleoplasm, nucleoli and nuclear bodies. 
# * Luckily we can see some **nice clusters that are more specific to seldom targets** like that with lipid droplets, cell junctions and peroxisomes. 
# * Same holds for aggresomes and aggresomes with cytoplasmic bodies. 

# In[ ]:


results["cluster_names"] = results.cluster.apply(lambda l: cluster_names[l])

# Both maps already yielded some insights which targets are likely to come together and how seldom a target is given a cluster. We can go one step further. The model was trained with $\mu$ and $\pi$. The latter shows us the probability per cluster: 

# ### How does the prior probability $\pi_{k}$ per cluster look like?

# In[ ]:


cluster_ids = np.arange(0, estimated_components)
names = [cluster_names[l] for l in cluster_ids]

pi = pd.Series(data=model.pi, index=names).sort_values(ascending=False)
plt.figure(figsize=(20,5))
sns.barplot(x=pi.index, y=pi.values, palette="Reds_r", order=pi.index)
plt.xticks(rotation=90);

# ### Take-away
# 
# Uhh, that's interesting! I haven't thought that the cluster with mitochondria has the highest prior probability. I expected cluster 5 and 6 to be most probable:
# 
# * 5: "various - RodsRings"
# * 6: "various - Mitotic spindle, Organizing center, Cytosol"
# 
# Both have high counts of nucleoplasmn and cytosol, hence most common targets. Even though Mitochondria with Nucleoli are at the top we can see that clusters that hold most common targets have higher prior probabilities.

# ## How do the target probabilities $\mu_{kd}$ look like given the cluster?

# In[ ]:


model.mu.shape

# In[ ]:


mu = pd.DataFrame(data=model.mu * 100, index=names, columns=results.drop(["cluster", "cluster_names"], axis=1).columns.values)
mu = mu.apply(np.round)

plt.figure(figsize=(20,20))
sns.heatmap(mu, cmap="Purples", annot=True, fmt="g", cbar=False)

# ### Take-Away
# 
# * We can see that the probability of each target to be present yields a lot of insights. Let's consider the golgi apparatus for example. If we choose the corresponding cluster named the same we can observe a probability to occur of 100%. Hence if we now that a sample is located within that cluster, we can say it has a golgi apparatus!
# * There a lot of target proteins that have a very high probability to occur in their cluster!
# * But there are some strange probabilities as well. Think of cluster 1 again: It's nearly fully occupied by aggresomes but the mu-proability for this target given this cluster is very low with 18 %. That's strange! It's related to the seldomness of the target and could be caused by a shift due to weighted sum during mu calculation. As the sum is taken over all samples, nearby targets (of other components) that still have some higher responsibility for the aggresome cluster can cause a shift of the mu-center of the cluster. 

# ## How many samples do the clusters hold?

# In[ ]:


cluster_counts = results.groupby("cluster").cluster.count()
cluster_counts = cluster_counts.sort_values()
names = [cluster_names[num] for num in cluster_counts.index]

plt.figure(figsize=(20,5))
sns.barplot(x=names, y=cluster_counts.values, order=names)
plt.xticks(rotation=90);

# ### Take-Away
# 
# * We can see that the clusters are occupied by very different amounts of samples.
# * This fits to what we have observed: Small clusters are mainly responsible for seldom targets and do from nice little groups.
# * In contrast big clusters hold various of different target proteins and perhaps clustering was not well for these samples. 

# ## How many multilabel target  do the clusters hold?

# In[ ]:


results["number_of_targets"] = results.drop("cluster", axis=1).sum(axis=1)

multilabel_stats = results.groupby("cluster_names").number_of_targets.value_counts() 
multilabel_stats /= results.groupby("cluster_names").number_of_targets.count()
multilabel_stats = multilabel_stats.unstack()
multilabel_stats.fillna(0, inplace=True)
multilabel_stats = 100 * multilabel_stats
multilabel_stats = multilabel_stats.apply(np.round)
multilabel_stats = multilabel_stats.astype(np.int)

plt.figure(figsize=(20,5))
sns.heatmap(multilabel_stats.transpose(),
            square=True,
            cbar=False,
            cmap="Greens",
            annot=True);

# ### Take-Away
# 
# * First of all we can see that the low dense clusters all have multiple targets. This can be the reason why their samples were not assigned to other clusters that would fit but are different in the number of total targets.
# * In addition we can confirm what we already know: Many samples only have one or two target proteins present, not more! This is especially interesting for clusters that hold various components. Often these clusters have relations to the cellular devision process.
# * There are only two clusters that have at least two targets present:
#     * Nucleoli fibrillar center & Cytoplasmic bodies
#     * Rods & Rings, Microtubule ends, Nuclear bodies 

# ## Which clusters are anomalistic?

# In[ ]:


sample_logLs = model.score_samples(X)

# For each sample we can try to find out how dense the region of the target feature space is, hence how many other samples are around it with the same kind of target structure. This density is given by the sample log-likelihood. 

# In[ ]:


my_threshold = np.quantile(sample_logLs, 0.05)

# To define which samples are anomal, I will setup a threshold of 5 %. If we compare this threshold with the distribution of the sample log-likelihoods, you can see...

# In[ ]:


plt.figure(figsize=(20,5))
sns.distplot(sample_logLs)
plt.axvline(my_threshold, color="Red")
plt.xlabel("Sample log likelihood of bernoulli mixture")
plt.title("Choosing a threshold to detect anomalies")
plt.ylabel("Density")

# ... that below this threshold there are some samples in the negative regim (< -8)  that are truely outliers compared to the other ones (> -8).

# In[ ]:


results["anomaly"] = np.where(sample_logLs <= my_threshold, 1, 0)

# ### How anomalistic are the clusters?

# In[ ]:


anomalies = results.groupby("cluster_names").anomaly.value_counts() / results.groupby("cluster_names").cluster.count() * 100
anomalies = anomalies.unstack()
anomalies.fillna(0, inplace=True)
anomalies = anomalies.apply(np.round)
anomalies = anomalies.astype(np.int)

plt.figure(figsize=(20,5))
sns.heatmap(anomalies.transpose(), cmap="Reds", annot=True, square=True, cbar=False);

# ### Take-Away
# 
# * I expected the low-dense clusters to be more anomalistic than the others but this seems to be only true for low-dense 3 and a bit for low dense 1.
# * Interestingly some other clusters have anomalies as well. This is especially the case for clusters that tend to have only one target protein but have some with 2 target proteins as well. How anomal a cluster may depend on the total number of targets of the samples in that cluster.
# * In contrast the various Rods&Rings cluster that have uncertain cluster assignments (look at the chapter below) is not located in a low dense region. Hence there are other clusters and targets around that could suite as well. 

# ## How certain are the cluster assignments?

# In[ ]:


results["certainty"] = np.sort(model.gamma, axis=1)[:,-1]

# In[ ]:


certainties = results.certainty.values

plt.figure(figsize=(20,5))
sns.distplot(certainties, color="Orange")
plt.xlabel("Certainty of cluster assignment")
plt.ylabel("Density")
plt.title("How sure was the model in predicting the winner?");

# ### Take-Away
# 
# * The certainties vary on a broad range. This suggests that we might have further cluster interactions or outlier target samples.
# * Let's have a look at the certainty distribution and statistics per cluster:

# In[ ]:


plt.figure(figsize=(20,5))
sns.boxplot(x="cluster_names", y="certainty", data=results)
plt.ylim([0,1])
plt.xticks(rotation=90)
plt.xlabel("");

# ### Take-Away
# 
# * Ah! Cool! Take a look at the various - RodsRings cluster. The model was very uncertain in its cluster assignments. If you like to take a look at the blue map, you can see that a lot of targets are present which correspond to the cellular devision process. I think we should consider the alternative clusters for this cluster, this are those that yielded a high probability to be responsible as well (that's gamma in the bernoulli model). The same should be done for the cluster that solely holds nuclear membranes.
# * In addition the occurence of seldom target proteins seems to make a cluster assignment more uncertain. The aggresome cluster for example holds 50 % of all aggresome proteins in the competition, but this cluster there is a more common target - nucleoplasmn - as well. If you take a look at the probability $\mu$ for Aggresomes in that cluster it's quite low with 18. Perhaps this cluster is not nice shaped and occupied. We don't know, how many nucleoplasmn targets come alone in that cluster or how many are coupled with aggresomes. Let's make a map for this case: 

# In[ ]:


aggresome_cluster = results.loc[results.cluster==1].drop(["cluster",
                                 "cluster_names",
                                 "number_of_targets", 
                                  "anomaly", 
                                 "certainty"], axis=1).copy()

counts = aggresome_cluster.sum()
columns_of_interest = list(counts[counts>0].index.values)
aggresome_cluster = aggresome_cluster.loc[:,columns_of_interest]

aggresome_combinations = pd.DataFrame(index=aggresome_cluster.columns.values,
                                      columns=aggresome_cluster.columns.values)

for col in aggresome_combinations.columns:
    aggresome_combinations.loc[col,:] = aggresome_cluster[aggresome_cluster[col] == 1].sum()

mask = np.zeros_like(aggresome_combinations, dtype=np.bool)
mask[np.triu_indices_from(mask, k=1)] = True


plt.figure(figsize=(8,8))
sns.set(style="white")
sns.heatmap(aggresome_combinations, mask=mask, cmap="Reds",
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, vmin=0, vmax=50, annot=True, fmt="g")

# ### Take-Away
# 
# * Ok, a lot of aggresomes seem to come alone as they have a lot of counts with themselves but only few counts for other target proteins.
# * Samples with two targets present are probably given by Nucleoplasmn-Aggresome combinations as this is the second highest count in the map.
# * By only considering the aggresome row we can see that there are some samples that are coupled with cytokinetic bridge or cell junctions with have themselves some couplings with the nucleoplasm. This cases are the 3-couped-targets of that cluster. 
# * As there are no 4-coupled-targets we can conclude that there are some combinations that do not really fit to the aggresome cluster. One example: Take a look at the aggresome row again and the microtubule organizing center column. You can see that there are zero couplings of aggresomes with this target. But you can see that the organizing center has couplings with cell junctions and nucleoplasmn. Consequently some of the 2- or 3-couplings could be given by such combinations without aggresomes. 

# In[ ]:


aggresome_cluster.shape

# Considering the total number of samples in the cluster given by 183, we can say, that there are 7 samples that do not have an aggresome!

# In[ ]:


af_cluster = results.loc[results.cluster==0].drop(["cluster",
                                 "cluster_names",
                                 "number_of_targets", 
                                  "anomaly", 
                                 "certainty"], axis=1).copy()

counts = af_cluster.sum()
columns_of_interest = list(counts[counts>0].index.values)
af_cluster = af_cluster.loc[:,columns_of_interest]

af_combinations = pd.DataFrame(index=af_cluster.columns.values,
                               columns=af_cluster.columns.values)

for col in af_combinations.columns:
    af_combinations.loc[col,:] = af_cluster[af_cluster[col] == 1].sum()

mask = np.zeros_like(af_combinations, dtype=np.bool)
mask[np.triu_indices_from(mask, k=1)] = True


plt.figure(figsize=(8,8))
sns.set(style="white")
sns.heatmap(af_combinations, mask=mask, cmap="Reds",
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, vmin=0, vmax=50, annot=True, fmt="g")

# ## Which kind of cluster interactions do we have?

# In[ ]:


results["alternative_cluster"] = np.argsort(model.gamma, axis=1)[:,-2]
results["alternative_names"] = results.alternative_cluster.apply(lambda l: cluster_names[l])
results["alternative_certainties"] = np.sort(model.gamma, axis=1)[:,-2]

# In[ ]:


competition = np.round(100 * results.groupby(
    "cluster_names").alternative_names.value_counts() / results.groupby(
    "cluster_names").alternative_names.count())
competition = competition.unstack()
competition.fillna(0, inplace=True)

# In[ ]:


plt.figure(figsize=(20,15))
sns.heatmap(competition, cmap="Greens", annot=True, fmt="g", square=True, cbar=False)

# ### Take-Away
# 
# * Let's only consider some example clusters that have low cluster assignment certainties. 
#     * The various Rods & Rings cluster has a tendency to join with the nucleoli fibrillar center & cytoplasmic bodies cluster.
#     * Aggresomes like to join with the mitotic spindle & organizing center cluster.

# ## Conclusion
# 
# Within this kernel you can find groups of targets that are likely to occur together. It covers many different topics that can help you to solve parts of the imbalance class problem or to improve target predictions. It's your turn now to figure out how to do this! ;-)
# 
# Happy coding!

# In[ ]:


results.to_csv("target_group_analysis.csv")
