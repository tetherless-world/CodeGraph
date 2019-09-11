#!/usr/bin/env python
# coding: utf-8

# **PS: If you like to fork and play with the results, don't plotly the 3D scatterplots due to instability of the kernel!  ;-) **
# 
# # Data Science Trainee Program
# 
# This kernel is part of Leas trainee program in data science & machine learning. During this program she studied what is meant by machine learning and how can a model find pattern in data. To obtain a realistic impression of the power and difficulties of this hot field of our time we decided to chose a powerful model - namely the Gaussian Mixture Model - and real (& dirty) data of openfoodfacts.  During this kernel we discovered the various applications and the structure of Gaussian Mixture like clustering with clusters of different densities and sizes, anomaly detection and uncertainty analysis. We examined the importance of feature preprocessing like scaling and transformation and the difficulty of setting hyperparameters like the number of clusters/components of our model.
# 
# We collaborated equally on this notebook and if you like our analysis, you can make us very happy with an upvote! :-)
# If you have further questions, feedback or ideas, don't hesitate to leave a comment.
# 
# 
# ## Can we cluster products of the same type? 
# 
# In this Kernel we want have a closer look at the OpenFoodFacts data. Everyone who already used the OpenFoodFacts app knows that it is complicated to type in all queried information and you also have to put high effort into this procedure as well. This leads to wrong entries, regardless of whether it happened on purpose or just because someone made a mistake. By using the app you can also realise that there is no check or test for the user inputs. For example you can even type in negative amounts of fat if you want to. Because of these circumstances we can assume that there are a lot of mistakes in the OpenFoodFacts data.
# 
# These mistakes are not advantageous for the app if it wants to tell its users the important information about a product easily and fast, what definetly is one of the aims of the app. So it seems like a huge problem that the app is full of these mistakes and can't prevent them.
# 
# We now want to look closer at the given data and analyze possible false data. Besides the obvious mistakes like negative or too high entries we also want to use the Gaussian Mixture Model to find anomalies depending on the natural structure of the given data. 
# 
# In addition to that we will also use the Gaussian Mixture Model to see if there are any natural structures in our data. This would be really helpful to classify products without entering lots of different categories like you have to do now while adding a product to the database. The app could also control itself by suggesting a category out of the given entries and then asking the user if the product really belongs to this category. 
# 
# After this analysis is done we will also discuss why the Gaussian Mixture Model was used and evaluate our results. Then we are also going to give an outlook for the possible future of the app.
# 
# ## Table of contents
# 
# 1. [Loading packages and data](#load) (complete)
# 2. [Helper methods](#helper) (complete)
# 3. [The Gaussian Mixture Model](#gmm) (complete)
# 4. [Excluding missing values](#missing) (complete)
# 5. [Feature engineering](#featureE) (complete)
# 6. [Eliminating obvious error sources](#errors) (complete)
# 7. [Feature scaling and transformation](#preprocessing) (complete)
# 8. [Generate test data](#test) (complete)
# 9. [Training](#training) (complete)
# 10. [Clustering of product types](#clustering) (complete)
# 11. [Certainty Analysis](#uncertainty) (complete)
# 12. [Anomaly detection](#anomalies) (complete)
# 13. [Conclusion](#conclusion) (complete)
# 14. [Outlook](#outlook) (complete)
# 
# 

# # Loading packages and data <a class="anchor" id="load"></a>
# 
# Before we can start we need to import some packages and the data of course:

# In[ ]:


import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
sns.set()

from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import os
print(os.listdir("../input"))


from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split

from scipy.stats import boxcox

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# **Peek at the data**

# In[ ]:


original = pd.read_csv("../input/en.openfoodfacts.org.products.tsv",
                       delimiter='\t',
                       encoding='utf-8')
original.head()

# In[ ]:


original.shape

# Scrolling through the features we can observe that we have a lot of missing values in this data.

# In[ ]:


original.product_name = original.product_name.astype(str)

# # Helper methods <a class="anchor" id="helper"></a>
# 
# We will use some helper methods that we collect in this chapter to reduce amount of code during analysis:
# 
# * get_outliers
# * make_word_cloud
# * split_data_by_nan
# * scale_and_log

# In[ ]:


def get_outliers(log_prob, treshold):
    epsilon = np.quantile(log_prob, treshold)
    outliers = np.where(log_prob <= epsilon, 1, 0)
    return outliers 

def make_word_cloud(data, cluster, subplotax, title):
    words = data[data.cluster==cluster]["product"].apply(lambda l: l.lower().split())
    cluster_words=words.apply(pd.Series).stack().reset_index(drop=True)

    text = " ".join(w for w in cluster_words)

    # Create and generate a word cloud image:
    wordcloud = WordCloud(max_font_size=30, max_words=30,
                          background_color="white", colormap="YlGnBu").generate(text)

    # Display the generated image:
    
    subplotax.imshow(wordcloud, interpolation='bilinear')
    subplotax.axis("off")
    subplotax.set_title(title)
    return subplotax

def split_data_by_nan(cutoff):
    cols_of_interest = nan_values[nan_values <= cutoff].index
    data = original[cols_of_interest]
    return data.copy()

def transform(data, feature, constant, lam):
    scaler = MinMaxScaler((0,5))
    scaled_feature = scaler.fit_transform(data[feature].values.copy().reshape(-1,1))
    data["boxcox_" + feature] = boxcox(scaled_feature + constant, lam)
    scaler = StandardScaler()
    data["transformed_" + feature] = scaler.fit_transform(data["boxcox_" + feature].values.reshape(-1,1))
    return data


class TransformParameter:
 
    def __init__(self, const, lam, color="Dodgerblue"):
        self.const = const
        self.lam = lam
        self.color = color

def make_violin(subax, cluster, nutrients):
    pos = np.arange(1, len(nutrients)+1)
    part = subax.violinplot(
            nutrition_table[nutrition_table.cluster==cluster]
                [nutrients].values,
            showmeans=True,
            showextrema=False)
    subax.set_title("Feature distributions of cluster: " + str(cluster), size = 20)
    subax.set_xticks(pos)
    subax.set_xticklabels(nutrients)
    set_color(part, len(nutrients))
    return subax

def set_color(axes, num_colors):
    cm = plt.cm.get_cmap('RdYlBu_r')
    NUM_COLORS=num_colors
    for n in range(len(axes["bodies"])):
        pc = axes["bodies"][n]
        pc.set_facecolor(cm(1.*n/NUM_COLORS))
        pc.set_edgecolor('black')
    return axes

# # The Gaussian Mixture Model <a class="anchor" id="gmm"></a>

# **First of all, what is the idea of a gaussian mixture model?**
# 
# Imagine you are given 200000 nutrition tables without further information. Your task is to find some grouping in this data. Your intuition and experience tell you that the structure you are looking for depends on other things like the kind of product and ingredients. Even though this information is not available for your task you have this in mind and you will try to group products asking yourself: "Could this be cheese?". This way you are dealing with a **hidden, so called latent information**. Their information is not given as a feature in your data but you are assuming that it somehow causes your patterns and structures. 
# 
# In the case of **mixture models it's assumed that there exist some latent variables that generate the distribution of data you observe**. In our case we hope that these latent variables walk along with product types and categories that fit to our experience of daily life. As a first attempt we thought about how many categories we may find, for example: pasta and grain products, bread, vegetables, fruits, meat, candies, oils, milk and milk products, beans. For this first try we had in mind the nurition pyramide. 
# 
# Now the gaussian mixture model descripes the following: **Each of our latent variable, each of these product categories, causes one multivariate gaussian**, this is a normal distribution over more than one dimension. We have least these features: carbohydrates, sugars, fat, proteins, energy and salt. Consequently our feature dimension is equally (or with feature engineering) higher than 6 and the gaussians are located in this > 6 dimensional room. In 3 dimensions you can think of foggy ellipsoids with different densities, locations and extentions. These gaussians try to cover the data structure and you take as many gaussians as you assume product categories. 
# 
# Then our distribution $p(x)$ of data we observe is said to be generated by these gaussians:
# 
# $$ p(x) = \sum_{k=1}^{K} \pi_{k} \cdot N(x|\mu_{k}, \Sigma_{k}) $$
# 
# 
# Then each gaussian is somehow responsible that a single data spot is placed where it is. **One gaussian may be more responsible than the others but they are all working in a mixture to explain your data**. During learning all of these ellipsoid clusters are moving in space and varying their shape trying to match perfactly the distributed data $p(x)$. **The learning procedure for models with latent variables is the expectation maximization**. This algorithm works in two consequtive steps that are repeated until all gaussians only have found their place and would only show slight changes in further steps. 
# 
# #### E-Step (expectation):  
# 
# At the start of the model are gaussians are placed and shaped randomly (or pretrained by k-means locations). Then for each gaussian, that represents one cluster, the model calculates how responsible the cluster $z_{k}$ is for one data spot $x_{n}$.
# 
# $$\gamma_{nk}  = \frac {\pi_{k} \cdot N(x_{n}|\mu_{k}, \Sigma_{k})} {\sum_{j=1}^{K} \pi_{j} \cdot N(x_{n}|\mu_{j}, \Sigma_{j})}$$
# 
# These responsibilities sum up to 1 over all clusters for one data spot. Here we can read out the winner at the end of the learning procedure that is assigned as the predicted cluster for that data spot $x_{n}$. 
# 
# #### M-Step (maximization):
# 
# During the maximization step, all parameters of the gaussians, the locations given by the cluster center and the shapes covered by the covariance matrices are updated:
# 
# $$\mu_{k} = \frac{1}{N_{k}} \sum_{n=1}^{N} \gamma_{nk} x_{n} $$
# 
# $$\Sigma_{k} =  \frac{1}{N_{k}}  \sum_{n=1}^{N}  \gamma_{nk} (x_{n} - \mu_{k}) (x_{n} - \mu_{k})^{T}$$ 
# 
# You can see that we obtain each cluster location $\mu_{k}$ by looking and summing over all data spots. This is a big difference between K-Means and GMM as the further only calculates the cluster centers using cluster members. But in our case, as already told, each gaussian is responsibile for all data points. Hence the new location is calculated by using them all, but weighting with the responsibility. Thisway points that could be well explained by this cluster contribute much more to the new location (mean) than the other data points. The same holds for the cluster shapes aka covariances.  
# 
# OhOhOh! What do we see there? :-)
# 
# We have to be very careful with outliers as our training procedure depends on the features $x_{n}$ themselves!! High outlier values shift the means and covariances of clusters to higher values as well! That motivates extensive data preprocessing for training data!
# 

# # Excluding missing values <a class="anchor" id="missing"></a>
# 
# To cluster products we like to focus on the nutrition table as this information is often typed in by the user and will be automatically read in by deep learning networks in the future. Consequently we will only use these features:

# In[ ]:


nutrition_table_cols = ["energy_100g",
                        "fat_100g",
                        "carbohydrates_100g",
                        "sugars_100g",
                        "proteins_100g",
                        "salt_100g"]

# Let's see how many products have NaN-entries in at least one of the features below. After that we are going to drop these products so our dataset doesn't have missing values anymore.

# In[ ]:


nutrition_table = original[nutrition_table_cols].copy()

nutrition_table["isempty"] = np.where(nutrition_table.isnull().sum(axis=1) >= 1, 1, 0)
percentage = nutrition_table.isempty.value_counts()[1] / nutrition_table.shape[0] * 100
print("Percentage of incomplete tables: " + str(percentage))

nutrition_table = nutrition_table[nutrition_table.isempty==0].copy()
nutrition_table.isnull().sum()

nutrition_table.drop("isempty", inplace=True,axis=1)
nutrition_table.dropna(axis = 0, how = "any", inplace=True)

# After dropping all incomplete nutrition tables, we are left with the following total number of samples:

# In[ ]:


nutrition_table.shape[0]

# # Feature engineering <a class="anchor" id="featureE"></a>
# 
# Before we start our analysis we have to make some more adjustments to our dataset. First of all we want to add some more features that could be helpful for clustering our data.
# 
# Therefore we add the feature **g_sum** which represents the **rounded sum of the fat-, carbohydrates-, proteins- and salt-values** in our data. By doing that we can easily see if there are some products with false entries.
# 
# Furthermore we add the feature **other_carbs**  which includes the value of **all carbs that are not sugars**. Because of that our model can see the correlation between carbohydrates and sugars.
# 
# The last feature we want to add is **reconstructed_engery**. It calculates the energy value of a product **based on energy values of the features fat, carbohydrates and proteins**. We can compare this feature to the amount of energy that is given in our dataset to see if there possibly are some wrong entries.

# In[ ]:


nutrition_table["g_sum"] = nutrition_table.fat_100g + nutrition_table.carbohydrates_100g + nutrition_table.proteins_100g + nutrition_table.salt_100g
nutrition_table["g_sum"] = round(nutrition_table.g_sum)
nutrition_table["other_carbs"] = nutrition_table.carbohydrates_100g - nutrition_table.sugars_100g

nutrition_table["reconstructed_energy"] = nutrition_table.fat_100g * 37 + (nutrition_table.proteins_100g + nutrition_table.carbohydrates_100g)* 17

# In[ ]:


nutrition_table.columns

# In[ ]:


meta = pd.DataFrame(index=nutrition_table.index)
for col in nutrition_table.columns:
    meta["zero_" + col] = np.where(nutrition_table[col] == 0, 1, 0)
meta["contains_zero"] = np.where(meta.sum(axis=1) > 0,1,0 )

# # Eliminating obvious error sources <a class="anchor" id="errors"></a>
# 
# Now that we implemented our new features we also want to exclude obvious wrong entries, so that we delete all products with:
# 
# * a feature (except for the energy-ones) higher than 100g
# * a feature with a negative entry
# * an energy-amount of more than 3700kJ (the maximum amount of energy a product can have; in this case it would conists of 100% fat)
# * more sugars than carbohydrates
# * g_sum higher than 100g

# In[ ]:


for col in nutrition_table.columns:
    if col not in ["energy_100g", "reconstructed_energy"]:
        nutrition_table = nutrition_table.loc[nutrition_table[col] <= 100]
    nutrition_table = nutrition_table.loc[nutrition_table[col] >= 0]

nutrition_table = nutrition_table.loc[nutrition_table.energy_100g <= 3700]
nutrition_table = nutrition_table.loc[nutrition_table.carbohydrates_100g >= nutrition_table.sugars_100g]
nutrition_table = nutrition_table.loc[nutrition_table.g_sum <= 100]

# In[ ]:


nutrition_table.head()

# # Feature scaling and transformation <a class="anchor" id="preprocessing"></a>
# 
# After these steps our dataset is cleaned and optimized for our analysis, but because we want to use the Gaussian Mixture Model we still have to scale and transform the data a bit. Looking closer on how our model learns, we can see two problems in the M-Step:
# 
# $$\mu_{k} = \frac{1}{N_{k}} \sum_{n=1}^{N} \gamma_{nk} x_{n} $$
# 
# $$\Sigma_{k} =  \frac{1}{N_{k}}  \sum_{n=1}^{N}  \gamma_{nk} (x_{n} - \mu_{k}) (x_{n} - \mu_{k})^{T}$$ 
# 
# ### Why should we transform features?
# 
# The cluster center $\mu_{k}$ and the covariance matrices $\Sigma_{k}$ are influenced by the feature values $x_{n}$ themselves. As we have skewed feature distributions with high outliers the cluster center are shifted towards higher values during each update step. This is caused by the weighted mean. Even though the responsibilities as weights are able to soften the effect of outliers, they can still contribute very much. In this case our model builds clusters during learning that try to match the structure of outliers. The cluster center would not explain the majority of its cluster members and could be even located in regions with no data spots at all. To solve this problem we transform our features using boxcox transformation. Doing so we like to expand the majority of data points to uncover hidden substructures and compress the outliers.
# 
# Even this sounds easy, this is the most toughest part of all! 
# 
# ### Why should we scale them?
# Even with transformed features there is still a problem left: So far all our features have positive values and the responsibilities are positive as well as they are probabilities between 0 and 1.  This will still cause a shift in the center and covariance updates as the assignments will be lead by the higher values. To correct this we want to shift our transformed feature distributions such that it is evenly distributed around 0. For this purpose we subtract the mean and scale to unit variance. 
# 
# ### How does to boxcox knead our features?
# Using the boxcox transformation we are able to knead the feature distributions as we like them to be. But with this this great power comes also great responsibility ;-) . Consequently we should try to understand what boxcox can do with our values!

# In[ ]:


to_knead = np.arange(0,10,0.1)

fig, ax = plt.subplots(1,2,figsize=(20,5))

k = ax[0].plot(to_knead, to_knead, color="Red", label="to_knead")
e1 = ax[0].plot(to_knead, boxcox(to_knead + 0.01, 1.3), color="darkblue", label="0.7")
e2 = ax[0].plot(to_knead, boxcox(to_knead + 0.01, 0.7), color="mediumblue", label="0.7")
e3 = ax[0].plot(to_knead, boxcox(to_knead + 0.01, 0.5), color="royalblue", label="0.5")
e4 = ax[0].plot(to_knead, boxcox(to_knead + 0.01, 0.01), color="cornflowerblue", label="0.01")
ax[0].set_title("Boxcox-Knead with different powers $\lambda$")
handles=[k, e1, e2, e3, e4]
ax[0].set_xlabel("Value")
ax[0].set_ylabel("Transformed Value")
#ax[0].legend(handles)

k = ax[1].plot(to_knead, to_knead, color="Red", label="to_knead")
e1 = ax[1].plot(to_knead, boxcox(to_knead + 0.01, 0.5), color="darkblue", label="0.01")
e2 = ax[1].plot(to_knead, boxcox(to_knead + 0.5, 0.5), color="mediumblue", label="0.5")
e3 = ax[1].plot(to_knead, boxcox(to_knead + 1, 0.5), color="royalblue", label="1")
e4 = ax[1].plot(to_knead, boxcox(to_knead + 1.5, 0.5), color="cornflowerblue", label="1.5")
ax[1].set_title("Boxcox-Knead with different constants $c$")
ax[1].set_xlabel("Value")
ax[1].set_ylabel("Transformed Value")
handles=[k, e1, e2, e3, e4]
#ax[0].legend(handles)

# Looking at the power parameter $\lambda$ of the boxcox transformation you can see that we are able to compress outliers more and more by chosing lower values of $\lambda$. In addition we obtain a "negative" stretching of low original values lower than one. If we fix $\lambda =0.5$ and vary the constant $c$ we can observe only slight differences of the high values compression. On the other hand we can see that the stretching of low values is even stronger of constants c close to zero. 
# 
# #### Take-away (not that simple but still good)
# 
# * If you like to compress outliers - chose a low $\lambda$
# * If you like to stretch low values - chose a low $c$ close to zero.
# 
# 
# ### Transform! 
# 
# Ok, let's go! If you like, fork and chose other values. You will see that the resulting clusters highly depend on these hyperparameters!

# In[ ]:


constants = {"carbohydrates_100g": TransformParameter(const=0.01, lam=0.9, color="Green"),
            "fat_100g" : TransformParameter(const=0.1, lam=0.05, color="orange"),
            "proteins_100g": TransformParameter(const=0.1, lam=0.1, color="Red"),
             "sugars_100g" : TransformParameter(const=0.07, lam=0.03, color="Fuchsia"),
            "other_carbs" : TransformParameter(const=0.05, lam=0.3, color="maroon"),
            "salt_100g" : TransformParameter(const=0.01, lam=0.005, color="Dodgerblue"),
            "energy_100g" : TransformParameter(const=0.3, lam=0.7, color="Purple"),
            "reconstructed_energy" : TransformParameter(const=0.3, lam=0.7, color="mediumvioletred"),
            "g_sum": TransformParameter(const=0.1, lam=1.2, color="turquoise")}

# In[ ]:


for key in constants.keys():
    transform(data=nutrition_table,
              feature=key,
              constant=constants[key].const,
              lam=constants[key].lam)

# ### Visual comparison
# 
# Let's have a look at our transformed features:

# In[ ]:


fig, ax = plt.subplots(9,3,figsize=(25,40))
n = 0
for feature in constants.keys():
    sns.distplot(nutrition_table[feature],
                 ax=ax[n,0],
                 color=constants[feature].color)
    sns.distplot(nutrition_table[nutrition_table[feature]>0]["boxcox_" + feature],
                 ax=ax[n,1],
                 color=constants[feature].color)
    sns.distplot(nutrition_table["transformed_" + feature],
                 ax=ax[n,2],
                 color=constants[feature].color)
    n+=1


# Many of them look far better than before but the problem of setting optimal hyperparameters is still very difficult.

# # Generate test and validation data <a class="anchor" id="test"></a>
# 
# * Find the optimal number of clusters (test)
# * Show outlier/anomalie detection on unseen data with outliers! 

# In[ ]:


nutrition_table, test_table = train_test_split(nutrition_table, test_size=0.3, random_state=0)

# # Training <a class="anchor" id="training"></a>

# As already explained we will use the transformed and scaled features during the learning procedure to improve the update of the M-Step. A difficult and not obvious task is to chose the right amount of components of our mixture model. If you uncomment the code part "searching for the optimal amount of clusters" you can see that we tried to estimate the optimal number of components on test data.

# In[ ]:


features = ["transformed_carbohydrates_100g",
            "transformed_fat_100g",
            "transformed_proteins_100g",
            "transformed_sugars_100g",
            "transformed_salt_100g",
            "transformed_other_carbs",
            "transformed_energy_100g",
            "transformed_reconstructed_energy",
            "transformed_g_sum"]

# ### Searching for the optimal amount of clusters
# 
# We are faced with the challange to find the optimal number of components (clusters) for our gaussian mixture model. This is the second difficult part of our analysis. If we would chose to less components we would obtain clusters of mixed product types and far too broad ranged features per gaussian. For example, it does not make sense for one cluster to occupy the whole span from 0 to 100g of carbohydrates. In contrast if we choose too many components, we may find some fine-grained, special clusters of seldom products but we may also find separate clusters of the same product type. To understand what a more theoretical approach would advice us we ran GMM for different amount of components and took a look at three criteria: 
# 
# * The log-likelihood on the test data 
# * The Bayesian Information Criterion (penalizes the log-likelihood with model complexity and number of data)
# * The Akaike Information Criterion (only penailizes the log-likelihood with model complexity)
# 

# In[ ]:


X_train = nutrition_table[features].values
X_test = test_table[features].values

# In[ ]:


print("Number of train samples: " + str(X_train.shape[0]))

# In[ ]:


# ONLY uncomment, if you have at least an hour time to wait for computation! 
# scores = []
# bic = []
# aic = []
# to_try = range(11,81,10)
# for components in to_try:
    # model = GaussianMixture(n_components=components, covariance_type="full", random_state=1)
    # model.fit(X_train)
    # scores.append(model.score(X_test))
    # bic.append(model.bic(X_test))
    # aic.append(model.aic(X_test))

# fig, ax = plt.subplots(1,3,figsize=(25,5))
# ax[0].plot(to_try, scores)
# ax[1].plot(to_try, bic)
# ax[2].plot(to_try, aic)
# ax[0].set_xlabel("Number of components")
# ax[0].set_ylabel("Log-Likelihood")
# ax[1].set_xlabel("Number of components")
# ax[1].set_title("BIC")
# ax[1].set_ylabel("Penalized Log-Likelihood")

# ax[2].set_ylabel("Penalized Log-Likelihood")
# ax[2].set_xlabel("Number of components")
# ax[2].set_title("AIC")

# #### Take-away
# 
# * The pure log-likelihood as well as the AIC and BIC tell us: "the more clusters, the better!"
# * But we know we should be very careful. Products of the same kind could spread widely in their nutrition features according to many differences in the ingredients. In addition we are challanged by dirty and real data. User can make various of different errors like flipping nutrients while typing in the nutrition table or setting commas in numerical values on the wrong place. Even mistakes on purpose with wrong entries are possible. 
# 
# Consequently we will try another approach as well to select a sufficent number of clusters by looking at the correlation of features between different clusters.

# In[ ]:


components = 20

model = GaussianMixture(n_components=components,
                        covariance_type="full",
                        random_state=1,
                        n_init=1,
                        max_iter=200,
                        init_params="kmeans")
model.fit(X_train)
print("Model converged: " + str(model.converged_))

nutrition_table["cluster"] = model.predict(X_train)
test_table["cluster"] = model.predict(X_test)

# # Clustering of product types <a class="anchor" id="clustering"></a>

# Fitting with 20 components seemed to be a good choice looking at cluster center correlations later in the analysis. Select 3 features of your choice and take a look at these interesting clusters:
# 
# * carbohydrates_100g
# * fat_100g
# * proteins_100g
# * salt_100g
# * other_carbs
# * sugars_100g
# * energy_100g
# * reconstructed_energy
# * g_sum
# 
# You can select the transformed features as well with "transformed_" + your_feature!
# 
# For the plot we only used 40000 data spots for stability reasons of plotly.

# In[ ]:


#plot_features = ["transformed_carbohydrates_100g", "transformed_proteins_100g", "transformed_fat_100g"]
plot_features = ["carbohydrates_100g", "proteins_100g", "fat_100g"]

# In[ ]:


N = 10000

trace1 = go.Scatter3d(
    x=nutrition_table[plot_features[0]].values[0:N], 
    y=nutrition_table[plot_features[1]].values[0:N],
    z=nutrition_table[plot_features[2]].values[0:N],
    mode='markers',
    marker=dict(
        color=nutrition_table.cluster.values,
        colorscale = "Jet",
        opacity=0.3,
        size=2
    )
)

figure_data = [trace1]
layout = go.Layout(
    title = 'Clustering results',
    scene = dict(
        xaxis = dict(title=plot_features[0]),
        yaxis = dict(title=plot_features[1]),
        zaxis = dict(title=plot_features[2]),
    ),
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=0
    ),
    showlegend=True
)

fig = go.Figure(data=figure_data, layout=layout)
py.iplot(fig, filename='simple-3d-scatter')

# ### Take-away
# 
# * Every cluser belongs to a component that has its own covariance matrice. As we allowed full matrices during training we can see clusters of various sizes and densities. This is one advantage of using the gaussian mixture model.
# * However some less dense clusters in boundary regions show some outliers even though they have a compact and dense base structure. In these cases we are curious if the model is centrain about its cluster assignment or if it detected those points as anomalies. 
# * If you compare the scattered data of original and transformed features you can see that our preprocessing was able to expand the room the data occupies. This way we uncovered substructures that would have been hidden using the skewed original feature distributions. 
# * We can find a cluster that somehow looks like protective shield. It seems to hold all outliers in the data and its cluster center may be located somewhere in the middle of all data. That is very interesting! We haven't expected that!

# ## How much data is covered per cluster?

# In[ ]:


cluster_count = nutrition_table.cluster.value_counts()

fig, ax = plt.subplots(1,2,figsize=(20,5))
sns.barplot(x=cluster_count.index,
            y=cluster_count.values / nutrition_table.shape[0] * 100,
            order=cluster_count.index, ax=ax[0])
ax[0].set_xlabel("Cluster Number")
ax[0].set_ylabel("Percentage of data")
sns.distplot(cluster_count.values / nutrition_table.shape[0] * 100, bins=10, ax=ax[1])
ax[1].set_xlabel("Data coverage in %")
ax[1].set_ylabel("Density")

# ### Take-away
# 
# * We can see that most of our clusters cover around  5 % of our data. 
# * Nonetheless there exist some clusters with verly low or very high coverage.
# 
# We should try to find out which kind of products and feature ranges belong to these clusters. This way we should try to find out if our clusters are inhomogenous or if we should use more components for our mixture model.

# ## Can we reveal the hidden product type?

# In[ ]:


nutrition_table["product"] = original.loc[nutrition_table.index, "product_name"]

fig, ax = plt.subplots(10,2, figsize=(20,50))
for m in range(10):
    for n in range(2):
        cluster = m*2+ n
        title = "Cluster " + str(cluster) 
        make_word_cloud(nutrition_table, cluster, ax[m,n], title)

# ### Take-away
# 
# * First of all: YES! We can find hidden product categories. That's amazing! No one told Gaussian Mixture something about product names or types. Only dealing with nutrition table information it was able to group similar products. We can find various interesting clusters of chocolate, pasta, ice cream, cheese, yoghurts, juice, grains, sauces, meat, water and nuts.
# * On the goarse-grained view our clustering was very successful. But if you take a closer look you may wonder why some clusters hold similar products or seem to be of a mixed type like cluster 6.
# 
# For this reason, some more question arise:
# 1. How sure is our model about our cluster assignments?
# 2. How valid is the data in context of app user errors given a cluster?

# ## How similar are different clusters?

# To obtain a first impression of the similarity of our clusters we can look at the correlation of the cluster center:

# In[ ]:


cluster_stats = pd.DataFrame(data=model.means_)
sns.set(style="white")
corr = cluster_stats.transpose().corr()

mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize=(20, 20))
sns.heatmap(corr, mask=mask, cmap="coolwarm",
            vmin=-1, vmax=1, square=True, center=0,
            linewidths=.5, cbar_kws={"shrink": .5},
           annot=True)
ax.set_xlabel("Cluster")
ax.set_ylabel("Cluster")
ax.set_title("Cluster center correlation")

# ### Take-away
# 
# * Many clusters do not correlate or show anti-correlation. This is what we like to have as we want them do be dissimilar in the feature space. This way we can make sure that they can cover different type of products. 
# * Unfortunately we also find some high correlating clusters that have nearby cluster centers like the pasta cluster 9 and the whole grain cluster 16. In this case it seems to make sense. Even though these kind of products are very similar, pasta seems to be different from whole grain. That's cool! But we should try to understand in the feature space what makes this difference.

# ## How are the nutrients of different clusters distributed? 

# In[ ]:


nutrients = ["fat_100g",
             "proteins_100g",
             "carbohydrates_100g",
             "sugars_100g", 
             "other_carbs",
             "salt_100g",
             "g_sum"]
energies = ["energy_100g", "reconstructed_energy"]

transformed_nutrients = ["transformed_" + nutrient for nutrient in nutrients]
transformed_energies = ["transformed_" + energy for energy in energies]

# In[ ]:


cluster_pair = [9,16]

# In[ ]:


sns.set()

fig, ax = plt.subplots(2,2,gridspec_kw = {'width_ratios':[3, 1]}, figsize=(20,10))
pair00 = make_violin(ax[0,0], cluster_pair[0], nutrients)
ax[0,0].set_ylim([0,100])
pair01 = make_violin(ax[0,1], cluster_pair[0], energies)
ax[0,1].set_ylim([0,4000])
pair10 = make_violin(ax[1,0], cluster_pair[1], nutrients)
ax[1,0].set_ylim([0,100])
pair11 = make_violin(ax[1,1], cluster_pair[1], energies)
ax[1,1].set_ylim([0,4000])

# ### Take-Away
# 
# * Comparing two clusters of our choice we can obtain an deeper understanding why similar clusters are still different. They are often very close in all features in the transformed feature space, sometimes the feature distributions overlap. But even if they are very close together small and tiny shifts in some features make the difference. 
# * In the case of the pasta and grain cluster we can see that pasta has a little lower fat and higher protein than the whole grain cluster. In contrast, the grain cluster holds more products with various amounts of sugars and salt.
# * Looking at the mixed type cluster 6 we can see that it covers outliers of each feature distribution. You might wonder and ask yourself why we still see broad distributions on low values as well but this can be explained when you keep in mind, that an outlier might only occur in one or few features per product whereas all others behave normal. 
# * Comparing cluster 6 and 12 we can observe that they are very similar as well. They both cover varios products over all feature ranges. They only differ in the amount of fat, proteins and carbohydrates. Whereas cluster 12 holds products with zero fat and zero proteins, but a borad range from 0 to 100 g carbohydrates (& sugars), cluster 6 expands the whole feature space. Consequently cluster 12 touches the 0-fat-0-protein plane whereas cluster 6 is a crowd more in the middle of the room.
# * These difference of cluster 6 and 12 somehow reflects the problem that we have many discrete 0-entries for proteins, fat and carbohydrates. In our opinion it does not make sense to have two outlier-clusters that are nearly the same. We will come to this problem in our outlook again and give a deeper explanation of this problem.

# ## How dense are our clusters?

# In[ ]:


plt.figure(figsize=(10,10))
q75 = nutrition_table.groupby("cluster")[transformed_nutrients].quantile(0.75)
q25 = nutrition_table.groupby("cluster")[transformed_nutrients].quantile(0.25)
iqr = q75-q25
sns.heatmap(iqr.transpose(), cmap="YlGnBu_r", cbar=False)

# ### Take-Away
# * The clusters differ a lot in their densities
# * Cluster 5 has the highest desity, which makes sense when we look at the produkt-types - beverages.  Products of this group all have in common that they contain almost no fat, proteins and salt. Only the sugar component differs due to sweet drinks like soda and unsweetened drinks like water. For this reason the sugar component has the lowest density.
# * Cluster 6 hast the lowest density. That makes sense as cluster 6 contains the most outlier products. For this reason it's the most mixed cluster of all - the product-types differ a lot and so do their nutritions.
# * There are also some other clusters with a quite high or low density:
#     * Cluster 12 has low density in carbohydrates, sugars, non-sugar-carbohydrates, salt and the g_sum. It contains candy, fruits and sauces and is of mixed type as well. As cluster 6 it contains many outliers that causes the low density.
#     * Cluster 8 and 19 have high densities in almost every feature. If we compare with the word-clouds we can see that they are very homogenous: milk & joghurts cluster and juices cluster.

# In[ ]:


plt.figure(figsize=(10,2))
q75 = nutrition_table.groupby("cluster")[["transformed_energy_100g",
                                          "transformed_reconstructed_energy"]].quantile(0.75)
q25 = nutrition_table.groupby("cluster")[["transformed_energy_100g",
                                          "transformed_reconstructed_energy"]].quantile(0.25)
iqr = q75-q25
energy_difference = iqr.transformed_energy_100g - iqr.transformed_reconstructed_energy
sns.heatmap(iqr.transpose(), cmap="YlOrRd_r", cbar=False)

# * In the energy-densities you can see a lot of different densities as well.
# * Cluster 5 and 9 have the highest densities. Cluster 5 (beverage) also has the highest density in its other features, but cluster 9 (pasta) also has a very high density - only sugars are a bit less dense.
# * Cluster 12, 6 and 17 have the lowest densities. Cluster 12 and 6 are outlier clusters. Consequently it makes sense that they are of low density. Interestingly cluster 17 (sauces & dressings) is of low density es well. It seems to hold salt outliers, as this feature is its most less dense of all features. 

# # Certainty Analysis <a class="anchor" id="uncertainty"></a>
# 
# Wouldn't it be nice to find out what kind of products are somehow inbetween of two or more clusters? 
# 
# We are lucky! Our Gaussian Mixture Model is a probabilistic model that can tell us, how certain it was during its cluster assignments. Intuitively one would say: "Let's take a look at $\pi_{k} N(\mu_{k}, \Sigma_{k})$ per cluster component. But as we want a probability, we need to normalize over all components of our model. Looking at the E-Step, we can see that this are our converged responsibilities! :-) During the E-Step our model calculates how responsible some cluster $k$ is for some data spot $n$. Consequently we can try to understand for which kind of products or model is not certain about in its cluster assignment. This is a great advantage of Gaussian Mixture compared to non-probabilistic models like K-Means.

# ## How certain are our cluster assignments?

# In[ ]:


species_probas = model.predict_proba(X_train)
best_species_idx = np.argmax(species_probas,axis=1)

color = np.zeros(best_species_idx.shape[0])
for n in range(len(color)):
    color[n] = np.round(species_probas[n,best_species_idx[n]], 4)
nutrition_table["certainty"] = color

# In[ ]:


fig, ax = plt.subplots(1,2, figsize = (20,5))
sns.distplot(nutrition_table.certainty, color = "red", ax = ax[0],kde = False)
sns.distplot(nutrition_table.certainty[nutrition_table.certainty < 0.95], color = "orange", kde = False)
ax[0].set_title("Distribution of certainty")
ax[1].set_title("certainty < 0.95")

# Obviously our model is very certain about its cluster assignments for the majority of the data spots. Looking at samples with centainty below 95 % we can see a sharp break at 0.5. Consequently there are only very few samples that our model is really not sure to which cluster they should belong. We shall check whether these points could be outliers or anomalies. 

# ## Are there some clusters with high uncertainty?
# 
# Perhaps we have some cluster with more uncertain assigments than others. And perhaps similar clusters are both more uncertain. Let's find it out!

# In[ ]:


plt.figure(figsize=(20,5))
sns.violinplot(x="cluster", y="certainty", data=nutrition_table)

# We can't find an obvious pattern!

# ## How do certain and uncertain product names look like?
# 
# Looking at the distributions of cluster assignment certainty we can't find interesting relationships. Probably the product name is more informative to find interesting patterns:

# In[ ]:


cluster_pair=[9, 16]

# In[ ]:


fig, ax = plt.subplots(2,2, figsize = (20,10))
make_word_cloud(nutrition_table[nutrition_table.certainty == 1.],
                cluster_pair[0],
                ax[0,0],
                title = "cluster:" + str(cluster_pair[0]) + " // " + "certainty == 1.0")
make_word_cloud(nutrition_table[nutrition_table.certainty < 1.],
                cluster_pair[0],
                ax[0,1],
                title = "cluster:" + str(cluster_pair[0]) + " // " + "certainty < 1.0")

make_word_cloud(nutrition_table[nutrition_table.certainty == 1.],
                cluster_pair[1],
                ax[1,0],
                title = "cluster:" + str(cluster_pair[1]) + " // " + "certainty == 1.0")
make_word_cloud(nutrition_table[nutrition_table.certainty < 1.],
                cluster_pair[1],
                ax[1,1],
                title = "cluster:" + str(cluster_pair[1]) + " // " + "certainty < 1.0")

# ### Take-Away
# 
# * Cool! That's great! Data spots of high certainty are even closer in the product name/type. 
# * In contrast those with less certainty are often different in the product type. 
# * The pasta and grain clusters for example show some overlaps in the uncertain data spots. There are some cereals in the pasta cluster that our model is uncertain about that are maybe under the hood of the grain cluster as well with a second peak of resposibility there. These would be products "inbetween". 
# 
# We should gain a deeper understanding of competing clusters by looking at the second highest responsibility per data spot and cluster. Let's visualize this competitor distribution with a heatmap. 

# ## Which cluster was the second choice of certainty per cluster?

# For this step we only take into account uncertain data spots with a certainty below 95 %. This way we can clearly see which is a competing cluster for those samples. 

# In[ ]:


nutrition_table["alternative_cluster"] = np.argsort(species_probas, axis=1)[:,-2]
uncertain_table = nutrition_table[nutrition_table.certainty < 0.95]

# In[ ]:


competition = np.round(100 * uncertain_table.groupby(
    "cluster").alternative_cluster.value_counts() / uncertain_table.groupby(
    "cluster").alternative_cluster.count())
competition = competition.unstack()
competition.fillna(0, inplace=True)

# In[ ]:


plt.figure(figsize=(20,10))
sns.heatmap(competition, cmap="Greens", annot=True, fmt="g")

# ### Take-Away
# 
# * That's again cool! Our pasta cluster 9 for example has in around 78 % of uncertain cases a high responsibility for the grain cluster 16. The other major uncertainties of the pasta cluster are assigned to our mixed type - outlier - cluster 6. 
# * In contrast our grain cluster 16 shows only low percentage in uncertain cases for the pasta cluster, namely 11 %. Consequently we can't conclude that if two cluster compete that they do it in both directions!
# * This heatmap is really an advantage if you like to know which clusters or groups are somehow interacting.
# * One interesting competing cluster is the creamy, cheesy cluster 4. Many clusters like the meat cluster 10, the cheese cluster 14 and the virgin oil and sauces cluster 17 are in love with it. If you compare the features in the violinplot above, we can see that the cluster 4 covers a broad range of proteins, fat and salt. This causes the high tendency of second highest resposibilites.

# ### Certainty 3D Scatterplot
# 
# Just to obtain an impression how uncertainty of cluster assignments are looking in the feature space:

# In[ ]:


trace2 = go.Scatter3d(
    x=nutrition_table[plot_features[0]].values[0:N], 
    y=nutrition_table[plot_features[1]].values[0:N],
    z=nutrition_table[plot_features[2]].values[0:N],
    mode='markers',
    marker=dict(
        color= color,
        colorscale = "Blackbody",
        opacity=0.5,
        size=2
    )
)

figure_data = [trace2]
layout = go.Layout(
    title = 'Certainty of Clustering',
    scene = dict(
        xaxis = dict(title= plot_features[0]),
        yaxis = dict(title= plot_features[1]),
        zaxis = dict(title= plot_features[2]),
    ),
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=0
    ),
    showlegend=True
)

fig = go.Figure(data=figure_data, layout=layout)
py.iplot(fig, filename='simple-3d-scatter')

# # Anomaly detection <a class="anchor" id="anomalies"></a>
# 
# The last and perhaps the most important part of the app is anomaly detection. But first of all, what is meant by an anomaly in the view of a gaussian mixture model? Before we start, we should be certain about the answer. Let's recap the formulation of the Gaussian Mixture Model:
# 
# $$ p(x) = \sum_{k=1}^{K} \pi_{k} \cdot N(x|\mu_{k}, \Sigma_{k})$$
# 
# This sum yields a so called probability density function. If you are not very familiar with measure theory you are in good company. If we assume that all data spots are drawn randomly from that density distribution and finally take the natural log, we would end up with the following formula:
# 
# $$ \ln p(x) = \sum_{n=1}^{N} ln \sum_{k=1}^{K} \pi_{k} \cdot N(x_{n}|\mu_{k}, \Sigma_{k}) $$
# 
# The stuff we need to score each data spot is now given by:
# 
# $$ \ln p_{n}(x) = \ln \sum_{k=1}^{K} \pi_{k} \cdot N(x_{n}|\mu_{k}, \Sigma_{k}) $$
# 
# This is what we obtain by calling score samples of scikit-learns gaussian mixture model. But what does it tell us? We think that it is a measure how dense the region is where data spot $x_{n}$ is located. Consequently in low dense regions of our probability density function this would yield a low number. Let's go further - what can be detected as anomaly in low dense regions?
# 
# * outliers that could be errors
# * seldom products
# 
# We should keep in mind that there is one kind of error we will not detect as anomaly: Imagine a user typed in nutrition table information that contains flips between features or a wrong product name. This kind of data spot can be located in dense regions and perfectly match with some cluster. Consequently we would not detect it as an anomaly and we can't find it as an error! 

# In[ ]:


log_prob = model.score_samples(X_train)

# ## How can we define at which density it's anomal?
# 
# The scoring of data points only yields sample log-likelihoods. No one tells us, if the sample is anomalous or not and we must define a criteria or some other strategy to proceed. In this case we decided to use a threshold that is based on the p-quartile of your choice. We set it by looking at the distribution over all sample-log-likelihoods. At the point were the distribution shows a sharp increase we set its value. This way we made sure that all spots in low dense regions are called an anomaly.   

# In[ ]:


your_choice = 0.07

# For us a value of 7 % anomalies in the data seemed to be suitable. Try another value if you like ;-) !

# In[ ]:


plt.figure(figsize=(20,5))

sns.distplot(log_prob, kde=False, bins=100, color="Red")
g1 = plt.axvline(np.quantile(log_prob, 0.25), color="Green", label="Q_25")
g2 = plt.axvline(np.quantile(log_prob, 0.5), color="Blue", label="Q_50 - Median")
g3 = plt.axvline(np.quantile(log_prob, 0.75), color="Green", label="Q_75")
g4 = plt.axvline(np.quantile(log_prob, your_choice), color="Purple", label="Q_ %i" % (int(your_choice*100)))
handles = [g1, g2, g3, g4]
plt.xlabel("log-probabilities of the data spots")
plt.ylabel("frequency")
plt.legend(handles) 

# In[ ]:


outliers = get_outliers(log_prob, your_choice)
nutrition_table["anomaly"] = outliers

# ## Is the percentage of anomalies dependent on the cluster?

# In[ ]:


anomalies_per_cluster = nutrition_table.groupby("cluster").anomaly.value_counts() / nutrition_table.groupby("cluster").anomaly.count()
reset = anomalies_per_cluster.reset_index(level="cluster")
percentage_anomalies = reset.loc[1].set_index("cluster", drop=True)

plt.figure(figsize=(20,5))
sns.barplot(x=percentage_anomalies.index, y=percentage_anomalies.anomaly.values * 100)
plt.ylabel("Percentage of Anomalies")

# We found that some clusters have a very high percentage of anomalies or are completely occupied by them. We have already seen that cluster 6 has the tendency to hold all outliers. Consequently it's not strange that it mainly consists of data spots in low dense regions. Let's try to find out, how certain the cluster assignment of anomalies is and how they look like in some example clusters.

# ## What can we tell about the anomaly cluster 6 and its counterpart 12 ?
# 
# We have found one special cluster that holds many high outlier products. Let's try to understand this special kind of cluster:

# In[ ]:


fig, ax = plt.subplots(1,2,figsize=(15,5))
make_word_cloud(nutrition_table,
                6,
                ax[0],
                title = "Anomalistic outlier cluster 6")
make_word_cloud(nutrition_table,
                12,
                ax[1],
                title = "Similar outlier cluster 12")

# In[ ]:


features = ["energy_100g", "reconstructed_energy"]

# In[ ]:


def get_values(cluster, feature, anomal):
    x = nutrition_table[(nutrition_table.cluster == cluster) & (nutrition_table.anomaly==anomal)][feature].values
    return x

size = 5
alpha = 0.5

fig, ax = plt.subplots(2,2, figsize = (20,19))
ax[0,0].scatter(get_values(6, features[0], 1),get_values(6, features[1], 1), c = "red",s=size, alpha=alpha)
ax[0,1].scatter(get_values(6, features[0], 0), get_values(6, features[1], 0), c = "blue",s=size, alpha=alpha)
ax[1,0].scatter(get_values(12, features[0], 1), get_values(12, features[1], 1), c = "red",s=size, alpha=alpha)
ax[1,1].scatter(get_values(12, features[0], 0), get_values(12, features[1], 0), c = "blue",s=size, alpha=alpha)
for n in range(2):
    for m in range(2):
        ax[n,m].set_xlabel(features[0])
        ax[n,m].set_ylabel(features[1])
ax[0,0].set_title("Anomalies in Cluster 6")
ax[0,1].set_title("Normal spots in Cluster 6")
ax[1,0].set_title("Anomalies in Cluster 12")
ax[1,1].set_title("Normal spots in Cluster 12")

ax[0,0].set_xlim([0,4000])
ax[0,1].set_xlim([0,4000])
ax[1,0].set_xlim([0,4000])
ax[1,1].set_xlim([0,4000])
ax[0,0].set_ylim([0,4000])
ax[0,1].set_ylim([0,4000])
ax[1,0].set_ylim([0,4000])
ax[1,1].set_ylim([0,4000])

# ### Take-away
# 
# * This plot looks so easy! But it has the most impact for further analysis!
# * On the left hand side you can see products that were marked as "anormal data" and on the right hand side you can see what the model calles "nomal". 
# * Cluster 6 is our anomaly cluster that spreads widely into the feature space. We can see that the detection of anomalies worked well for its cluster members. Every spot that shows a discrepancy between energy and the reconstructed energy based on the typed-in nutrients was detected as anomal. That's what we want to have!
# * Cluster 12 in contrast sits with his back on the zero-zero-plane of fats and proteins. All data spots in this cluster have zero fat and zero proteins. For all other features the cluster spreads widely into the space like cluster 6 does. **Now your alarm bells should ring** ... Why are both clusters that different in anomaly detection? Why went something wrong with cluster 12? We still expect to detect errors of energy discrepancies as anomalies.
# * **BOOM** Here it comes out very clear - our model assumptions don't fit to our data! We used gaussian mixture to show its advantages over simple clusterings like k-means and we transformed features to make them more like gaussians. **But you see... our data is neither completely continouus nor discrete.** We have discrete zero-entries per feature that do not fit to a gaussian and continuous values that spread widely and would fit to a gaussian approach. 
# * This plot motivates to introduce and build a more sophisticated model that can deal with this half-half nature of our data. We need to somehow glue Bernoullis and Gaussians to lead this model to success.

# ## How certain is our model about its anomalies in cluster assignment ?

# In[ ]:


plt.figure(figsize=(20,5))
sns.boxplot(x="cluster", y="certainty", data=nutrition_table[nutrition_table.anomaly==1])
plt.title("Certainty of anomalies")

# ### Take-Away
# 
# * We can find some very interesting patterns: In cluster 5 (water), 6 (mixed outlier cluster), 12 (sweets & sauces) and 17 (oils and cream dressings) the model is very certain in its cluster assignments in most of the anomaly cases! 
# * In contrast cluster 3 (chocolate & cookies) and 1 (potatoes and beans) the model is very uncertain (below 0.6) about the cluster assignments of the majority of its anomalies. 
# * Looking at all clusters we can see that most of them spread widely in their certainties.

# ## How do anomalies of normal clusters look like?

# In[ ]:


fig, ax = plt.subplots(2,2, figsize=(20,10))
make_word_cloud(nutrition_table[nutrition_table.anomaly==1], 1, ax[0,0], "Anomalies in Bean Cluster 1")
make_word_cloud(nutrition_table[nutrition_table.anomaly==1], 13, ax[0,1], "Anomalies in Bread Cluster 13")
make_word_cloud(nutrition_table[nutrition_table.anomaly==1], 9, ax[1,0], "Anomalies in Pasta Cluster 9")
make_word_cloud(nutrition_table[nutrition_table.anomaly==1], 16, ax[1,1], "Anomalies in Grain Cluster 16")

# ### Take-Away
# 
# * By looking at the most common product names of anomalies per cluster we can see different kinds:
#     *  We can find **outlier products like blackeye peas, pumpernickel, puffed grain, products without gluten**
#     * But we can also find **seldom products** that do not fit to their clusters like **sugarfree products** in pasta cluster for example
# * Besides these outliers there are probably user errors of the data type-in input process as well. We don't know! 

# ### Visualizing anomalies

# In[ ]:


plot_features = ["energy_100g", "reconstructed_energy", "g_sum"]

# In[ ]:


trace2 = go.Scatter3d(
    x=nutrition_table[plot_features[0]].values[0:N], 
    y=nutrition_table[plot_features[1]].values[0:N],
    z=nutrition_table[plot_features[2]].values[0:N],
    mode='markers',
    marker=dict(
        color=nutrition_table.anomaly.values,
        colorscale = "Portland",
        opacity=0.8,
        size=1.5
    )
)

figure_data = [trace2]
layout = go.Layout(
    title = 'Log3D of Proteins, Fat and Carbohydrates',
    scene = dict(
        xaxis = dict(title=plot_features[0]),
        yaxis = dict(title=plot_features[1]),
        zaxis = dict(title=plot_features[2]),
    ),
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=0
    ),
    showlegend=True
)

fig = go.Figure(data=figure_data, layout=layout)
py.iplot(fig, filename='simple-3d-scatter')

# In[ ]:


nutrition_table.to_csv("hidden_treasures_groceries_gmm.csv")

# # Conclusion <a class="anchor" id="conclusion"></a>
# 
# * The Gaussian Mixture Model is able to cluster the data in a useful way depending on their nutrition-tables. We have found various type of clusters that hold differnt product categories.
# * The model can even tell us how certain it is about the clustering assignment. 
# * The probabilistic nature of the model can tell us something about competing clusters as well. Competitors could be a good choice for some products. Sometimes the competitors are neighborhood clusters but additionally we found some attractions towards a mixed outlier clusters.
# * We can also find anomalies (both rare and wrong entries).
# * There is also an anomaly-cluster - that holds many outliers, wrong entries and seldom products.
# * We found that there is a strong need to build a model that takes into account the discrete nature of zero-entries like zero fat or zero proteins. This would probably have a huge impact to the quality of clustering results, certainty and anomaly detection!
# * Besides these benefits the model needs much predatory work (cleaning, transforming, scaling the data)
# * It is also really depending on the decisions we make (e.g. the transformation-features, the number of components (cluster), the threshold we set to define an anomaly). Consequently hyperparameter selection is the most crucial part.
# * But the model seems to be worth this work! It has much more to offer than a bunch of other clustering algorithms like K-Means for example.

# # Outlook <a class="anchor" id="outlook"></a>

# * We should try to find an automatic and well working way to choose the transformation-features and the number of components so our model gets more precisely and doesn't depend on our manual choices too much. It would also help when we add alot of new and different products because this would probably change our distributions and therefore our transformation-features and the number of components.
# * There is a strong [need to build a more sophisticated model](https://www.kaggle.com/allunia/lighthouses-for-hidden-treasures-in-our-groceries) that can deal with discrete zero-entries and continous features higher than zero. For this model we have to setup the Expectation-Maximization learning and we have to show that this model works much better.
# * First of all we should test our model on new, unseen data to see if it can also work with them correctly. As the input of new features is a saturating procedure, there will be less common products in the future and perhaps more specials. In addition you can use the app from all-over the world. Consequently the data is time- and space- dependent. For this reason a good strategy for refitting and validation is needed. This could include user feedbacks as well to make sure that predicted anomalies are really anomalies or if an uncertain product could be placed into another cluster. 
# * To make our model stronger we could include more features like the ingredients list or the product names we always had peeked at. This could mean to introduce a further clustering based on these features or to build a model that deals with all features at once each update step.
# * Luckily gaussian mixture can be split into parallel working data chunks. As responsibilites are calculated per data spot and the center  & covariance updates are sums that can be split there is no barrier for bigdata without exploding number of components.
# 
# 
# 
