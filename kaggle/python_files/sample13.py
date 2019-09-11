#!/usr/bin/env python
# coding: utf-8

# ![](http://images.trendingstories.net/es/s/Ctrl?Command=ScaleImage&usecache=1&maxw=600&maxh=1200&red=0&if=jpg&imgurl=http%3A%2F%2Fc512cfd2a3775cdd40e6-222afba0d566c26fa1accb64e6414783.r55.cf1.rackcdn.com%2FLifestyle%2FBest%2520iPad%2520Apps%2520You%2520Probably%2520Dont%2520Have%2FIcon-collage.jpg)
# 
# ## 6 Interesting Questions to ask
# 
# 1. How do you visualize price distribution of paid apps ?
# 2. How does the price distribution get affected by category ?
# 3. What about paid apps Vs Free apps ?
# 4. Are paid apps good enough ?
# 5. As the size of the app increases do they get pricier ?
# 6. How are the apps distributed category wise ? can we split by paid category ?

# In[ ]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

# # Understand what is the data set about

# - The data set comprises of information on 7200 apps on App store with following imp details
# 
#     - "id" : App ID
#     - "track_name": App Name
#     - "size_bytes": Size (in Bytes)
#     - "price": Price amount
#     - "rating_count_tot": User Rating counts (for all version)
#     - "rating_count_ver": User Rating counts (for current version)
#     - "prime_genre": Primary Genre

# In[ ]:


add = "../input/AppleStore.csv"
data = pd.read_csv(add)
data.head()


# ## 1. How do you visualize price distribution of paid apps ?

# In[ ]:


#fact generator 
print ('1. Free apps are ' + str(sum(data.price == 0)))
print ('2. Counting (outliers) super expensive apps ' + str(sum(data.price > 50)))
print (' -  which is around ' + str(sum(data.price > 50)/len(data.price)*100) +
       " % of the total Apps")
print (' Thus we will dropping the following apps')
outlier=data[data.price>50][['track_name','price','prime_genre','user_rating']]
freeapps = data[data.price==0]
outlier

# In[ ]:


# removing
paidapps =data[((data.price<50) & (data.price>0))]
print('Now the max price of any app in new data is : ' + str(max(paidapps.price)))
print('Now the min price of any app in new data is : ' + str(min(paidapps.price)))
#paidapps.prime_genre.value_counts()

# In[ ]:


plt.style.use('fivethirtyeight')
plt.figure(figsize=(15,15))
plt.subplot(2,1,1)

plt.hist(paidapps.price,log=True)
plt.title('Price distribution of apps (Log scale)')
plt.ylabel("Frequency Log scale")
plt.xlabel("Price Distributions in ($) ")

plt.subplot(2,1,2)
plt.title('Visual price distribution')
sns.stripplot(data=paidapps,y='price',jitter= True,orient = 'h' ,size=6)
plt.show()

# ## Insights
# 1. Count of paid apps is exponentially decreases as the price increases
# 2. Very few apps have been priced above 30 \$. So its important to keep price of your app below 30$

# ## 2. How does the price distribution get affected by category ?
# 

# In[ ]:


yrange = [0,25]
fsize =15

plt.figure(figsize=(15,10))

plt.subplot(4,1,1)
plt.xlim(yrange)
games = paidapps[paidapps.prime_genre=='Games']
sns.stripplot(data=games,y='price',jitter= True , orient ='h',size=6,color='#eb5e66')
plt.title('Games',fontsize=fsize)
plt.xlabel('') 

plt.subplot(4,1,2)
plt.xlim(yrange)
ent = paidapps[paidapps.prime_genre=='Entertainment']
sns.stripplot(data=ent,y='price',jitter= True ,orient ='h',size=6,color='#ff8300')
plt.title('Entertainment',fontsize=fsize)
plt.xlabel('') 

plt.subplot(4,1,3)
plt.xlim(yrange)
edu = paidapps[paidapps.prime_genre=='Education']
sns.stripplot(data=edu,y='price',jitter= True ,orient ='h' ,size=6,color='#20B2AA')
plt.title('Education',fontsize=fsize)
plt.xlabel('') 

plt.subplot(4,1,4)
plt.xlim(yrange)
pv = paidapps[paidapps.prime_genre=='Photo & Video']
sns.stripplot(data=pv,y='price',jitter= True  ,orient ='h',size=6,color='#b84efd')
plt.title('Photo & Video',fontsize=fsize)
plt.xlabel('') 

plt.show()

# ### Insights
# - Paid gaming apps are highly priced and distribution extends till 25 $
# - Paid Entertainment apps have a lower price range

# ## 3. What about paid apps Vs Free apps ?
# 

# In[ ]:


print("There are total of " + str(len(data.prime_genre.value_counts().index)) 
      + " categories which is little too much")
print ("Lets limit our categories to 5")

# In[ ]:


# reducing the number of categories

s = data.prime_genre.value_counts().index[:4]
def categ(x):
    if x in s:
        return x
    else : 
        return "Others"

data['broad_genre']= data.prime_genre.apply(lambda x : categ(x))

# In[ ]:


free = data[data.price==0].broad_genre.value_counts().sort_index().to_frame()
paid = data[data.price>0].broad_genre.value_counts().sort_index().to_frame()
total = data.broad_genre.value_counts().sort_index().to_frame()
free.columns=['free']
paid.columns=['paid']
total.columns=['total']
dist = free.join(paid).join(total)
dist ['paid_per'] = dist.paid*100/dist.total
dist ['free_per'] = dist.free*100/dist.total
dist

# In[ ]:


list_free= dist.free_per.tolist()
tuple_free = tuple(list_free)
tuple_paidapps = tuple(dist.paid_per.tolist())

# In[ ]:


plt.figure(figsize=(15,8))
N=5
ind = np.arange(N)    # the x locations for the groups
width =0.56   # the width of the bars: can also be len(x) sequence

p1 = plt.bar(ind, tuple_free, width, color='#45cea2')
p2 = plt.bar(ind, tuple_paidapps, width,bottom=tuple_free,color='#fdd470')
plt.xticks(ind,tuple(dist.index.tolist() ))
plt.legend((p1[0], p2[0]), ('free', 'paid'))
plt.show()
# for pie chart
pies = dist[['free_per','paid_per']]
pies.columns=['free %','paid %']
plt.show()

# In[ ]:


plt.figure(figsize=(15,8))
pies.T.plot.pie(subplots=True,figsize=(20,4),colors=['#45cea2','#fdd470'])
plt.show()

# ### Insights
# - Education has significant % of Paid apps.
# - On the contrary - Entertainment category hosts high % of free apps

# ## 4. Are paid apps good enough ?

# In[ ]:


def paid(x):
    if x>0:
        return 'Paid'
    else :
        return'Free'

data['category']= data.price.apply(lambda x : paid(x))
data.tail()


# 

# In[ ]:


plt.figure(figsize=(15,8))
plt.style.use('fast')
plt.ylim([0,5])
plt.title("Distribution of User ratings")
sns.violinplot(data=data, y ='user_rating',x='broad_genre',hue='category',
               vertical=True,kde=False,split=True ,linewidth=2,
               scale ='count', palette=['#fdd470','#45cea2'])
plt.xlabel(" ")
plt.ylabel("Rating (0-5)")

plt.show()

# #### trick :  Why cant we use swarm plot ? 
#  - Swarm is a non overlap plot i.e it will plot each point seperately on the graph.
#  - If the number of points on particular value is a lot - lets say 100.
#  - It will have to plot each value seperately on the graph
#  - the graph will expand horizontally 

# ## 5. As the size of the app increases do they get pricier ?
# 

# In[ ]:


sns.color_palette("husl", 8)
sns.set_style("whitegrid")
flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
data ['MB']= data.size_bytes.apply(lambda x : x/1048576)
paidapps_regression =data[((data.price<30) & (data.price>0))]
sns.lmplot(data=paidapps_regression,
           x='MB',y='price',size=4, aspect=2,col_wrap=2,hue='broad_genre',
           col='broad_genre',fit_reg=False,palette = sns.color_palette("husl", 5))
plt.show()

# #### Answer : NO ! 

# ## 6. How are the apps distributed category wise ? can we split by paid category ?

# In[ ]:


BlueOrangeWapang = ['#fc910d','#fcb13e','#239cd3','#1674b1','#ed6d50']
plt.figure(figsize=(10,10))
label_names=data.broad_genre.value_counts().sort_index().index
size = data.broad_genre.value_counts().sort_index().tolist()
my_circle=plt.Circle( (0,0), 0.5, color='white')
plt.pie(size, labels=label_names, colors=BlueOrangeWapang)
p=plt.gcf()
p.gca().add_artist(my_circle)
plt.show()


# In[ ]:


plt.figure(figsize=(10,10))
f=pd.DataFrame(index=np.arange(0,10,2),data=dist.free.values,columns=['num'])
p=pd.DataFrame(index=np.arange(1,11,2),data=dist.paid.values,columns=['num'])
final = pd.concat([f,p],names=['labels']).sort_index()
final.num.tolist()

plt.figure(figsize=(20,20))
group_names=data.broad_genre.value_counts().sort_index().index
group_size=data.broad_genre.value_counts().sort_index().tolist()
h = ['Free', 'Paid']
subgroup_names= 5*h
sub= ['#45cea2','#fdd470']
subcolors= 5*sub
subgroup_size=final.num.tolist()


# First Ring (outside)
fig, ax = plt.subplots()
ax.axis('equal')
mypie, _ = ax.pie(group_size, radius=2.5, labels=group_names, colors=BlueOrangeWapang)
plt.setp( mypie, width=1.2, edgecolor='white')

# Second Ring (Inside)
mypie2, _ = ax.pie(subgroup_size, radius=1.6, labels=subgroup_names, labeldistance=0.7, colors=subcolors)
plt.setp( mypie2, width=0.8, edgecolor='white')
plt.margins(0,0)

# show it
plt.show()


# # THE END
