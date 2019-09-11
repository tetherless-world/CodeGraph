#!/usr/bin/env python
# coding: utf-8

# Hi, Kagglers!
# 
# Hereafter I will try to create **some baselines for your submissions to start from.**
# <br> This Kernel touches **submission part mostly AND ONE WELL-KNOWN FILM :)**
# <br/>For more details about Dataset - please check my **[Data Exploration Kernel](https://www.kaggle.com/frednavruzov/instacart-exploratory-data-analysis/)**
# 
# **Brief description**
# 
# The Dataset is an anonymized sample of over 3,000,000 grocery orders from more than 200,000 Instacart users. 
# <br>The goal of a competition is to predict which previously purchased products will be in a user’s next order. 
# 
# ### Stay tuned, this notebook will be updated on a regular basis
# **P.s. Upvotes and comments would let me update it faster and in a more smart way :)**

# In[ ]:


import pandas as pd # dataframes
import numpy as np # algebra & calculus
import nltk # text preprocessing & manipulation
# from textblob import TextBlob
import matplotlib.pyplot as plt # plotting
import seaborn as sns # plotting

from functools import partial # to reduce df memory consumption by applying to_numeric

color = sns.color_palette() # adjusting plotting style
import warnings
warnings.filterwarnings('ignore') # silence annoying warnings

# In[ ]:


# aisles
aisles = pd.read_csv('../input/aisles.csv', engine='c')
print('Total aisles: {}'.format(aisles.shape[0]))
aisles.head()

# In[ ]:


# departments
departments = pd.read_csv('../input/departments.csv', engine='c')
print('Total departments: {}'.format(departments.shape[0]))
departments.head()

# In[ ]:


# products
products = pd.read_csv('../input/products.csv', engine='c')
print('Total products: {}'.format(products.shape[0]))
products.head(5)

# In[ ]:


# combine aisles, departments and products (left joined to products)
goods = pd.merge(left=pd.merge(left=products, right=departments, how='left'), right=aisles, how='left')
# to retain '-' and make product names more "standard"
goods.product_name = goods.product_name.str.replace(' ', '_').str.lower() 
print(goods.info())

goods.head()

# In[ ]:


# load datasets

# train dataset
op_train = pd.read_csv('../input/order_products__train.csv', engine='c', 
                       dtype={'order_id': np.int32, 'product_id': np.int32, 
                              'add_to_cart_order': np.int16, 'reordered': np.int8})
print('Total ordered products(train): {}'.format(op_train.shape[0]))
op_train.head(10)

# In[ ]:


# test dataset (submission)
test = pd.read_csv('../input/sample_submission.csv', engine='c')
print('Total orders(test): {}'.format(test.shape[0]))
test.head()

# In[ ]:


#prior dataset
op_prior = pd.read_csv('../input/order_products__prior.csv', engine='c', 
                       dtype={'order_id': np.int32, 
                              'product_id': np.int32, 
                              'add_to_cart_order': np.int16, 
                              'reordered': np.int8})

print('Total ordered products(prior): {}'.format(op_prior.shape[0]))
op_prior.head()

# In[ ]:


# orders
orders = pd.read_csv('../input/orders.csv', engine='c', dtype={'order_id': np.int32, 
                                                           'user_id': np.int32, 
                                                           'order_number': np.int32, 
                                                           'order_dow': np.int8, 
                                                           'order_hour_of_day': np.int8, 
                                                           'days_since_prior_order': np.float16})
print('Total orders: {}'.format(orders.shape[0]))
print(orders.info())
orders.head()

# ### Combine (orders, order details, product hierarchy) into 1 dataframe order_details 
# **(be careful, high memory consumption, about 3GB RAM itself)**

# In[ ]:


from functools import partial

# merge train and prior together iteratively, to fit into 8GB kernel RAM
# split df indexes into parts
indexes = np.linspace(0, len(op_prior), num=10, dtype=np.int32)

# initialize it with train dataset
order_details = pd.merge(
                left=op_train,
                 right=orders, 
                 how='left', 
                 on='order_id'
        ).apply(partial(pd.to_numeric, errors='ignore', downcast='integer'))

# add order hierarchy
order_details = pd.merge(
                left=order_details,
                right=goods[['product_id', 
                             'aisle_id', 
                             'department_id']].apply(partial(pd.to_numeric, 
                                                             errors='ignore', 
                                                             downcast='integer')),
                how='left',
                on='product_id'
)

print(order_details.shape, op_train.shape)

# delete (redundant now) dataframes
del op_train

order_details.head()

# In[ ]:


# update by small portions
for i in range(len(indexes)-1):
    order_details = pd.concat(
        [   
            order_details,
            pd.merge(left=pd.merge(
                            left=op_prior.loc[indexes[i]:indexes[i+1], :],
                            right=goods[['product_id', 
                                         'aisle_id', 
                                         'department_id' ]].apply(partial(pd.to_numeric, 
                                                                          errors='ignore', 
                                                                          downcast='integer')),
                            how='left',
                            on='product_id'
                            ),
                     right=orders, 
                     how='left', 
                     on='order_id'
                ) #.apply(partial(pd.to_numeric, errors='ignore', downcast='integer'))
        ]
    )
        
print('Datafame length: {}'.format(order_details.shape[0]))
print('Memory consumption: {:.2f} Mb'.format(sum(order_details.memory_usage(index=True, 
                                                                         deep=True) / 2**20)))
# check dtypes to see if we use memory effectively
print(order_details.dtypes)

# make sure we didn't forget to retain test dataset :D
test_orders = orders[orders.eval_set == 'test']

# delete (redundant now) dataframes
del op_prior, orders

# ### 1. Greedy Dumb Submission :) <br>(0.2164845 Public LB Score)
# ![Lloyd level, still better than banana baseline][1]
# 
# 
#   [1]: http://www.punchnels.com/wp-content/uploads/Jim_Carrey_Dumb-and-Dumber-Inside.jpg

# In[ ]:


# dumb submission
test_history = order_details[(order_details.user_id.isin(test_orders.user_id))]\
.groupby('user_id')['product_id'].apply(lambda x: ' '.join([str(e) for e in set(x)])).reset_index()
test_history.columns = ['user_id', 'products']

test_history = pd.merge(left=test_history, 
                        right=test_orders, 
                        how='right', 
                        on='user_id')[['order_id', 'products']]

test_history.to_csv('dumb_submission.csv', encoding='utf-8', index=False)

# ### Still Greedy But Smarter (Customer Will Take All Reordered) <br>(0.2996690 Public LB Score)
# ![Lloyd appreciates this][1]
# 
# 
#   [1]: https://i.ytimg.com/vi/CHCmLxqTPOs/hqdefault.jpg

# In[ ]:


# dumb submission
test_history = order_details[(order_details.user_id.isin(test_orders.user_id)) 
                             & (order_details.reordered == 1)]\
.groupby('user_id')['product_id'].apply(lambda x: ' '.join([str(e) for e in set(x)])).reset_index()
test_history.columns = ['user_id', 'products']

test_history = pd.merge(left=test_history, 
                        right=test_orders, 
                        how='right', 
                        on='user_id')[['order_id', 'products']]

test_history.to_csv('dumb2_subm.csv', encoding='utf-8', index=False)

# ### Less Dumb - Repeat Last Order<br>(0.3276746 Public LB Score)
# ![Lloyd appreciates this][1]
# 
# 
#   [1]: https://www.spin1038.com/content/000/images/000052/54551_60_news_hub_multi_630x0.png

# In[ ]:


test_history = order_details[(order_details.user_id.isin(test_orders.user_id))]
last_orders = test_history.groupby('user_id')['order_number'].max()

def get_last_orders():
    t = pd.merge(
            left=pd.merge(
                    left=last_orders.reset_index(),
                    right=test_history,
                    how='inner',
                    on=['user_id', 'order_number']
                )[['user_id', 'product_id']],
            right=test_orders[['user_id', 'order_id']],
            how='left',
            on='user_id'
        ).groupby('order_id')['product_id'].apply(lambda x: ' '.join([str(e) for e in set(x)])).reset_index()
    t.columns = ['order_id', 'products']
    return t

# save submission
get_last_orders().to_csv('less_dumb_subm_last_order.csv', encoding='utf-8', index=False)

# ### Less Dumb - Repeat Last Order (Reordered Products Only)<br>(0.3276826 Public LB Score)
# ![America is great again!][1]
# 
# 
#   [1]: http://www.totalprosports.com/wp-content/uploads/2016/11/lloyd-and-harry-dumb-and-dumber.jpg

# In[ ]:


test_history = order_details[(order_details.user_id.isin(test_orders.user_id))]
last_orders = test_history.groupby('user_id')['order_number'].max()

def get_last_orders_reordered():
    t = pd.merge(
            left=pd.merge(
                    left=last_orders.reset_index(),
                    right=test_history[test_history.reordered == 1],
                    how='left',
                    on=['user_id', 'order_number']
                )[['user_id', 'product_id']],
            right=test_orders[['user_id', 'order_id']],
            how='left',
            on='user_id'
        ).fillna(-1).groupby('order_id')['product_id'].apply(lambda x: ' '.join([str(int(e)) for e in set(x)]) 
                                                  ).reset_index().replace(to_replace='-1', 
                                                                          value='None')
    t.columns = ['order_id', 'products']
    return t

# save submission
get_last_orders_reordered().to_csv('less_dumb_subm_last_order_reordered_only.csv', 
                         encoding='utf-8', 
                         index=False)

# ### To be continued... 
# 
# **(TODO: more creative baselines)**

# ### Stay tuned, this notebook will be updated on a regular basis
# **P.s. Upvotes and comments would let me update it faster and in a more smart way :)**
