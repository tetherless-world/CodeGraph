#!/usr/bin/env python
# coding: utf-8

# In[292]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

# In[ ]:




# In[293]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
feats = [c for c in df.columns if c not in ['codigo_mun', 'comissionados_por_servidor']]
df=test.append(train)



# In[294]:


#df.head().T
#test.head().T
#train.head().T # nota_mat

# In[295]:


regx = re.compile('\(\d+\)')
regx2 = re.compile('\,?\.?')
df['populacao']=df['populacao'].str.replace(regx2,'').str.replace(regx, '').astype(dtype=np.int)

# In[296]:


df['area']=df['area'].str.replace(',','').astype(float)

# In[297]:


df['densidade_dem']=df['densidade_dem'].str.replace(',','').astype(float)

df['cat_porte'] = df['porte'].astype('category').cat.codes
df['cat_regiao'] = df['regiao'].astype('category').cat.codes
df['cat_estado'] = df['estado'].astype('category').cat.codes

# In[298]:


# campos nulos: densidade_dem, participacao_transf_receita, servidores, perc_pop_econ_ativa,
# gasto_pc_saude, hab_p_medico, exp_vida, gasto_pc_educacao, exp_anos_estudo

#feats = [c for c in df.columns if c not in ['codigo_mun', 'comissionados_por_servidor','nota_mat']]
feats = [c for c in df.columns if c not in ['codigo_mun', 'comissionados_por_servidor','nota_mat', 'densidade_dem', 
                                            'participacao_transf_receita', 'servidores', 'perc_pop_econ_ativa', 
                                            'gasto_pc_saude', 'hab_p_medico', 'exp_vida', 'gasto_pc_educacao', 
                                            'exp_anos_estudo', 'regiao', 'estado', 'porte', 'municipio']]

# In[299]:


df.shape, train.shape, test.shape

# In[300]:


train=df[~df.nota_mat.isnull()]
test_submissao=df[df.nota_mat.isnull()]

# In[301]:


from sklearn.model_selection import train_test_split
train_2, test = train_test_split(train, test_size=0.20, random_state=42)
train_2, valid = train_test_split(train_2, test_size=0.20, random_state=42)
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(random_state=42, n_estimators=100, min_samples_split=5, max_depth=4)


# In[ ]:




# In[302]:


rf.fit(train_2[feats], train_2['nota_mat'])
preds = rf.predict(valid[feats])


# In[303]:


from sklearn.metrics import accuracy_score
accuracy_score(valid['nota_mat'], preds)


# In[304]:


accuracy_score(test['nota_mat'], rf.predict(test[feats]))


# In[305]:


pd.Series(rf.feature_importances_,index=feats).sort_values().plot.barh()

# In[306]:


from sklearn.ensemble import RandomForestRegressor
rf=RandomForestRegressor(random_state=42, n_estimators=100)
rf.fit(train_2[feats], train_2['nota_mat'])
preds = rf.predict(valid[feats])
from sklearn.metrics import mean_squared_error
mean_squared_error(valid['nota_mat'], preds)**(1/2) 


# In[307]:


from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor(random_state=44)
dt.fit(train_2[feats], train_2['nota_mat'])
dt.predict(train_2[feats])


from sklearn import tree
import graphviz
from IPython.display import SVG
from IPython.display import display
data = tree.export_graphviz (dt, out_file=None, feature_names=feats, class_names=['nota_mat'],
                            filled=True, rounded=True, special_characters=True, max_depth=2)

# In[308]:


graph=graphviz.Source(data)
display(SVG(graph.pipe(format='svg')))
