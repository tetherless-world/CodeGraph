#!/usr/bin/env python
# coding: utf-8

# In[82]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import xgboost as xgb
import lightgbm as lgbm
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier,GradientBoostingRegressor,GradientBoostingClassifier
from sklearn.metrics import confusion_matrix,accuracy_score,r2_score,roc_auc_score,precision_score,recall_score,f1_score
from sklearn.model_selection import KFold,GridSearchCV,RandomizedSearchCV

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

# In[83]:


columns=["id","cycle","op1","op2","op3","sensor1","sensor2","sensor3","sensor4","sensor5","sensor6","sensor7","sensor8",
         "sensor9","sensor10","sensor11","sensor12","sensor13","sensor14","sensor15","sensor16","sensor17","sensor18","sensor19"
         ,"sensor20","sensor21","sensor22","sensor23"]

# In[84]:


train=pd.read_csv("../input/train_FD001.txt",sep=" ",names=columns)
test=pd.read_csv("../input/test_FD001.txt",sep=" ",names=columns)
gercek_sonuc=pd.read_csv("../input/RUL_FD001.txt",sep=" ",header=None)

# In[85]:


train.head()

# In[86]:


test.head()

# In[87]:


gercek_sonuc.columns=["rul","bos"]
gercek_sonuc.head()

# In[88]:


gercek_sonuc.drop(["bos"],axis=1,inplace=True)
gercek_sonuc['id']=gercek_sonuc.index+1
gercek_sonuc.head()

# In[89]:


rul = pd.DataFrame(test.groupby('id')['cycle'].max()).reset_index()
rul.columns = ['id', 'max']

# In[90]:


rul.head()

# In[91]:


gercek_sonuc['rul_failed']=gercek_sonuc['rul']+rul['max']
gercek_sonuc.head()

# In[92]:


gercek_sonuc.drop(["rul"],axis=1,inplace=True)

# In[93]:


test=test.merge(gercek_sonuc,on=['id'],how='left')

# In[94]:


test["kalan_cycle"]=test["rul_failed"]-test["cycle"]
test.head()

# In[95]:


df_train=train.drop(["sensor22","sensor23"],axis=1)
df_test=test.drop(["sensor22","sensor23"],axis=1)

# In[96]:


df_test.drop(["rul_failed"],axis=1,inplace=True)

# In[97]:


df_train['kalan_cycle'] = df_train.groupby(['id'])['cycle'].transform(max)-df_train['cycle']

# In[98]:


df_train.head()

# In[99]:


devir=30
df_train['label'] = df_train['kalan_cycle'].apply(lambda x: 1 if x <= devir else 0)
df_test['label'] = df_test['kalan_cycle'].apply(lambda x: 1 if x <= devir else 0)

# In[100]:


op_set=["op"+str(i) for i in range(1,4)]
sensor=["sensor"+str(i) for i in range(1,22)]
sb.pairplot(train.query("cycle"),x_vars=op_set,y_vars=sensor)

# In[101]:


sb.pairplot(test.query("cycle"),x_vars=op_set,y_vars=sensor)

# In[102]:


a=train.drop(["id","cycle","sensor22","sensor23"],axis=1)
b=df_train.label

# In[103]:


xgb_reg=xgb.XGBRegressor()
xgb_reg.fit(a,b)
b_pred_xgb_reg=xgb_reg.predict(a)
print("R Square: ",r2_score(b,b_pred_xgb_reg))

# In[104]:


onemli_degiskenler = pd.Series(data=xgb_reg.feature_importances_,index=a.columns)
onemli_degiskenler.sort_values(ascending=False,inplace=True)
print("Verimizdeki sonuçlara göre en iyi değişkenler:\n",onemli_degiskenler)

# In[105]:


df_train.drop(["id","cycle","op1","op2","op3","sensor1","sensor5","sensor6","sensor10","sensor16","sensor18","sensor19"],axis=1,inplace=True)
df_test.drop(["id","cycle","op1","op2","op3","sensor1","sensor5","sensor6","sensor10","sensor16","sensor18","sensor19"],axis=1,inplace=True)

# In[106]:


df_train.head()

# In[107]:


df_train = df_train.drop(['kalan_cycle'],axis=1)

# In[108]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(df_train.iloc[0:-1:,0:-1],df_train.iloc[0:-1:,-1], test_size=0.2, random_state=3)
# gc.collect()  
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# In[109]:


import lightgbm as lgb
lgb_clss = lgb.LGBMClassifier(learning_rate=0.01,n_estimators=5000,num_leaves=100,objective='binary', metrics='auc',random_state=50,n_jobs=-1)
lgb_clss.fit(X_train, y_train)
lgb_clss.score(X_test, y_test)
preds2 = lgb_clss.predict(X_test)
print('Acc Score: ',accuracy_score(y_test, preds2))
print('Roc Auc Score: ',roc_auc_score(y_test, preds2))
print('Doğru ve Yanlışlık oranı - Hassasiyet Score: ',precision_score(y_test, preds2))
print('Pozitif Örnek hatırlama oranı: ',recall_score(y_test, preds2))
print('f1 score: ',f1_score(y_test, preds2,'binary'))

# In[111]:


cv=KFold(5)
xgb_clss=xgb.XGBClassifier()
param= {'learning_rate':[0.01],
 'max_depth':[8],
 'min_child_weight':[11],
 'n_estimators':[500]
}
rs_cv=RandomizedSearchCV(estimator=xgb_clss,cv=cv,param_distributions=param,n_jobs=-1)
rs_cv.fit(X_train,y_train)
xgb_pred=rs_cv.predict(X_test)
print("XGB ACC Score: ",accuracy_score(y_test,xgb_pred))

# In[112]:


df_test_pred = rs_cv.predict(df_test.drop(['kalan_cycle','label'],axis=1))
cm=confusion_matrix(df_test.iloc[:,-1], df_test_pred, labels=None, sample_weight=None)

# In[113]:


print("Test Accuracy Score: ", accuracy_score(df_test.iloc[:,-1],df_test_pred))

# In[114]:


sb.heatmap(cm,annot=True,linewidths=0.7,linecolor="black",cmap="YlGnBu",fmt="d")
