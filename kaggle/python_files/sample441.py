#!/usr/bin/env python
# coding: utf-8

# The main aim of this kernel is to use Linear regression in order to figure out how to maximize the Yearly Amount spent by the customers.

# In[ ]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
print(os.listdir("../input"))
# Any results you write to the current directory are saved as output.

# Importing data 

# In[ ]:


customer_df=pd.read_csv('../input/Ecommerce Customers.csv')
customer_df.head()

# Lets explore the data

# In[ ]:


customer_df.describe()

# In[ ]:


customer_df.info()

#  **Lets Start Exploring the data to learn more..**

# In[ ]:


sns.set_palette('GnBu_r')

# Comparing time spent on the site and amount of money spent yearly

# In[ ]:


sns.jointplot(x='Time on Website',y='Yearly Amount Spent',data=customer_df)
plt.show()

# Comparing time spent on the app and amount of money spent yearly

# In[ ]:


sns.jointplot(x='Time on App',y='Yearly Amount Spent',data=customer_df)
plt.show()

# Lets Compare the time spent on the app and the length of membership

# In[ ]:


sns.jointplot(x='Time on App',y='Length of Membership',data=customer_df,kind='hex')
plt.show()

# Lets Compare the time spent on the website and the length of membership

# In[ ]:


sns.jointplot(x='Time on Website',y='Length of Membership',data=customer_df,kind='hex')
plt.show()

# We could simply explore similar relationships across the data by creating pairplots

# In[ ]:


sns.pairplot(customer_df)
plt.show()

# From the pairplot above we can observe that the length of membership is most closely related to the yearly amount spent by the consumers.

# In[ ]:


sns.lmplot(x='Length of Membership',y='Yearly Amount Spent',data=customer_df)
plt.show()

# Lets Split the data into training and testing set. We will set 'y' as the Yearly Amount Spent and 'X' will be the numerical features related to the customers from the data 

# In[ ]:


y=customer_df['Yearly Amount Spent']
X=customer_df[['Avg. Session Length','Time on App','Time on Website','Length of Membership']]

# In[ ]:


#Train-test split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.4,random_state=42)
# I have set random_state=42 in order to get the same output every time i run this kernel

# In[ ]:


#creating and training the model
from sklearn.linear_model import LinearRegression
lm=LinearRegression()
lm.fit(X_train,y_train)
#coefficients of the model
print('Coefficients: \n',lm.coef_)

# Now that we have our fit model lets see how well we can predict the test values

# In[ ]:


predictions =lm.predict(X_test)

# Lets create a scatter plot of real test values vs the predicted values

# In[ ]:


plt.scatter(y_test,predictions)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')
plt.show()

# Now lets evaluate the model

# In[ ]:


from sklearn import metrics
print('MAE= ', metrics.mean_absolute_error(y_test,predictions))
print('MSE= ', metrics.mean_squared_error(y_test,predictions))
print('RMSE= ', np.sqrt(metrics.mean_squared_error(y_test,predictions)))

# Now that we have a good model Lets explore the residuals

# In[ ]:


sns.distplot((y_test-predictions),bins=40);

# Now lets figure out how we can boost the yearly amount spent by the customers

# In[ ]:


cdf=pd.DataFrame(lm.coef_,X.columns,columns=['Coeff'] )
cdf

# We can interpret this data to make certain observations, for example:-
# If all other factors are kept constant then increasing the "Avg. Session Length" by one unit will increase the yearly amount spent by a customer by an estimate of \$26.
# 
# Similarly increasing "Time on App" by one unit increases yearly amount spent by \$38.5 whereas, increasing Time on Website by one unit increases yearly amount spent by only $0.6 approximately.
# 
# According to the data above the largest increase of approximately $61.5  is observed when the "Length of Membership" is increased by one unit.

# Using the predictions above we can develop ways in order to increase yearly amount spent by the customers. We can improve the app experience so that the users spend more of their time on the app or we can also focus on the website and develop it so that it becomes as efficient as the app or we can focus on customer relationship so that people remain members for long periods of time.
