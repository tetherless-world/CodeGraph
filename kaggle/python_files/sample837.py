#!/usr/bin/env python
# coding: utf-8

# # Modified Naive Bayes scores 0.899 LB - Santander
# In this kernel we demonstrate that unconstrained Naive Bayes can score 0.899 LB. I call it "unconstrained" because it doesn't assume that each variable has a Gaussian distribution like typical Naive Bayes. Instead we allow for arbitrary distributions and we plot these distributions below. I called it "modified" because we don't reverse the conditional probabilities.
# 
# This kernel is useful because (1) it shows that an accurate score can be achieved using a simple model that assumes the variables are independent. And (2) this kernel displays interesting EDA which provides insights about the data.
#   
# # Load Data

# In[ ]:


import numpy as np, pandas as pd
train = pd.read_csv('../input/train.csv')
train0 = train[ train['target']==0 ].copy()
train1 = train[ train['target']==1 ].copy()
train.sample(5)

# # Statistical Functions
# Below are functions to calcuate various statistical things.

# In[ ]:


# CALCULATE MEANS AND STANDARD DEVIATIONS
s = [0]*200
m = [0]*200
for i in range(200):
    s[i] = np.std(train['var_'+str(i)])
    m[i] = np.mean(train['var_'+str(i)])
    
# CALCULATE PROB(TARGET=1 | X)
def getp(i,x):
    c = 3 #smoothing factor
    a = len( train1[ (train1['var_'+str(i)]>x-s[i]/c)&(train1['var_'+str(i)]<x+s[i]/c) ] ) 
    b = len( train0[ (train0['var_'+str(i)]>x-s[i]/c)&(train0['var_'+str(i)]<x+s[i]/c) ] )
    if a+b<500: return 0.1 #smoothing factor
    # RETURN PROBABILITY
    return a / (a+b)
    # ALTERNATIVELY RETURN ODDS
    # return a / b
    
# SMOOTH A DISCRETE FUNCTION
def smooth(x,st=1):
    for j in range(st):
        x2 = np.ones(len(x)) * 0.1
        for i in range(len(x)-2):
            x2[i+1] = 0.25*x[i]+0.5*x[i+1]+0.25*x[i+2]
        x = x2.copy()
    return x

# # Display Target Density and Target Probability
# Below are two plots for each of the 200 variables. The first is the density of `target=1` versus `target=0`. The second gives the probability that `target=1` given different values for `var_k`.

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# DRAW PLOTS, YES OR NO
Picture = True
# DATA HAS Z-SCORE RANGE OF -4.5 TO 4.5
rmin=-5; rmax=5; 
# CALCULATE PROBABILITIES FOR 501 BINS
res=501
# STORE PROBABILITIES IN PR
pr = 0.1 * np.ones((200,res))
pr2 = pr.copy()
xr = np.zeros((200,res))
xr2 = xr.copy()
ct2 = 0
for j in range(50):
    if Picture: plt.figure(figsize=(15,8))
    for v in range(4):
        ct = 0
        # CALCULATE PROBABILITY FUNCTION FOR VAR
        for i in np.linspace(rmin,rmax,res):
            pr[v+4*j,ct] = getp(v+4*j,m[v+4*j]+i*s[v+4*j])
            xr[v+4*j,ct] = m[v+4*j]+i*s[v+4*j]
            xr2[v+4*j,ct] = i
            ct += 1
        if Picture:
            # SMOOTH FUNCTION FOR PRETTIER DISPLAY
            # BUT USE UNSMOOTHED FUNCTION FOR PREDICTION
            pr2[v+4*j,:] = smooth(pr[v+4*j,:],res//10)
            # DISPLAY PROBABILITY FUNCTION
            plt.subplot(2, 4, ct2%4+5)
            plt.plot(xr[v+4*j,:],pr2[v+4*j,:],'-')
            plt.title('P( t=1 | var_'+str(v+4*j)+' )')
            xx = plt.xlim()
            # DISPLAY TARGET DENSITIES
            plt.subplot(2, 4, ct2%4+1)            
            sns.distplot(train0['var_'+str(v+4*j)], label = 't=0')
            sns.distplot(train1['var_'+str(v+4*j)], label = 't=1')
            plt.title('var_'+str(v+4*j))
            plt.legend()
            plt.xlim(xx)
            plt.xlabel('')
        if (ct2%8==0): print('Showing vars',ct2,'to',ct2+7,'...')
        ct2 += 1
    if Picture: plt.show()

# # Target Probability Function
# Above, the target probability function was calculated for each variable with resolution equal to `standard deviation / 50` from -5 to 5. For example, we know the `Probability ( target=1 | var=x )` for `z-score = -5.00, -4.98, ..., -0.02, 0, 0.02, ..., 4.98, 5.00` where `z-score = (x - var_mean) / (var_standard_deviation)`. The python function below accesses these pre-calculated values from their numpy array.

# In[ ]:


def getp2(i,x):
    z = (x-m[i])/s[i]
    ss = (rmax-rmin)/(res-1)
    if res%2==0: idx = min( (res+1)//2 + z//ss, res-1)
    else: idx = min( (res+1)//2 + (z-ss/2)//ss, res-1)
    idx = max(idx,0)
    return pr[i,int(idx)]

# # Validation
# We will ignore the training data's target and make our own prediction for each training observation. Then using our predictions and the true value, we will calculate validation AUC. (There is a leak in this validation method but none-the-less it gives an approximation of CV score. If you wish to tune this model, you should use a proper validation set. Current actual 5-fold CV is 0.8995)

# In[ ]:


from sklearn.metrics import roc_auc_score
print('Calculating 200000 predictions and displaying a few examples...')
pred = [0]*200000; ct = 0
for r in train.index:
    p = 0.1
    for i in range(200):
        p *= 10*getp2(i,train.iloc[r,2+i])
    if ct%25000==0: print('train',r,'has target =',train.iloc[r,1],'and prediction =',p)
    pred[ct]=p; ct += 1
print('###############')
print('Validation AUC =',roc_auc_score(train['target'], pred))

# In[ ]:


#https://stackoverflow.com/questions/25009284/how-to-plot-roc-curve-in-python
from sklearn import metrics
fpr, tpr, threshold = metrics.roc_curve(train['target'], pred)
roc_auc = metrics.auc(fpr, tpr)

plt.figure(figsize=(6,6))
plt.title('Validation ROC')
plt.plot(fpr, tpr, 'b', label = 'Val AUC = %0.3f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

# # Predict Test and Submit
# Naive Bayes is a simple model. Given observation with `var_0 = 15`, `var_1 = 5`, `var_2 = 10`, etc. We compute the probability that `target=1` by calculating `P(t=1) * P(t=1 | var_0=15)/P(t=1) * P(t=1 | var_1=5)/P(t=1) * P(t=1 | var_2=10)/P(t=1) * ...` where `P(t=1)=0.1` and the other probabilities are computed above by counting occurences in the training data. So each observation has 200 variables and we simply multiply together the 200 target probabilities given by each variable. (In typical Naive Bayes, you use Bayes formula, reverse the probabilities, and find `P(var_0=15 | t=1)`. This is modified Naive Bayes and more intuitive.)

# In[ ]:


test = pd.read_csv('../input/test.csv')
print('Calculating 200000 predictions and displaying a few examples...')
pred = [0]*200000; ct = 0
for r in test.index:
    p = 0.1
    for i in range(200):
        p *= 10*getp2(i,test.iloc[r,1+i])
    if ct%25000==0: print('test',r,'has prediction =',p)
    pred[ct]=p
    ct += 1
sub = pd.read_csv('../input/sample_submission.csv')
sub['target'] = pred
sub.to_csv('submission.csv',index=False)
print('###############')
print('Finished. Wrote predictions to submission.csv')

# # Plot Predictions

# In[ ]:


sub.loc[ sub['target']>1 , 'target'] = 1
b = plt.hist(sub['target'], bins=200)

# # Conclusion
# In conclusion we used modified Naive Bayes to predict Santander Customer transactions. Since we achieved an accurate score of 0.899 LB (which rivals other methods that capture interactions), this demonstrates that there is little or no interaction between the 200 variables. Additionally in this kernel we observed some fascinating EDA which provide insights about the variables. Can this method be improved? Perhaps by tuning this model better (adjust smoothing, resolution, etc) we can increase validation AUC and increase LB AUC but I don't think we can score over 0.902 with this method. There are other secrets hiding in the Santander data.
# ![image](http://playagricola.com/Kaggle/score32319.png)
