#!/usr/bin/env python
# coding: utf-8

# *Note (April 16): This notebook shows that the public test set has a lower mean label than the training set, but it does not say why, because I did not know. Ash Hafez has probably now found the reason why in the kernel [Temporal pattern in train response rates?][1]*
# 
# 
#   [1]: https://www.kaggle.com/ashhafez/quora-question-pairs/temporal-pattern-in-train-response-rates

# *Note 2 (April 23): When I first wrote this, I only used 2 decimal places in the calculation. As noted in the [comments below][1] and in [another kernel][2], the correct percentage is more like 0.175. I have corrected this below. I have also typeset the single equation because the number of parenthesis was causing confusion.*
# 
# 
#   [1]: https://www.kaggle.com/davidthaler/quora-question-pairs/how-many-1-s-are-in-the-public-lb#177070
#   [2]: https://www.kaggle.com/badat0202/quora-question-pairs/estimate-distribution-of-data-in-lb

# In the kernel [Exploratory Data Analysis][1], Anokas noted that there are a different number of positives in the training data than on the public test set. He also reported getting a score of 0.554 on the LB with a constant prediction at the training set mean of 0.369. That is enough information to determine the fraction of 1's in the public LB. So let's do that.
# 
#   [1]: https://www.kaggle.com/anokas/quora-question-pairs/exploratory-data-analysis

# In[ ]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import log_loss

# In[ ]:


l = []
p = [0.37] * 1000
for r in range(1, 1000):
    y = [1]*r + [0]*(1000-r)
    l.append(log_loss(y, p))
l = np.array(l)
x = np.arange(0.1, 100, 0.1)

# In[ ]:


plt.plot(x, l, '_')
plt.title('Log Loss vs. Pct. Positve with Constant Prediction of 0.37')
plt.xlabel('% Positve in LB')
plt.ylabel('Log Loss for Constant Prediction 0.37')
plt.grid()
plt.show()

# From the graph, the log loss of 0.55 that Anokas got looks like it occurs at around 0.165 or 0.17. In fact, we can compute it directly. Using the log loss formula from [here][1], and using the fact that this is a constant prediction, we get:    
# 
# $$r = \frac{logloss + log(1-p)}{log\big( \frac{1-p}{p}\big)}$$
# 
# where r is the fraction of positives. In that expression, p and the logloss are known for Anokas' constant prediction of p=0.369, which gave loss of 0.554. That yields r of 0.174, about the same as the graph.
# 
#   [1]: https://www.kaggle.com/wiki/LogarithmicLoss

# In[ ]:


test = pd.read_csv('../input/test.csv')
sub = test[['test_id']].copy()
sub['is_duplicate'] = 0.174
sub.to_csv('constant_sub.csv', index=False)

# So that get's about 0.463 on the LB. Now the bigger question is: How many 1's are in the private LB? Is it just a fluke that the public LB has so many fewer 1's than train?
