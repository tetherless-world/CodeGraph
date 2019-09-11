#!/usr/bin/env python
# coding: utf-8

# Thanks for **UPVOTING** this kernel! Trying to become a Kernel Master. 👍
# 
# > Check out other cool projects by Pavlo Fesenko:
# - [📊 Interactive Titanic dashboard using Bokeh](https://www.kaggle.com/pavlofesenko/interactive-titanic-dashboard-using-bokeh)
# - [Extending Titanic dataset using Wikipedia](https://www.kaggle.com/pavlofesenko/extending-titanic-dataset-using-wikipedia)
# - [🥇 Strategies to earn discussion medals](https://www.kaggle.com/pavlofesenko/strategies-to-earn-discussion-medals)
# 
# ---

# # 1. BAM!!! 😎
# 
# 8 lines of code, public LB score = 0.80861 (top 10%)

# In[ ]:


import pandas as pd, catboost
train, test = pd.read_csv('../input/train.csv'), pd.read_csv('../input/test.csv')
train['Boy'], test['Boy'] = [(df.Name.str.split().str[1] == 'Master.').astype('int') for df in [train, test]]
train['Surname'], test['Surname'] = [df.Name.str.split(',').str[0] for df in [train, test]]
model = catboost.CatBoostClassifier(one_hot_max_size=4, iterations=100, random_seed=0, verbose=False)
model.fit(train[['Sex', 'Pclass', 'Embarked', 'Boy', 'Surname']].fillna(''), train['Survived'], cat_features=[0, 2, 4])
pred = model.predict(test[['Sex', 'Pclass', 'Embarked', 'Boy', 'Surname']].fillna('')).astype('int')
pd.concat([test['PassengerId'], pd.DataFrame(pred, columns=['Survived'])], axis=1).to_csv('submission.csv', index=False)

# # 2. Wait, whaaat?! 😲
# 
# Your first reaction might be: "Wait, what was that? Where are 5 extra features, 10 stacked models, 100 hyperparameter sets? Where is even feature encoding???"
# 
# Well, it's true, there are only 2 extra features, only 1 model and only 1 untuned hyperparameter set. And wait, THERE IS feature encoding. 😜
# 
# All feature encoding is built in the gradient boosting library called [Catboost](https://catboost.ai/). This library handles categorical features like a pro, hence the name "Cat-boost". By default Catboost performs one-hot encoding for all categorical features with the number of categorical values less or equal than 2. This number, however, can be adjusted using the parameter `one_hot_max_size`. The rest of the categorical features are encoded using the mean (or target) encoding which is the secret sauce of Catboost. Briefly, this method assigns the mean of the target value (instead of 1 as in one-hot encoding) for each category and thus creates better separation between classes. To understand this concept, I would strongly recommend you to check this awesome [video about mean encoding](https://www.coursera.org/lecture/competitive-data-science/concept-of-mean-encoding-b5Gxv) from Higher School of Economics.
# 
# For my model I generated 2 new features `Boy` and `Surname`. Among all the features the categorical ones are `Sex`, `Embarked` and `Surname`. The features `Sex` and `Embarked` are encoded using one-hot encoding by setting `one_hot_max_size=4` that matches the maximumn number of possible categories, namely in `Embarked` (`'C'`, `'Q'`, `'S'` and an empty string `''` that filled in `NaN` values). The feature `Surname` is encoded using mean encoding. This encoding method assigns a regularized mean survival rate for each surname. Intuitively, this makes sense: if the majority of passengers with the same surname survived, other passengers with the same surname (relatives) would probably survived as well.
# 
# After this intro you might have plenty of questions, for example: "If mean encoding is better than one-hot encoding, why did you apply mean encoding only to `Surname`?", "Why did you choose only these 5 features and ignored the rest?", "Why does your model contain other hyperparameters like `iterations`, `random_seed`, etc.?". The answers for these and other questions are in the next section so if you are curious, keep reading. 😉

# # 3. OK, tell me more 🤩
# 
# Let's go step-by-step and understand how each feature impacts the model.
# 
# The model with the features `Sex`, `Pclass` and `Embarked` is well-known and gives the public LB score of 0.77990 that is better than the baseline model using only `Sex` with the score of 0.76555 (see the section 2.4 [here](https://www.kaggle.com/pliptor/how-am-i-doing-with-my-score)).
# 
# The table below shows the mean survival rate of females/males depending on `Pclass` and `Embarked`. The cells are highilighted in red/green if the survival rate is below/above the threshold of 0.5. Clearly, the females from the 3rd class who embarked in Southampton had lower chances of survival and this additional information increases the model score.

# In[ ]:


def highlight(value):
    if value >= 0.5:
        style = 'background-color: palegreen'
    else:
        style = 'background-color: pink'
    return style

pd.pivot_table(train, values='Survived', index=['Pclass', 'Embarked'], columns='Sex').style.applymap(highlight)

# Following the best practices, let's calculate our own CV score that will be used as a reference. This can be conveniently done using the Catboost method `cv()`. Note that `cv()` requires the data to be passed the Catboost object `Pool()`. If the parameter `plot=True` is set, it will plot the metrics for learning and evaluation datasets versus the number of iterations. This plot is realized using Ipywidgets and at the moment is only displayed in the kernel Edit mode. In order to see the plot, you should fork this kernel and run it in the editing mode.
# 
# For better CV precision, I chose 10 folds which, on the other hand, increases the calculation time. So to speed things up, I limited the number of iterations to 100 in `CatBoostClassifier()` (by default 1000). Note that decreasing/increasing number of iterations might under-/overfit your model if the learning rate is fixed. Catboost, however, automatically calculates the optimized learning rate if it isn't explicitly mentioned. Moreover, the CV plot will show if under-/overfitting occurs.
# 
# There are also several more parameters in `CatBoostClassifier()` that I haven't explained so far. To make results reproducible from run to run, I fixed `random_seed=0`. To avoid printing iterations results in stdout, I set `verbose=False`. Finally, to display accuracy on the CV plot as the main metric (by default logarithmic loss), I specified `eval_metric='Accuracy'`.
# 
# Our CV plot shows that accuracy saturates very quickly and reaches a constant plateau at 0.81145. This means that there is no under-/overfitting at 100 iterations.

# In[ ]:


features = ['Sex', 'Pclass', 'Embarked']

X_train = train[features].fillna('')
y_train = train['Survived']
X_test = test[features].fillna('')

model = catboost.CatBoostClassifier(
    one_hot_max_size=4,
    iterations=100,
    random_seed=0,
    verbose=False,
    eval_metric='Accuracy'
)

pool = catboost.Pool(X_train, y_train, cat_features=[0, 2])
print('To see the Catboost plots, fork this kernel and run it in the editing mode.')
cv_scores = catboost.cv(pool, model.get_params(), fold_count=10, plot=True)
print('CV score: {:.5f}'.format(cv_scores['test-Accuracy-mean'].values[-1]))

# You can check yourself the public LB score of this model (0.77990) by submitting the file `submission2.csv` from the section Output of this kernel.

# In[ ]:


model.fit(pool)
pred = model.predict(X_test).astype('int')
output = pd.concat([test['PassengerId'], pd.DataFrame(pred, columns=['Survived'])], axis=1)
output.to_csv('submission2.csv', index=False)

# Next I consider the generated feature `Boy`. It takes the value 1 if the title in `Name` is "Master" and the value 0 otherwise.
# 
# In other kernels you might find a similar feature called `Minor` (or `Child`) that takes the value 1 if `Age` is less than 14 and `Pclass` isn't 3, and the value 0 otherwise (see the section 2.6 [here](https://www.kaggle.com/pliptor/how-am-i-doing-with-my-score)). There are, however, some problems with this feature. The 3rd class has many missing age values that doesn't allow accurate predictions, hence the additional rule to exclude it. As an alternative to this, imputation of missing values could be used. In the most successful kernels this is done based on titles so we are actually doing almost the same as for the feature `Boy`. This could also be the reason why importance of the generated feature `Title` is quite high in some kernels.
# 
# You might wonder that the feature `Boy` won't take girls into account? Actually, since girls are in the female class, most of them will be predicted to survive anyway so additional feature is not needed.
# 
# And why not simply using `Age` without engineering new features? Again, due to the large number of missing values, the model wouldn't be able to generalize well. To solve this issue, you could use my [Titanic extended dataset](https://www.kaggle.com/pavlofesenko/titanic-extended) where almost all age values are present.
# 
# The table below shows the mean survival rate of boys/not boys depending on `Pclass`. I didn't include `Embarked` because for some categories there are simply no boys. Clearly, boys from the 1st and 2nd class are very likely to survive while boys from the 3rd class aren't. This additional information should improve the model score.

# In[ ]:


pd.pivot_table(train, values='Survived', index='Pclass', columns='Boy').style.applymap(highlight)

# Indeed, upon adding the feature `Boy` the CV score increases to 0.82941 after 100 iterations. Just like on the previous CV plot, the accuracy reaches a constant plateau which indicates no under-/overfitting.

# In[ ]:


features = ['Sex', 'Pclass', 'Embarked', 'Boy']

X_train = train[features].fillna('')
y_train = train['Survived']
X_test = test[features].fillna('')

model = catboost.CatBoostClassifier(
    one_hot_max_size=4,
    iterations=100,
    random_seed=0,
    verbose=False,
    eval_metric='Accuracy'
)

pool = catboost.Pool(X_train, y_train, cat_features=[0, 2])
print('To see the Catboost plots, fork this kernel and run it in the editing mode.')
cv_scores = catboost.cv(pool, model.get_params(), fold_count=10, plot=True)
print('CV score: {:.5f}'.format(cv_scores['test-Accuracy-mean'].values[-1]))

# You can check yourself the public LB score of this model (0.78947) by submitting the file `submission3.csv` from the section Output of this kernel.

# In[ ]:


model.fit(pool)
pred = model.predict(X_test).astype('int')
output = pd.concat([test['PassengerId'], pd.DataFrame(pred, columns=['Survived'])], axis=1)
output.to_csv('submission3.csv', index=False)

# Next I consider the generated feature `Surname`. In total there are 667 unique surnames in the training dataset. If they are one-hot encoded, additional 667 new features will be created. Finding patterns in all these features would be impossible using gradient boosting decision trees that typically have small depth (by default 6 in Catboost).
# 
# Now imagine that instead of 667 features, we create only 1 feature that will correspond to the mean survival rate of each surname. This will allow clear sepearation between the surnames with higher and lower chances for survival (see the example table for the first 15 surnames below). This task is much more manageable for gradient boosting trees. This is the main principle behind mean (or target) encoding.
# 
# The major issue with mean encoding is that you most likely overfit your model by blending a categorical feature with the target. To avoid it, regularization is an absolute must. Catboost performs this regularization under the hood using the method called expanding mean. You can learn more about regularization methods for mean encoding [here](https://www.coursera.org/lecture/competitive-data-science/regularization-LGYQ2). Still, the regularization quality in Catboost can be irregular, therefore, I prefer to use mean encoding only for categorical features with high cardinality (like `Surname`) and one-hot encoding for the rest (like `Sex` and `Embarked`).

# In[ ]:


pd.pivot_table(train, values='Survived', index='Surname')[:15].sort_values('Survived').style.applymap(highlight)

# Upon adding the feature `Surname` the CV score again increases to 0.85185 after 100 iterations. This time the CV plot doesn't have a constant plateau but it is still quite stable to assume no major under-/overfitting.

# In[ ]:


features = ['Sex', 'Pclass', 'Embarked', 'Boy', 'Surname']

X_train = train[features].fillna('')
y_train = train['Survived']
X_test = test[features].fillna('')

model = catboost.CatBoostClassifier(
    one_hot_max_size=4,
    iterations=100,
    random_seed=0,
    verbose=False,
    eval_metric='Accuracy'
)

pool = catboost.Pool(X_train, y_train, cat_features=[0, 2, 4])
print('To see the Catboost plots, fork this kernel and run it in the editing mode.')
cv_scores = catboost.cv(pool, model.get_params(), fold_count=10, plot=True)
print('CV score: {:.5f}'.format(cv_scores['test-Accuracy-mean'].values[-1]))

# If you submit the results of this model `submission4.csv`, you should get the same public LB score (0.80861) as using the original 8-line model.

# In[ ]:


model.fit(pool)
pred = model.predict(X_test).astype('int')
output = pd.concat([test['PassengerId'], pd.DataFrame(pred, columns=['Survived'])], axis=1)
output.to_csv('submission4.csv', index=False)

# # 4. Sky is the limit 🚀
# 
# In my model I used only 5 features due to the fact that gradient boosting trees with small depth work better with fewer features. But I haven't used any hyperparameter optimization so there is definitely room for improvement. As I mentioned earlier, Catboost automatically selects its learning rate depending on the number of iterations. This learning rate, however, is not always optimal so manual adjustment of the hyperparameter pair learning rate/number of iterations should result in even better score. The CV plots are really helpful to find these optimum values. Feel free to post your maximum score for this model in the comments below.

# # 5. Credits 🤘
# 
# - [Concept of mean encoding](https://www.coursera.org/lecture/competitive-data-science/concept-of-mean-encoding-b5Gxv) by Higher School of Economics
# - [How am I doing with my score?](https://www.kaggle.com/pliptor/how-am-i-doing-with-my-score) by Oscar Takeshita
# - [Yandex CatBoost demo](https://www.kaggle.com/pacifik80/yandex-catboost-demo/) by Stanislav Ermolov

# ---
# Thanks for **UPVOTING** this kernel! Trying to become a Kernel Master. 👍
# 
# > Check out other cool projects by Pavlo Fesenko:
# - [📊 Interactive Titanic dashboard using Bokeh](https://www.kaggle.com/pavlofesenko/interactive-titanic-dashboard-using-bokeh)
# - [Extending Titanic dataset using Wikipedia](https://www.kaggle.com/pavlofesenko/extending-titanic-dataset-using-wikipedia)
# - [🥇 Strategies to earn discussion medals](https://www.kaggle.com/pavlofesenko/strategies-to-earn-discussion-medals)
