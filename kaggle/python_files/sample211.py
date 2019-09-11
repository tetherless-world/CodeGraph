#!/usr/bin/env python
# coding: utf-8

# # Data exploration about the recent history of the Olympic Games
# 
# Hey, thanks for viewing my Kernel!
# 
# **If you like my work, please, leave an upvote: it will be really appreciated and it will motivate me in offering more content to the Kaggle community ! :)**
# 
# Today, we will explore a dataset on the modern Olympic Games, including all the Games from Athens 1896 to Rio 2016. 
# 
# The data have been scraped from www.sports-reference.com in May 2018.
# 
# **Content**
# 
# The file athlete_events.csv contains 271116 rows and 15 columns; Each row corresponds to an individual athlete competing in an individual Olympic event (athlete-events). 
# The columns are the following:
# 
# 1. ID - Unique number for each athlete;
# 2. Name - Athlete's name;
# 3. Sex - M or F;
# 4. Age - Integer;
# 5. Height - In centimeters;
# 6. Weight - In kilograms;
# 7. Team - Team name;
# 8. NOC - National Olympic Committee 3-letter code;
# 9. Games - Year and season;
# 10. Year - Integer;
# 11. Season - Summer or Winter;
# 12. City - Host city;
# 13. Sport - Sport;
# 14. Event - Event;
# 15. Medal - Gold, Silver, Bronze, or NA.
# 
# ![Picture by Time.com](https://timedotcom.files.wordpress.com/2018/02/180209-olympic-medal-worth.jpg?quality=85)

# # Changelog 
# 
# * 24/08/2018 - New section added: what is the median height/weight of an Olympic medalist? 
# 
# * 25/08/2018 - Inserted a new section "Evolution of the Olympics over time" thanks to [the great suggestion Rodolfo Mendes gave me in my question in the Q&A forum](https://www.kaggle.com/questions-and-answers/63823#375542).
# 
# * 26/08/2018 - Added the sections 'Variation of age and weight along time' with 4 new graphs (boxplot and pointplot).
# 
# * 27/08/2018 - Added the section 'Variation of height along time' with 2 new pointplots, added a short analysis of age over time for Italian athletes.
# 
# * 28/08/2018 - Added a new section about change in height and weight for Gymnasts over time.
# 
# * 29/08/2018 - Added a new section about change in height and weight for Lifters over time, added index of content at the beginning of the kernel.

# # Index of content
# 
# 1. Importing the modules.
# 2. Data importing.
# 3. Collecting information about the two dataset.
# 4. Joining the dataframes.
# 5. Distribution of the age of gold medalists.
# 6. Women in Athletics.
# 7. Medals per country.
# 8. Disciplines with the greatest number of Gold Medals.
# 9. What is the median height/weight of an Olympic medalist?
# 10. Evolution of the Olympics over time.
# 
#     10.1 Variation of male/female athletes over time (Summer Games).
# 
#     10.2 Variation of age along time.
# 
#     10.3 Variation of weight along time.
# 
#     10.4 Variation of height along time.
# 
#     10.5 Variation of age for Italian athletes.
# 
#     10.6 Variation of height/weight along time for particular disciplines.
# 
#     10.6.1 Gymnastic.
# 
#     10.6.2 Weightlifting.
# 
# 11. Conclusions.

# # 1. Importing the modules 

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

import os
print(os.listdir("../input"))

# # 2. Data Importing

# In[ ]:


data = pd.read_csv('../input/athlete_events.csv')
regions = pd.read_csv('../input/noc_regions.csv')

# # 3. Collecting information about the two dataset

# We are going to:
# 
# 1. Review the first lines of the data;
# 2. Use the describe and info functions to collect statistical information, datatypes, column names and other information.

# In[ ]:


data.head(5)

# In[ ]:


data.describe()

# In[ ]:


data.info()

# In[ ]:


regions.head(5)

# # 4. Joining the dataframes

# We can now join the two dataframes using as key the NOC column with the Pandas 'Merge' function ([see documentation](https://pandas.pydata.org/pandas-docs/stable/merging.html))

# In[ ]:


merged = pd.merge(data, regions, on='NOC', how='left')

# Let's see the result:

# In[ ]:


merged.head()

# # 5. Distribution of the age of gold medalists

# Let's start creating a new dataframe including only gold medalists.

# In[ ]:


goldMedals = merged[(merged.Medal == 'Gold')]
goldMedals.head()

# I would like to have a plot of the Age to see the distribution but I need to check first if the Age column contains NaN values..

# In[ ]:


goldMedals.isnull().any()

# ..and it does.
# 
# Let's take only the values that are different from NaN.

# In[ ]:


goldMedals = goldMedals[np.isfinite(goldMedals['Age'])]

# We can now create a countplot to see the result of our work:

# In[ ]:


plt.figure(figsize=(20, 10))
plt.tight_layout()
sns.countplot(goldMedals['Age'])
plt.title('Distribution of Gold Medals')

# It seems that we have people with Age greater that 50 with a gold medal: Let's know more about those people!

# In[ ]:


goldMedals['ID'][goldMedals['Age'] > 50].count()

# 65 people: Great! 
# But which disciplines allows you to land a gold medal after your fifties?
# 
# We will now create a new dataframe called masterDisciplines in which we will insert this new set of people and then create a visualization with it.

# In[ ]:


masterDisciplines = goldMedals['Sport'][goldMedals['Age'] > 50]

# In[ ]:


plt.figure(figsize=(20, 10))
plt.tight_layout()
sns.countplot(masterDisciplines)
plt.title('Gold Medals for Athletes Over 50')

# It seems that our senior gold medalists are shooters, archers, sailors and, above all, horse riders!
# 
# It makes sense: I cannot imagine a sprinter making 100 meters in 10 seconds at 55, but who knows!

# # 6. Women in Athletics

# Studying the data we can try to understand how much medals we have only for women in the recent history of the Summer Games.
# 
# ![Credits to the Daily Mail for the picture](https://i.dailymail.co.uk/i/pix/2012/08/13/article-2187749-147C70BD000005DC-253_964x608.jpg)

# Let's create a filtered datased:

# In[ ]:


womenInOlympics = merged[(merged.Sex == 'F') & (merged.Season == 'Summer')]

# Done. Let's now review our work:

# In[ ]:


womenInOlympics.head(10)

# To plot the curve over time, let's create a plot in which we put the year (on the x-axis) and count of the number of medals per edition of the games (consider that we will have more medals for the same athlete).

# In[ ]:


sns.set(style="darkgrid")
plt.figure(figsize=(20, 10))
sns.countplot(x='Year', data=womenInOlympics)
plt.title('Women medals per edition of the Games')

# Usually I cross-check the data: below I tried to review only the medalists for the 1900 Summer edition to see if the visualization is correct. 

# In[ ]:


womenInOlympics.loc[womenInOlympics['Year'] == 1900].head(10)

# Okay, let's count the rows (same code as above adding the count() function and filtering only for ID).

# In[ ]:


womenInOlympics['ID'].loc[womenInOlympics['Year'] == 1900].count()

# So we have 33 records (with repetitions, for example 'Marion Jones (-Farquhar)' won a medal both for Tennis Women's Singles and Tennis Mixed Doubles - To be sure I cross-checked also with [Wikipedia](https://en.wikipedia.org/wiki/Marion_Jones_Farquhar) and the outcome seems correct).

# # 7. Medals per country

# Let's now review the top 5 gold medal countries:

# In[ ]:


goldMedals.region.value_counts().reset_index(name='Medal').head(5)

# Let's plot this:

# In[ ]:


totalGoldMedals = goldMedals.region.value_counts().reset_index(name='Medal').head(5)
g = sns.catplot(x="index", y="Medal", data=totalGoldMedals,
                height=6, kind="bar", palette="muted")
g.despine(left=True)
g.set_xlabels("Top 5 countries")
g.set_ylabels("Number of Medals")
plt.title('Medals per Country')

# The USA seems to be the most winning country.
# 
# But which are the most awarded disciplines of American Athletes?

# # 8. Disciplines with the greatest number of Gold Medals

# Let's create a dataframe to filter the gold medals only for the USA.

# In[ ]:


goldMedalsUSA = goldMedals.loc[goldMedals['NOC'] == 'USA']

# Done! Now, we can count the medals per discipline:

# In[ ]:


goldMedalsUSA.Event.value_counts().reset_index(name='Medal').head(20)

# And, of course, basketball is the leading discipline!
# 
# Maybe a part of the success is due to the people below.. maybe.
# 
# ![](http://www.elsitodesandro.com/wp-content/uploads/2016/11/dreamteam-1030x781.jpg)

# But hey, wait a minute: We are reviewing a list of athletes, but maybe we are counting the medal of each member of the team instead of counting the medals per team.
# 
# Let's slice the dataframe using only the data of male athletes to better review it:

# In[ ]:


basketballGoldUSA = goldMedalsUSA.loc[(goldMedalsUSA['Sport'] == 'Basketball') & (goldMedalsUSA['Sex'] == 'M')].sort_values(['Year'])

# In[ ]:


basketballGoldUSA.head(15)

# What we supposed is true: the medals are not grouped by Edition/Team but we were counting the gold medals of each member of the team!
# 
# Let's proceed grouping by year the athletes - the idea is to create a new dataframe to make a pre-filter using only the first record for each member of the team.

# In[ ]:


groupedBasketUSA = basketballGoldUSA.groupby(['Year']).first()
groupedBasketUSA

# Let's count the records obtained:

# In[ ]:


groupedBasketUSA['ID'].count()

# And so we have 15 records - cross-checking with the [related Wikipedia page](https://en.wikipedia.org/wiki/United_States_men%27s_national_basketball_team) it seems that our filtering operation has obtained the desired result!

# # 9. What is the median height/weight of an Olympic medalist? 

# Let's try to plot a scatterplot of height vs weight to see the distribution of values (without grouping by discipline).
# 
# First of all, we have to take again the goldMedals dataframe

# In[ ]:


goldMedals.head()

# We can see that we have NaN values both in height and weight columns.
# 
# At this point, we can act as follows:
# 
# 1. Using only the rows that has a value in the Height and Weight columns;
# 2. Replace the value with the mean of the column.
# 
# Solution 2 in my opinion it is not the best way to go: we are talking about data of athletes of different ages and different disciplines (that have done different training).
# 
# Let's go with solution 1.
# 
# The first thing to do is to collect general information about the dataframe that we have to use: goldMedals.

# In[ ]:


goldMedals.info()

# Okay, we have more than 13.000 rows.
# 
# We will now create a dataframe filtering only the rows that has the column Height and Weight populated.

# In[ ]:


notNullMedals = goldMedals[(goldMedals['Height'].notnull()) & (goldMedals['Weight'].notnull())]

# Okay, let's see the first rows of the dataset and the new information with the info function.

# In[ ]:


notNullMedals.head()

# In[ ]:


notNullMedals.info()

# Okay, we have 10.000 rows now, let's create the scatterplot:

# In[ ]:


plt.figure(figsize=(12, 10))
ax = sns.scatterplot(x="Height", y="Weight", data=notNullMedals)
plt.title('Height vs Weight of Olympic Medalists')

# The vast majority of the samples show a linear relation between height and weight (the more the weight, the more the height).
# 
# We have exceptions and I am willing to know more!
# 
# For example, let's see which is the athlete that weighs more than 160 kilograms

# In[ ]:


notNullMedals.loc[notNullMedals['Weight'] > 160]

# Weighlifters: that makes sense :)

# # 10. Evolution of the Olympics over time

# A great thank you to [Rodolfo Mendes](https://www.kaggle.com/rodolfomendes) for giving me the idea for this paragraph.
# 
# We will now try to answer the following questions:
# 
# * How the number of athletes/countries varied along time ?
# * How the proportion of Men/Women varied with time ?
# * How about mean age, weight and height along time ?

# *** 10.1 Variation of male/female athletes over time (Summer Games) ***

# We will now create two dataframes dividing the population of our dataset using Sex and Season (we would like to review only the summer games)

# In[ ]:


MenOverTime = merged[(merged.Sex == 'M') & (merged.Season == 'Summer')]
WomenOverTime = merged[(merged.Sex == 'F') & (merged.Season == 'Summer')]

# Done, let's check the head of one of the new dataframes to see the result:

# In[ ]:


MenOverTime.head()

# Okay, at this time we are ready to create the plots.
# 
# The first one is for men, the second for women:

# In[ ]:


part = MenOverTime.groupby('Year')['Sex'].value_counts()
plt.figure(figsize=(20, 10))
part.loc[:,'M'].plot()
plt.title('Variation of Male Athletes over time')

# In[ ]:


part = WomenOverTime.groupby('Year')['Sex'].value_counts()
plt.figure(figsize=(20, 10))
part.loc[:,'F'].plot()
plt.title('Variation of Female Athletes over time')

# What I immediately saw is that for women:
# 
# 1. We have a steep increase in the population;
# 2. The grow is constant.
# 
# On the other hand, the grow for men seems less strong:
# 
# 1. After the 1990 we can see a relevant decrease in the number of male athletes at the summer games;
# 2. The growth has slowly restarted recently.

# *** 10.2 Variation of age along time ***

# Another really interesting question can be: *"How the age of the athletes has changed over time?"*
# 
# Let's use a [box plot](https://en.wikipedia.org/wiki/Box_plot): In descriptive statistics, a box plot or boxplot is a method for graphically depicting groups of numerical data through their quartiles. 
# 
# Box plots may also have lines extending vertically from the boxes (whiskers) indicating variability outside the upper and lower quartiles, hence the terms box-and-whisker plot and box-and-whisker diagram. 
# 
# Outliers may be plotted as individual points. Box plots are non-parametric: they display variation in samples of a statistical population without making any assumptions of the underlying statistical distribution. 
# 
# The spacings between the different parts of the box indicate the degree of dispersion (spread) and skewness in the data, and show outliers. 
# 
# In addition to the points themselves, they allow one to visually estimate various L-estimators, notably the interquartile range, midhinge, range, mid-range, and trimean. 
# 
# Box plots can be drawn either horizontally or vertically. Box plots received their name from the box in the middle. 

# In[ ]:


plt.figure(figsize=(20, 10))
sns.boxplot('Year', 'Age', data=MenOverTime)
plt.title('Variation of Age for Male Athletes over time')

# What is strange for me is the age of some athletes in the games between the 1924 and the 1948: let's check all the people with age greater than 80.

# In[ ]:


MenOverTime.loc[MenOverTime['Age'] > 80].head(10)

# To be honest, I did not know that the Olympics included Art Competitions!
# 
# After a brief research, [I discovered more](https://en.wikipedia.org/wiki/Art_competitions_at_the_Summer_Olympics): Art competitions formed part of the modern Olympic Games during its early years, from 1912 to 1948. The competitions were part of the original intention of the Olympic Movement's founder, Pierre de Frédy, Baron de Coubertin. Medals were awarded for works of art inspired by sport, divided into five categories: architecture, literature, music, painting, and sculpture. 

# Okay, after this brief parenthesis we can do the same graph for women:

# In[ ]:


plt.figure(figsize=(20, 10))
sns.boxplot('Year', 'Age', data=WomenOverTime)
plt.title('Variation of Age for Female Athletes over time')

# Interesting points for me:
# 
# * Generally, the age distribution starts has a lower minimum and a lower maximum;
# * In 1904 the age distribution is strongly different from the other Olympics: let's know more about this point:

# In[ ]:


WomenOverTime.loc[WomenOverTime['Year'] == 1904]

# ***10.3 Variation of weight along time ***

# We will now try using a pointplot to visualize the variation in weight over athletes.
# 
# The first graph will show data for men, the second for women:

# In[ ]:


plt.figure(figsize=(20, 10))
sns.pointplot('Year', 'Weight', data=MenOverTime)
plt.title('Variation of Weight for Male Athletes over time')

# In[ ]:


plt.figure(figsize=(20, 10))
sns.pointplot('Year', 'Weight', data=WomenOverTime)
plt.title('Variation of Weight for Female Athletes over time')

# What we can see is that it seems that we do not have data for women before 1924.
# 
# Let's try filtering all the women athletes for that period to review this point:

# In[ ]:


womenInOlympics.loc[womenInOlympics['Year'] < 1924].head(20)

# Okay, the first values seems all NaN (Not a number) so the information is correct.

# ***10.4 Variation of height along time***

# Using the same pointplot (with a different palette) we can plot the weight change along time.
# 
# The first graph will show the information for men, the second for women:

# In[ ]:


plt.figure(figsize=(20, 10))
sns.pointplot('Year', 'Height', data=MenOverTime, palette='Set2')
plt.title('Variation of Height for Male Athletes over time')

# In[ ]:


plt.figure(figsize=(20, 10))
sns.pointplot('Year', 'Height', data=WomenOverTime, palette='Set2')
plt.title('Variation of Height for Female Athletes over time')

# What we may see:
# 
# * For both men and women, the height is incrementing over time but it is decreasing between the 2012 and the 2016.
# * For women we have a peak between 1928 and 1948, let's deepen this point:

# In[ ]:


WomenOverTime.loc[(WomenOverTime['Year'] > 1924) & (WomenOverTime['Year'] < 1952)].head(10)

# The list is full of NaN values (that is why the data for the period deviates from what expected).

# ***10.5 Variation of age for Italian athletes *** 

# ![](data:image/jpeg;base64,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)

# Let's see the age over time for Italian athletes.
# 
# I will start reviewing the dataset MenOverTime to refresh the columns:

# In[ ]:


MenOverTime.head(5)

# Let's create a sliced dataframe including only male athletes from Italy

# In[ ]:


itMenOverTime = MenOverTime.loc[MenOverTime['region'] == 'Italy']

# Let's review the first rows:

# In[ ]:


itMenOverTime.head(5)

# Okay, now we can plot the change over time:

# In[ ]:


sns.set(style="darkgrid")
plt.figure(figsize=(20, 10))
sns.countplot(x='Year', data=itMenOverTime, palette='Set2')
plt.title('Variation of Age for Italian Male Athletes over time')

# Okay, we can quickly do the same operation for women:

# In[ ]:


itWomenOverTime = WomenOverTime.loc[WomenOverTime['region'] == 'Italy']

# In[ ]:


sns.set(style="darkgrid")
plt.figure(figsize=(20, 10))
sns.countplot(x='Year', data=itWomenOverTime, palette='Set2')
plt.title('Variation of Age for Italian Female Athletes over time')

# What we see is that the Italian women participation is increasing, while the men participation is decreasing starting from the 2008 games.

# ***10.6 Variation of height/weight along time for particular disciplines***

# **10.6.1 Gymnastic**

# Let's see the trend of height/weight for Gymnasts, starting from men and then women following the usual approach:

# In[ ]:


MenOverTime.head(5)

# Let's first of all isolate all the discipline of the Olympics dataframe.
# 
# My idea is to see if Gymnastics is called differently or if there is any typo.

# In[ ]:


MenOverTime['Sport'].unique().tolist()

# Okay, the string to use to filter is 'Gymnastics': let's create two new dataframes for men and women.

# In[ ]:


gymMenOverTime = MenOverTime.loc[MenOverTime['Sport'] == 'Gymnastics']
gymWomenOverTime = WomenOverTime.loc[WomenOverTime['Sport'] == 'Gymnastics']

# Okay: let's now create our plot for male and female athletes and then we can make our observations

# In[ ]:


plt.figure(figsize=(20, 10))
sns.barplot('Year', 'Weight', data=gymMenOverTime)
plt.title('Weight over year for Male Gymnasts')

# In[ ]:


plt.figure(figsize=(20, 10))
sns.barplot('Year', 'Height', data=gymMenOverTime)
plt.title('Height over year for Male Gymnasts')

# In[ ]:


plt.figure(figsize=(20, 10))
sns.barplot('Year', 'Weight', data=gymWomenOverTime)
plt.title('Weight over year for Female Gymnasts')

# In[ ]:


plt.figure(figsize=(20, 10))
sns.barplot('Year', 'Height', data=gymWomenOverTime)
plt.title('Height over year for Female Gymnasts')

# A few things I noticed:
# * The weight for female Gymnasts has go down for 60 to 50 kilograms on average;
# * The weight for men has been more or less stable since 1964;
# * The height is more stable for both men and women.
# 
# Also, men weight data from 1924 seems missing: let's check.

# In[ ]:


gymMenOverTime['Weight'].loc[gymMenOverTime['Year'] == 1924].isnull().all()

# It seems that we do not have any information about the athletes in 1924.

# **10.6.2 Weightlifting**

# ![](data:image/jpeg;base64,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)

# Let's work on an analysis similar to what we have done for Gymnastics also for the Lifters.
# 
# We can start creating a new, dedicated dataframe.

# In[ ]:


wlMenOverTime = MenOverTime.loc[MenOverTime['Sport'] == 'Weightlifting']
wlWomenOverTime = WomenOverTime.loc[WomenOverTime['Sport'] == 'Weightlifting']

# Okay: let's now create our plot for male and female athletes and then we can make our observations

# In[ ]:


plt.figure(figsize=(20, 10))
sns.pointplot('Year', 'Weight', data=wlMenOverTime, palette='Set2')
plt.title('Weight over year for Male Lifters')

# In[ ]:


plt.figure(figsize=(20, 10))
sns.pointplot('Year', 'Height', data=wlMenOverTime, palette='Set2')
plt.title('Height over year for Male Lifters')

# In[ ]:


plt.figure(figsize=(20, 10))
sns.pointplot('Year', 'Weight', data=wlWomenOverTime)
plt.title('Weight over year for Female Lifters')

# In[ ]:


plt.figure(figsize=(20, 10))
sns.pointplot('Year', 'Height', data=wlWomenOverTime)
plt.title('Height over year for Female Lifters')

# It seems that we do not have data for female athletes before the 2000 Games.
# 
# Let's check this point.

# In[ ]:


wlWomenOverTime['Weight'].loc[wlWomenOverTime['Year'] < 2000].isnull().all()

# Our observation seems correct.

# # 11. Conclusions

# **First of all, thank you so much for reading! If you liked my work, please, do not forget to leave an upvote: it will be really appreciated and it will motivate me in offering more content to the Kaggle community ! :)**
# 
# I will review and update the kernel periodically following your suggestions or if I want to discover something new (see the changelog at the beginning with the history of the updates).
# 
# If you want to ask something, feel free to comment!
