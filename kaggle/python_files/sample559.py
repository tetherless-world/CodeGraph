#!/usr/bin/env python
# coding: utf-8

# # Welcome to data visualization
# 
# Welcome to the Data Visualization tutorial!
# 
# Data visualization is one of the core skills in data science. In order to start building useful models, we need to understand the underlying dataset. You will never be an expert on the data you are working with, and will always need to explore the variables in great depth before you can move on to building a model or doing something else with the data. Effective data visualization is the most important tool in your arsenal for getting this done, and hence an critical skill for you to master.
# 
# In this tutorial series, we will cover building effective data visualizations in Python. We will cover the `pandas` and `seaborn` plotting tools in depth. We will also touch upon `matplotlib`, and discuss `plotly` and `plotnine` in later (optional) sections. We still start with simple single-variable plots and work our way up to plots showing two, three, or even more dimensions. Upon completing this tutorial you should be well-equipped to start doing useful exploratory data analysis (EDA) with your own datasets!
# 
# ## Prerequisites
# 
# There's one thing to do before we get started, however: learn about `pandas`, the linchpin of the Python data science ecosystem. `pandas` contains all of the data reading, writing, and manipulation tools that you will need to probe your data, run your models, and of course, visualize. As a result, a working understanding of `pandas` is critical to being a sucessful builder of data visualizations.
# 
# If you are totally unfamiliar with `pandas`, the following prerequisites will cover just enough `pandas` to get you started. These tutorial sections are targetted at total beginners; if you are already at least mildly familiar with the library, you should skip ahead to the next section.
# 
# <table style="width:800px;">
# <tr>
# <td colspan=2 style="padding:25px; text-align:center; font-size:18px; width:33%"><a href="https://www.kaggle.com/sohier/tutorial-accessing-data-with-pandas/">Accessing Data with Pandas</a></td>
# <td colspan=2 style="padding:25px; text-align:center; font-size:18px; width:33%"><a href="https://www.kaggle.com/dansbecker/selecting-and-filtering-in-pandas">Selecting and Filtering in Pandas</a></td>
# <!--
# <td colspan=2 style="padding:25px; text-align:center; font-size:18px; width:33%"><a href="https://www.kaggle.com/residentmario/just-enough-pandas-optional/">Prerequisite 3</a></td>
# -->
# </tr>
# </table>
# 
# ## Contents
# 
# This tutorial consists of the following sections:
# 
# <table style="width:800px">
# <tr>
# <td colspan=2 style="padding:25px; text-align:center; font-size:18px;"><a href="https://www.kaggle.com/residentmario/univariate-plotting-with-pandas">Univariate plotting with pandas</a></td>
# </tr>
# <tr>
# <td colspan=2 style="padding:25px; text-align:center; font-size:18px;"><a href="https://www.kaggle.com/residentmario/bivariate-plotting-with-pandas">Bivariate plotting with pandas</a></td>
# </tr>
# <tr>
# <td style="padding:25px; text-align:center; font-size:18px; width:50%"><a href="https://www.kaggle.com/residentmario/styling-your-plots/">Styling your plots</a>
# </td>
# <td style="padding:25px; text-align:center; font-size:18px; width:50%"><a href="https://www.kaggle.com/residentmario/plotting-with-seaborn">Plotting with seaborn</a>
# </td>
# </tr>
# <tr>
# <td style="padding:25px; text-align:center; font-size:18px; width:50%"><a href="https://www.kaggle.com/residentmario/subplots/">Subplots</a></td>
# <td style="padding:25px; text-align:center; font-size:18px;width:50%"><a href="https://www.kaggle.com/residentmario/faceting-with-seaborn/">Faceting with seaborn</a></td>
# </tr>
# <tr>
# <td colspan=2 style="padding:25px; text-align:center; font-size:18px;"><a href="https://www.kaggle.com/residentmario/multivariate-plotting">Multivariate plotting</a></td>
# </tr>
# <tr>
# <td colspan=2 style="padding:25px; text-align:center; font-size:18px;"><a href="https://www.kaggle.com/residentmario/introduction-to-plotly-optional/">Plotting with plotly</a></td>
# </tr>
# <tr>
# <td colspan=2 style="padding:25px; text-align:center; font-size:18px;"><a href="https://www.kaggle.com/residentmario/grammer-of-graphics-with-plotnine-optional/">Grammar of graphics with plotnine</a></td>
# </tr>
# </table>
# 
# .
# 
# Each section will focus on one particular aspect of plotting in Python, relying on the knowledge you have aquired up until that point to get the job done.
# 
# Ready? [To start the tutorial, proceed to the next section, "Univariate plotting with pandas"](https://www.kaggle.com/residentmario/univariate-plotting-with-pandas/).

# In[ ]:



