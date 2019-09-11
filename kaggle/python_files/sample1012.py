#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

# Question 1:
# 30 → 30  →   30 →  30
#          
#             40     24     40 24      40
#                                  ↓
#                                  ↓        58
# 
# 
#             30          ←    30         ←      30
#         24      40      24       40        24       40
#         
#              26      58      26       58                 58
#              
#         25      48               48                  48

# Question 2:
# 1               1          2          3        3
# 2                 3     1    3    1           2
# 3               2                     2      1

#                           62
#                    50              78
#                           54
#                 44       
#                 
#                 17        48   52

#                           4 8 12
#                           
#                           
#                           
# 123             567                91011            131415

# 5           5 16              5 16 22
# 
# insertion of key value 45-splits into two nodes
# 
#                 22         
#          
#          5 16       45    
#          
# insertiom of key value 2-no overflow
#                  
#                  22
#                  
#          2 5 16         45
#          
# insertion of key 10- overflow
# 
#                         10 22
#         2 5             16             45
#         
# insertion of key values 18,30,50,12,1- no overflow
#          
#                   10 22 
#                   
#        1 2 5      12 16 18      30 45 50
#          
#          

# 10
# 
# insertion of other entries implying splay
# 10           insertion of 16           16
#              and splay
#    16        →                     10
#    
#    
#     16
#                 insertion of 12          12
# 10              and splay        
#               →                      10         16
#      12
#    
#    
#      12                                   14
#                 insertion of 14   
#  10       16    and splay              12       16
#                →
#       14                            10
#  
#       14                                   13
#                 insertion of 13  
#  12        16   and splay              12        14
#                 →           
#  10    13                          10               16
#  
#  
#          13                                   15
#                   insertion of 15     
#      12      14   and splay                13      16
#                  →              
#  10               16                     12     14
#                
#               15                       10
