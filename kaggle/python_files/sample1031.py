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

# Suppose S is a list of n bits, that is, n 0s and 1s. How long will it take to sort S with the merge-sort
# algorithm? What about quick-sort?

# In[ ]:


Merge-sort takes O(n log n) time, as it is oblivious to the special case of only two possible values.
On the other hand, choosing 1 as the pivot, the quick-sort algorithm takes only one iteration to partition
the input into two segments, one containing all zeros and the other all ones. At this point the list is sorted
in only O(n) time.


# 2. Suppose we are given two n-element sorted sequences A and B that may contain duplicate entries.
# Describe an O(n)-time method for computing a sequence representing all elements in A or B with no
# duplicates.
# 

# In[ ]:


Apply the merge method on A and B to create C. This takes O(n) time. Now, perform a linear scan
through C removing all duplicate elements (i.e. if the next element is equal to the current element,
remove it). So, the total running time is O(n).

# 3. Given an array A of n entries with keys equal to 0 or 1, describe an in-place method for ordering A
# so that all 0s are before every 1.

# Similarly to question 1, select 1 as the pivot and perform quick-sort. After the first iteration, the sequence
# is sorted. This takes O(n) time.

# 4. Suppose we are given an n-element sequence S such that each element in S represents a different
# vote for president, where each vote is given as an integer representing a particular candidate.
# Design an O(n lg n)-time algorithm to see who wins the election S represents, assuming the
# candidate with the most vote wins - even if there are O(n) candidates.

# In[ ]:


First sort the sequence S by the candidate's ID - this takes O(n lg n) time.
Then walk through the sorted sequence, storing the current max count and the count of the current
candidate ID as you go. When you move on to a new ID, check it against the current max and replace the
max if necessary - this takes O(n) time.
Therefore, the total running time is O(n lg n).

# 5. Consider the voting problem from above, but now suppose the number of candidates are a small
# constant number. Describe an O(n)-time algorithm for determining who wins the election.

# In this case, the input data has only a small constant number of different values. Let k be the
# number of candidates. So, we can perform the Bucket-Sort algorithm.
# Create an array A of size k, and initialise all values to 0. Create a table, by assigning every candidate a
# unique integer from 0 to k - 1. These two steps take O(1) time.
# Now, walk through the unsorted sequence S, and for every visited ID, add one to the content of A[i],
# where i is the corresponding number in the look-up table. This process takes O(n) time.
# Thus, the total running time is O(n).
