#!/usr/bin/env python
# coding: utf-8

# These exercises accompany the tutorial on [functions and getting help](https://www.kaggle.com/colinmorris/functions-and-getting-help-daily).
# 
# As before, don't forget to run the setup code below before jumping into question 1.

# In[ ]:


# SETUP. You don't need to worry for now about what this code does or how it works.
from learntools.core import binder; binder.bind(globals())
from learntools.python.ex2 import *
print('Setup complete.')

# # Exercises

# ## 1.
# 
# Complete the body of the following function according to its docstring.
# 
# HINT: Python has a builtin function `round`

# In[ ]:


def round_to_two_places(num):
    """Return the given number rounded to two decimal places. 
    
    >>> round_to_two_places(3.14159)
    3.14
    """
    # Replace this body with your own code.
    # ("pass" is a keyword that does literally nothing. We used it as a placeholder
    # because after we begin a code block, Python requires at least one line of code)
    pass

q1.check()

# In[ ]:


# Uncomment the following for a hint
#q1.hint()
# Or uncomment the following to peek at the solution
#q1.solution()

# ## 2.
# The help for `round` says that `ndigits` (the second argument) may be negative.
# What do you think will happen when it is? Try some examples in the following cell?
# 
# Can you think of a case where this would be useful?

# In[ ]:


# Put your test code here

# In[ ]:


#q2.solution()

# ## 3.
# 
# In a previous programming problem, the candy-sharing friends Alice, Bob and Carol tried to split candies evenly. For the sake of their friendship, any candies left over would be smashed. For example, if they collectively bring home 91 candies, they'll take 30 each and smash 1.
# 
# Below is a simple function that will calculate the number of candies to smash for *any* number of total candies.
# 
# Modify it so that it optionally takes a second argument representing the number of friends the candies are being split between. If no second argument is provided, it should assume 3 friends, as before.
# 
# Update the docstring to reflect this new behaviour.

# In[ ]:


def to_smash(total_candies):
    """Return the number of leftover candies that must be smashed after distributing
    the given number of candies evenly between 3 friends.
    
    >>> to_smash(91)
    1
    """
    return total_candies % 3

q3.check()

# In[ ]:


#q3.hint()

# In[ ]:


#q3.solution()

# ## 4.
# 
# It may not be fun, but reading and understanding error messages will be an important part of your Python career.
# 
# Each code cell below contains some commented-out buggy code. For each cell...
# 
# 1. Read the code and predict what you think will happen when it's run.
# 2. Then uncomment the code and run it to see what happens. (**Tip**: In the kernel editor, you can highlight several lines and press `ctrl`+`/` to toggle commenting.)
# 3. Fix the code (so that it accomplishes its intended purpose without throwing an exception)
# 
# <!-- TODO: should this be autochecked? Delta is probably pretty small. -->

# In[ ]:


# ruound_to_two_places(9.9999)

# In[ ]:


# x = -10
# y = 5
# # Which of the two variables above has the smallest absolute value?
# smallest_abs = min(abs(x, y))

# In[ ]:


# def f(x):
#     y = abs(x)
# return y

# print(f(5))

# ## 5. <span title="A bit spicy" style="color: darkgreen ">🌶️</span>
# 
# For this question, we'll be using two functions imported from Python's `time` module.
# 
# The [time](https://docs.python.org/3/library/time.html#time.time) function returns the number of seconds that have passed since the Epoch (aka [Unix time](https://en.wikipedia.org/wiki/Unix_time)). 
# 
# <!-- We've provided a function called `seconds_since_epoch` which returns the number of seconds that have passed since the Epoch (aka [Unix time](https://en.wikipedia.org/wiki/Unix_time)). -->
# 
# Try it out below. Each time you run it, you should get a slightly larger number.

# In[ ]:


# Importing the function 'time' from the module of the same name. 
# (We'll discuss imports in more depth later)
from time import time
t = time()
print(t, "seconds since the Epoch")

# We'll also be using a function called [sleep](https://docs.python.org/3/library/time.html#time.sleep), which makes us wait some number of seconds while it does nothing particular. (Sounds useful, right?)
# 
# You can see it in action by running the cell below:

# In[ ]:


from time import sleep
duration = 5
print("Getting sleepy. See you in", duration, "seconds")
sleep(duration)
print("I'm back. What did I miss?")

# With the help of these functions, complete the function `time_call` below according to its docstring.
# 
# <!-- (The sleep function will be useful for testing here since we have a pretty good idea of what something like `time_call(sleep, 1)` should return.) -->

# In[ ]:


def time_call(fn, arg):
    """Return the amount of time the given function takes (in seconds) when called with the given argument.
    """
    pass

# How would you verify that `time_call` is working correctly? Think about it, and then check the answer with the `solution` function below`.

# In[ ]:


#q5.hint()
#q5.solution()

# ## 6. <span title="A bit spicy" style="color: darkgreen ">🌶️</span>
# 
# *Note: this question depends on a working solution to the previous question.*
# 
# Complete the function below according to its docstring.

# In[ ]:


def slowest_call(fn, arg1, arg2, arg3):
    """Return the amount of time taken by the slowest of the following function
    calls: fn(arg1), fn(arg2), fn(arg3)
    """
    pass

# In[ ]:


#q6.hint()

# In[ ]:


#q6.solution()

# 
# **You'll get another email tomorrow so you can keep learning. See you then.**
