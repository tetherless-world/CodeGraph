#!/usr/bin/env python
# coding: utf-8

# Welcome to day 4 of the Python Challenge! If you missed any of the previous days, here are the links:
# 
# - [Day 1 (syntax, variable assignment, numbers)](https://www.kaggle.com/colinmorris/learn-python-challenge-day-1)
# - [Day 2 (functions and getting help)](https://www.kaggle.com/colinmorris/learn-python-challenge-day-2)
# - [Day 3 (booleans and conditionals)](https://www.kaggle.com/colinmorris/learn-python-challenge-day-3)
# 
# Today we'll be talking about **lists**.
# 
# Lists in Python represent ordered sequences of values. They can be defined with comma-separated values between square brackets. For example, here is a list of the first few prime numbers:

# In[ ]:


primes = [2, 3, 5, 7]

# We can put other types of things in lists:

# In[ ]:


planets = ['Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune']

# Including other lists:

# In[ ]:


hands = [
    ['J', 'Q', 'K'],
    ['2', '2', '2'],
    ['6', 'A', 'K'], # (Comma after the last element is optional)
]
# (I could also have written this on one line, but it can get hard to read)
hands = [['J', 'Q', 'K'], ['2', '2', '2'], ['6', 'A', 'K']]

# A list can contain a mix of different types:

# In[ ]:


my_favourite_things = [32, 'raindrops on roses', help]
# (Yes, Python's help function is *definitely* one of my favourite things)

# ## Indexing
# 
# We can access individual list elements using Python's square bracket indexing syntax.
# 
# Which planet is closest to the sun? Python uses *zero-based* indexing, so the first element has index 0.

# In[ ]:


planets[0]

# What's the next closest planet?

# In[ ]:


planets[1]

# Which planet is *furthest* from the sun?
# 
# Elements at the end of the list can be accessed with negative numbers, starting from -1:

# In[ ]:


planets[-1]

# In[ ]:


planets[-2]

# ## Slicing
# 
# What are the first three planets? We can answer this question using *slicing*:

# In[ ]:


planets[0:3]

# `planets[0:3]` is our way of asking Python for the elements of `planets` starting from index 0 and continuing up to *but not including* index 3.
# 
# The starting and ending indices are both optional. If I leave out the start index, it's assumed to be 0. So I could rewrite the expression above as:

# In[ ]:


planets[:3]

# If I leave out the end index, it's assumed to be the length of the list.

# In[ ]:


planets[3:]

# i.e. the expression above means "give me all the planets from index 3 onward".
# 
# We can also use negative indices when slicing:

# In[ ]:


# All the planets except the first and last
planets[1:-1]

# In[ ]:


# The last 3 planets
planets[-3:]

# <!-- TODO: include third 'step size' slice argument and its use for reversing? -->
# 
# ## Mutating lists
# 
# Lists are mutable, meaning they can be modified "in place".
# 
# One way to modify a list is to assign to an index or slice expression.
# 
# For example, let's say we want to rename Mars:

# In[ ]:


planets[3] = 'Malacandra'
planets

# Hm, that's quite a mouthful. Let's compensate by shortening the names of the first 3 planets.

# In[ ]:


planets[:3] = ['Mur', 'Vee', 'Ur']
print(planets)
# (Okay, that was rather silly. Let's give them back their old names)
planets[:4] = ['Mercury', 'Venus', 'Earth', 'Mars',]

# ## List functions
# 
# Python has several useful builtin functions for working with lists.
# 
# `len` gives the length of a list:

# In[ ]:


# How many planets are there?
len(planets)

# `sorted` returns a sorted version of a list:

# In[ ]:


# The planets sorted in alphabetical order
sorted(planets)

# `sum` does what you might expect:

# In[ ]:


primes = [2, 3, 5, 7]
sum(primes)

# We've previously used the `min` and `max` to get the minimum or maximum of several arguments. But we can also pass in a single list argument.

# In[ ]:


max(primes)

# ## Interlude: objects
# 
# I've been throwing around the term 'object' a lot so far - you may even remember me claiming that *everything* in Python is an object. But what does that actually mean?
# 
# In short, objects carry some stuff around with them. We can access that stuff using Python's dot syntax.
# 
# For example, numbers in Python carry around an associated variable called `imag` representing their imaginary part. (You'll probably never need to use this unless you're doing some very weird math.)

# In[ ]:


x = 12
# x is a real number, so its imaginary part is 0.
print(x.imag)
# Here's how to make a complex number, in case you've ever been curious:
c = 12 + 3j
print(c.imag)

# The things an object carries around can also include functions. A function attached to an object is called a **method**. (Non-function things attached to an object, such as `imag`, are called *attributes*).
# 
# For example, numbers have a method called `bit_length`. Again, we access it using dot syntax:

# In[ ]:


x.bit_length

# To actually call it, we add parentheses:

# In[ ]:


x.bit_length()

# > **Aside:** You've actually been calling methods for the last few days if you've been doing the exercises. In the exercise notebooks `q1`, `q2`, `q3`, etc. are all objects which have methods called `check`, `hint`, and `solution`.
# 
# In the same way that we can pass functions to the `help` function (e.g. `help(max)`), we can also pass in methods:

# In[ ]:


help(x.bit_length)

# <!-- TODO:
# dir?
# A useful builtin method for interacting with objects is `dir`. `dir` asks: what are the names of all the things (methods, and attributes) that this object is carrying around?
# help(x)?
# -->
# 
# The examples above were utterly obscure. None of the types of objects we've looked at so far (numbers, functions, booleans) have attributes or methods you're likely ever to use.
# 
# But it turns out that lists have several methods which you'll use all the time.

# ## List methods

# `list.append` modifies a list by adding an item to the end:

# In[ ]:


# Pluto is a planet darn it!
planets.append('Pluto')

# Why does the cell above have no output? Let's check the documentation by calling `help(planets.append)`.
# 
# > **Aside:** `append` is a method carried around by *all* objects of type list, not just `planets`, so we also could have called `help(list.append)`. However, if we try to call `help(append)`, Python will complain that no variable exists called "append". The "append" name only exists within lists - it doesn't exist as a standalone name like builtin functions such as `max` or `len`.

# In[ ]:


help(planets.append)

# The `-> None` part is telling us that `list.append` doesn't return anything. But if we check the value of `planets`, we can see that the method call modified the value of `planets`:

# In[ ]:


planets

# `list.pop` removes and returns the last element of a list:

# In[ ]:


planets.pop()

# In[ ]:


planets

# ### Searching lists
# 
# Where does Earth fall in the order of planets? We can get its index using the `list.index` method.

# In[ ]:


planets.index('Earth')

# It comes third (i.e. at index 2 - 0 indexing!).
# 
# At what index does Pluto occur?

# In[ ]:


planets.index('Pluto')

# Oh, that's right...
# 
# To avoid unpleasant surprises like this, we can use the `in` operator to determine whether a list contains a particular value:

# In[ ]:


# Is Earth a planet?
"Earth" in planets

# In[ ]:


# Is Calbefraques a planet?
"Calbefraques" in planets

# There are a few more interesting list methods we haven't covered. If you want to learn about all the methods and attributes attached to a particular object, we can call `help()` on the object itself. For example, `help(planets)` will tell us about *all* the list methods: 

# In[ ]:


help(planets)

# Click the "output" button to see the full help page. Lists have lots of methods with weird-looking names like `__eq__` and `__iadd__`. Don't worry too much about these for now. (You'll probably never call such methods directly. But they get called behind the scenes when we use syntax like indexing or comparison operators.) The most interesting methods are toward the bottom of the list (`append`, `clear`, `copy`, etc.).

# ## Tuples
# 
# Tuples are almost exactly the same as lists. They differ in just two ways.
# 
# **1:** The syntax for creating them uses (optional) parentheses rather than square brackets

# In[ ]:


t = (1, 2, 3)

# In[ ]:


t = 1, 2, 3 # equivalent to above
t

# **2:** They cannot be modified (they are *immutable*).

# In[ ]:


t[0] = 100

# Tuples are often used for functions that have multiple return values.
# 
# For example, the ``as_integer_ratio()`` method of float objects returns a numerator and a denominator in the form of a tuple:

# In[ ]:


x = 0.125
x.as_integer_ratio()

# These multiple return values can be individually assigned as follows:

# In[ ]:


numerator, denominator = x.as_integer_ratio()
print(numerator / denominator)

# Finally we have some insight into this classic Stupid Python Trick™ for swapping two variables!

# In[ ]:


a = 1
b = 0
a, b = b, a
print(a, b)

# 
# 
# # Your turn!
# 
# Head over to [the Exercises notebook](https://www.kaggle.com/kernels/fork/961685) to get some hands-on practice working with lists.
