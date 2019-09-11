#!/usr/bin/env python
# coding: utf-8

# Welcome to day 2 of the Learn Python challenge! If you missed day 1, you can find the notebook [here](https://www.kaggle.com/colinmorris/learn-python-challenge-day-1/notebook). 
# 
# Today we'll be talking about functions: calling them, defining them, and looking them up using Python's built-in documentation.
# 
# In some languages, functions must be defined to always take a specific number of arguments, each having a particular type. Python functions are allowed much more flexibility. The `print` function is a good example of this:

# In[ ]:


print("The print function takes an input and prints it to the screen.")
print("Each call to print starts on a new line.")
print("You'll often call print with strings, but you can pass any kind of value. For example, a number:")
print(2 + 2)
print("If print is called with multiple arguments...", "it joins them",
      "(with spaces in between)", "before printing.")
print('But', 'this', 'is', 'configurable', sep='!...')
print()
print("^^^ print can also be called with no arguments to print a blank line.")

# ## "What does this function do again?"
# 
# I showed the `abs` function in the previous lesson, but what if you've forgotten what it does?
# 
# The `help()` function is possibly the most important Python function you can learn. If you can remember how to use `help()`, you hold the key to understanding just about any other function in Python.

# In[ ]:


help(abs)

# When applied to a function, `help()` displays...
# 
# - the header of that function `abs(x, /)`. In this case, this tells us that `abs()` takes a single argument `x`. (The forward slash isn't important, but if you're curious, you can read about it [here](https://stackoverflow.com/questions/24735311/python-what-does-the-slash-mean-in-the-output-of-helprange))
# - A brief English description of what the function does.

# **Common pitfall:** when you're looking up a function, remember to pass in the name of the function itself, and not the result of calling that function. 
# 
# What happens if we invoke help on a *call* to the function `abs()`? Unhide the output of the cell below to see.

# In[ ]:


help(abs(-2))

# Python evaluates an expression like this from the inside out. First it calculates the value of `abs(-2)`, then it provides help on whatever the value of that expression is.
# 
# <small>(And it turns out to have a lot to say about integers! In Python, even something as simple-seeming as an integer is actually an object with a fair amount of internal complexity. After we talk later about objects, methods, and attributes in Python, the voluminous help output above will make more sense.)</small>
# 
# `abs` is a very simple function with a short docstring. `help` shines even more when dealing with more complex, configurable functions like `print`:

# In[ ]:


help(print)

# Some of this might look inscrutable for now (what's `sys.stdout`?), but this docstring does shed some light on that `sep` parameter we used in one of our `print` examples at the beginning. 

# ## Defining functions
# 
# Builtin functions are great, but we can only get so far with them before we need to start defining our own functions. Below is a simple example.

# In[ ]:


def least_difference(a, b, c):
    diff1 = abs(a - b)
    diff2 = abs(b - c)
    diff3 = abs(a - c)
    return min(diff1, diff2, diff3)

# This creates a function called `least_difference`, which takes three arguments, `a`, `b`, and `c`.
# 
# Functions start with a header introduced by the `def` keyword. The indented block of code following the `:` is run when the function is called.
# 
# `return` is another keyword uniquely associated with functions. When Python encounters a `return` statement, it exits the function immediately, and passes the value on the right hand side to the calling context.
# 
# Is it clear what `least_difference()` does from the source code? If we're not sure, we can always try it out on a few examples:

# In[ ]:


print(
    least_difference(1, 10, 100),
    least_difference(1, 10, 10),
    least_difference(5, 6, 7), # Python allows trailing commas in argument lists. How nice is that?
)

# Or maybe the `help()` function can tell us something about it.

# In[ ]:


help(least_difference)

# Unsurprisingly, Python isn't smart enough to read my code and turn it into a nice English description. However, when I write a function, I can provide a description in what's called the **docstring**.
# 
# ### Docstrings

# In[ ]:


def least_difference(a, b, c):
    """Return the smallest difference between any two numbers
    among a, b and c.
    
    >>> least_difference(1, 5, -5)
    4
    """
    diff1 = abs(a - b)
    diff2 = abs(b - c)
    diff3 = abs(a - c)
    return min(diff1, diff2, diff3)

# The docstring is a triple-quoted string (which may span multiple lines) that comes immediately after the header of a function. When we call `help()` on a function, it shows the docstring.

# In[ ]:


help(least_difference)

# > **Aside: example calls**
# > The last two lines of the docstring are an example function call and result. (The `>>>` is a reference to the command prompt used in Python interactive shells.) Python doesn't run the example call - it's just there for the benefit of the reader. The convention of including 1 or more example calls in a function's docstring is far from universally observed, but it can be very effective at helping someone understand your function. For a real-world example of, see [this docstring for the numpy function `np.eye`](https://github.com/numpy/numpy/blob/v1.14.2/numpy/lib/twodim_base.py#L140-L194).

# 
# 
# Docstrings are a nice way to document your code for others - or even for yourself. (How many times have you come back to some code you were working on a day ago and wondered "what was I thinking?")

# ## Functions that don't return
# 
# What would happen if we didn't include the `return` keyword in our function?

# In[ ]:


def least_difference(a, b, c):
    """Return the smallest difference between any two numbers
    among a, b and c.
    """
    diff1 = abs(a - b)
    diff2 = abs(b - c)
    diff3 = abs(a - c)
    min(diff1, diff2, diff3)
    
print(
    least_difference(1, 10, 100),
    least_difference(1, 10, 10),
    least_difference(5, 6, 7),
)

# Python allows us to define such functions. The result of calling them is the special value `None`. (This is similar to the concept of "null" in other languages.)
# 
# Without a `return` statement, `least_difference` is completely pointless, but a function with side effects may do something useful without returning anything. We've already seen two examples of this: `print()` and `help()` don't return anything. We only call them for their side effects (putting some text on the screen). Other examples of useful side effects include writing to a file, or modifying an input.

# In[ ]:


mystery = print()
print(mystery)

# ## Default arguments
# 
# When we called `help(print)`, we saw that the `print` function has several optional arguments. For example, we can specify a value for `sep` to put some special string in between our printed arguments:

# In[ ]:


print(1, 2, 3, sep=' < ')

# But if we don't specify a value, `sep` is treated as having a default value of `' '` (a single space).

# In[ ]:


print(1, 2, 3)

# Adding optional arguments with default values to the functions we define turns out to be pretty easy:

# In[ ]:


def greet(who="Colin"):
    print("Hello,", who)
    
greet()
greet(who="Kaggle")
# (In this case, we don't need to specify the name of the argument, because it's unambiguous.)
greet("world")

# <!-- Mention that non-default args must come before default? -->
# 
# ## Functions are objects too

# In[ ]:


def f(n):
    return n * 2

x = 12.5

# The syntax for creating them may be different, but `f` and `x` in the code above aren't so fundamentally different. They're each variables that refer to objects. `x` refers to an object of type `float`, and `f` refers to an object of type... well, let's ask Python:

# In[ ]:


print(
    type(x),
    type(f), sep='\n'
)

# We can even ask Python to print `f` out:

# In[ ]:


print(x)
print(f)

# ...though what it shows isn't super useful.
# 
# Notice that the code cells above have examples of functions (`type`, and `print`) taking *another function* as input. This opens up some interesting possibilities - we can *call* the function we receive as an argument.

# In[ ]:


def call(fn, arg):
    """Call fn on arg"""
    return fn(arg)

def squared_call(fn, arg):
    """Call fn on the result of calling fn on arg"""
    return fn(fn(arg))

print(
    call(f, 1),
    squared_call(f, 1), 
    sep='\n', # '\n' is the newline character - it starts a new line
)

# You probably won't often define [higher order functions](https://en.wikipedia.org/wiki/Higher-order_function) like this yourself, but there are some existing ones (built in to Python and in libraries like pandas or numpy) that you might find useful to call. For example, `max`.
# 
# By default, `max` returns the largest of its arguments. But if we pass in a function using the optional `key` argument, it returns the argument `x` that maximizes `key(x)` (aka the 'argmax').

# In[ ]:


def mod_5(x):
    """Return the remainder of x after dividing by 5"""
    return x % 5

print(
    'Which number is biggest?',
    max(100, 51, 14),
    'Which number is the biggest modulo 5?',
    max(100, 51, 14, key=mod_5),
    sep='\n',
)

# ### Lambda functions
# 
# If you're writing a short throwaway function whose body fits into a single line (like `mod_5` above), Python's `lambda` syntax is conveniently compact.

# In[ ]:


mod_5 = lambda x: x % 5

# Note that we don't use the "return" keyword above (it's implicit)
# (The line below would produce a SyntaxError)
#mod_5 = lambda x: return x % 5

print('101 mod 5 =', mod_5(101))

# In[ ]:


# Lambdas can take multiple comma-separated arguments
abs_diff = lambda a, b: abs(a-b)
print("Absolute difference of 5 and 7 is", abs_diff(5, 7))

# In[ ]:


# Or no arguments
always_32 = lambda: 32
always_32()

# With judicious use of lambdas, you can occasionally solve complex problems in a single line. 

# In[ ]:


# Preview of lists and strings. (We'll go in depth into both soon)
# - len: return the length of a sequence (such as a string or list)
# - sorted: return a sorted version of the given sequence (optional key 
#           function works similarly to max and min)
# - s.lower() : return a lowercase version of string s
names = ['jacques', 'Ty', 'Mia', 'pui-wa']
print("Longest name is:", max(names, key=lambda name: len(name))) # or just key=len
print("Names sorted case insensitive:", sorted(names, key=lambda name: name.lower()))

# ## Your Turn
# 
# <!-- [Click here to start the coding exercises for day 2!](https://www.kaggle.com/kernels/fork/934361) -->
# 
# To get started on the day 2 exercises, head over to [this notebook](https://www.kaggle.com/colinmorris/learn-python-challenge-day-2-exercises) and click the "Fork Notebook" button.

# 
