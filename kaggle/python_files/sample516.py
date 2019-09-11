#!/usr/bin/env python
# coding: utf-8

# **[Python Micro-Course Home Page](https://www.kaggle.com/learn/python)**
# 
# ---
# 

# This lesson will be a double-shot of essential Python types: **strings** and **dictionaries**.

# # Strings

# One place where the Python language really shines is in the manipulation of strings.
# This section will cover some of Python's built-in string methods and formatting operations.
# 
# Such string manipulation patterns come up often in the context of data science work, and is one big perk of Python in this context.
# 
# ## String syntax
# 
# You've already seen plenty of strings in examples during the previous lessons, but just to recap, strings in Python can be defined using either single or double quotations. They are functionally equivalent.

# In[ ]:


x = 'Pluto is a planet'
y = "Pluto is a planet"
x == y

# Double quotes are convenient if your string contains a single quote character (e.g. representing an apostrophe).
# 
# Similarly, it's easy to create a string that contains double-quotes if you wrap it in single quotes:

# In[ ]:


print("Pluto's a planet!")
print('My dog is named "Pluto"')

# If we try to put a single quote character inside a single-quoted string, Python gets confused:

# In[ ]:


'Pluto's a planet!'

# We can fix this by "escaping" the single quote with a backslash. 

# In[ ]:


'Pluto\'s a planet!'

# The table below summarizes some important uses of the backslash character.
# 
# | What you type... | What you get | example               | `print(example)`             |
# |--------------|----------------|--------------------------------------------------------|
# | `\'`         | `'`            | `'What\'s up?'`         | `What's up?`                 |  
# | `\"`         | `"`            | `"That's \"cool\""`     | `That's "cool"`              |  
# | `\\`         | `\`            |  `"Look, a mountain: /\\"` |  `Look, a mountain: /\`  |
# | `\n`        |   <br/>      |   `"1\n2 3"`                       |   `1`<br/>`2 3`              |

# The last sequence, `\n`, represents the *newline character*. It causes Python to start a new line.

# In[ ]:


hello = "hello\nworld"
print(hello)

# In addition, Python's triple quote syntax for strings lets us include newlines literally (i.e. by just hitting 'Enter' on our keyboard, rather than using the special '\n' sequence). We've already seen this in the docstrings we use to document our functions, but we can use them anywhere we want to define a string.

# In[ ]:


triplequoted_hello = """hello
world"""
print(triplequoted_hello)
triplequoted_hello == hello

# The `print()` function automatically adds a newline character unless we specify a value for the keyword argument `end` other than the default value of `'\n'`:

# In[ ]:


print("hello")
print("world")
print("hello", end='')
print("pluto", end='')

# ## Strings are sequences
# 
# Strings can be thought of as sequences of characters. Almost everything we've seen that we can do to a list, we can also do to a string.

# In[ ]:


# Indexing
planet = 'Pluto'
planet[0]

# In[ ]:


# Slicing
planet[-3:]

# In[ ]:


# How long is this string?
len(planet)

# In[ ]:


# Yes, we can even loop over them
[char+'! ' for char in planet]

# But a major way in which they differ from lists is that they are *immutable*. We can't modify them.

# In[ ]:


planet[0] = 'B'
# planet.append doesn't work either

# ## String methods
# 
# Like `list`, the type `str` has lots of very useful methods. I'll show just a few examples here.

# In[ ]:


# ALL CAPS
claim = "Pluto is a planet!"
claim.upper()

# In[ ]:


# all lowercase
claim.lower()

# In[ ]:


# Searching for the first index of a substring
claim.index('plan')

# In[ ]:


claim.startswith(planet)

# In[ ]:


claim.endswith('dwarf planet')

# ### Going between strings and lists: `.split()` and `.join()`
# 
# `str.split()` turns a string into a list of smaller strings, breaking on whitespace by default. This is super useful for taking you from one big string to a list of words.

# In[ ]:


words = claim.split()
words

# Occasionally you'll want to split on something other than whitespace:

# In[ ]:


datestr = '1956-01-31'
year, month, day = datestr.split('-')

# `str.join()` takes us in the other direction, sewing a list of strings up into one long string, using the string it was called on as a separator.

# In[ ]:


'/'.join([month, day, year])

# In[ ]:


# Yes, we can put unicode characters right in our string literals :)
' 👏 '.join([word.upper() for word in words])

# ### Building strings with `.format()`
# 
# Python lets us concatenate strings with the `+` operator.

# In[ ]:


planet + ', we miss you.'

# If we want to throw in any non-string objects, we have to be careful to call `str()` on them first

# In[ ]:


position = 9
planet + ", you'll always be the " + position + "th planet to me."

# In[ ]:


planet + ", you'll always be the " + str(position) + "th planet to me."

# This is getting hard to read and annoying to type. `str.format()` to the rescue.

# In[ ]:


"{}, you'll always be the {}th planet to me.".format(planet, position)

# So much cleaner! We call `.format()` on a "format string", where the Python values we want to insert are represented with `{}` placeholders.
# 
# Notice how we didn't even have to call `str()` to convert `position` from an int. `format()` takes care of that for us.
# 
# If that was all that `format()` did, it would still be incredibly useful. But as it turns out, it can do a *lot* more. Here's just a taste:

# In[ ]:


pluto_mass = 1.303 * 10**22
earth_mass = 5.9722 * 10**24
population = 52910390
#         2 decimal points   3 decimal points, format as percent     separate with commas
"{} weighs about {:.2} kilograms ({:.3%} of Earth's mass). It is home to {:,} Plutonians.".format(
    planet, pluto_mass, pluto_mass / earth_mass, population,
)

# In[ ]:


# Referring to format() arguments by index, starting from 0
s = """Pluto's a {0}.
No, it's a {1}.
{0}!
{1}!""".format('planet', 'dwarf planet')
print(s)

# You could probably write a short book just on `str.format`, so I'll stop here, and point you to [pyformat.info](https://pyformat.info/) and [the official docs](https://docs.python.org/3/library/string.html#formatstrings) for further reading.

# # Dictionaries
# 
# Dictionaries are a built-in Python data structure for mapping keys to values.

# In[ ]:


numbers = {'one':1, 'two':2, 'three':3}

# In this case `'one'`, `'two'`, and `'three'` are the **keys**, and 1, 2 and 3 are their corresponding values.
# 
# Values are accessed via square bracket syntax similar to indexing into lists and strings.

# In[ ]:


numbers['one']

# We can use the same syntax to add another key, value pair

# In[ ]:


numbers['eleven'] = 11
numbers

# Or to change the value associated with an existing key

# In[ ]:


numbers['one'] = 'Pluto'
numbers

# Python has *dictionary comprehensions* with a syntax similar to the list comprehensions we saw in the previous tutorial.

# In[ ]:


planets = ['Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune']
planet_to_initial = {planet: planet[0] for planet in planets}
planet_to_initial

# The `in` operator tells us whether something is a key in the dictionary

# In[ ]:


'Saturn' in planet_to_initial

# In[ ]:


'Betelgeuse' in planet_to_initial

# A for loop over a dictionary will loop over its keys

# In[ ]:


for k in numbers:
    print("{} = {}".format(k, numbers[k]))

# We can access a collection of all the keys or all the values with `dict.keys()` and `dict.values()`, respectively.

# In[ ]:


# Get all the initials, sort them alphabetically, and put them in a space-separated string.
' '.join(sorted(planet_to_initial.values()))

# The very useful `dict.items()` method lets us iterate over the keys and values of a dictionary simultaneously. (In Python jargon, an **item** refers to a key, value pair)

# In[ ]:


for planet, initial in planet_to_initial.items():
    print("{} begins with \"{}\"".format(planet.rjust(10), initial))

# To read a full inventory of dictionaries' methods, click the "output" button below to read the full help page, or check out the [official online documentation](https://docs.python.org/3/library/stdtypes.html#dict).

# In[ ]:


help(dict)

# # Your Turn
# 
# Try the [hands-on exercise](https://www.kaggle.com/kernels/fork/1275185) with strings and dictionaries
# 

# ---
# **[Python Micro-Course Home Page](https://www.kaggle.com/learn/python)**
# 
# 
