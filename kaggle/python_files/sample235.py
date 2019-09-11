#!/usr/bin/env python
# coding: utf-8

# ## Strings

# Strings are ordered text based data which are represented by enclosing the same in single/double/triple quotes.

# In[1]:


String0 = 'Taj Mahal is beautiful'
String1 = "Taj Mahal is beautiful"
String2 = '''Taj Mahal
is
beautiful'''

# In[2]:


print(String0 , type(String0))
print(String1, type(String1))
print(String2, type(String2))

# String Indexing and Slicing are similar to Lists which was explained in detail earlier.

# In[3]:


print(String0[4])
print(String0[4:])

# ### Built-in Functions

# **find( )** function returns the index value of the given data that is to found in the string. If it is not found it returns **-1**. Remember to not confuse the returned -1 for reverse indexing value.

# In[4]:


print(String0.find('al'))
print(String0.find('am'))

# The index value returned is the index of the first element in the input data.

# In[5]:


print(String0[7])

# One can also input **find( )** function between which index values it has to search.

# In[6]:


print(String0.find('j',1))
print(String0.find('j',1,3))

# **capitalize( )** is used to capitalize the first element in the string.

# In[7]:


String3 = 'observe the first letter in this sentence.'
print(String3.capitalize())

# **center( )** is used to center align the string by specifying the field width.

# In[8]:


String0.center(70)

# One can also fill the left out spaces with any other character.

# In[9]:


String0.center(70,'-')

# **zfill( )** is used for zero padding by specifying the field width.

# In[10]:


String0.zfill(30)

# **expandtabs( )** allows you to change the spacing of the tab character. '\t' which is by default set to 8 spaces.

# In[12]:


s = 'h\te\tl\tl\to'
print(s)
print(s.expandtabs(1))
print(s.expandtabs())

# **index( )** works the same way as **find( )** function the only difference is find returns '-1' when the input element is not found in the string but **index( )** function throws a ValueError

# In[13]:


print(String0.index('Taj'))
print(String0.index('Mahal',0))
print(String0.index('Mahal',10,20))

# **endswith( )** function is used to check if the given string ends with the particular char which is given as input.

# In[14]:


print(String0.endswith('y'))

# The start and stop index values can also be specified.

# In[15]:


print(String0.endswith('l',0))
print(String0.endswith('M',0,5))

# **count( )** function counts the number of char in the given string. The start and the stop index can also be specified or left blank. (These are Implicit arguments which will be dealt in functions)

# In[16]:


print(String0.count('a',0))
print(String0.count('a',5,10))

# **join( )** function is used add a char in between the elements of the input string.

# In[17]:


'a'.join('*_-')

# '*_-' is the input string and char 'a' is added in between each element

# **join( )** function can also be used to convert a list into a string.

# In[18]:


a = list(String0)
print(a)
b = ''.join(a)
print(b)

# Before converting it into a string **join( )** function can be used to insert any char in between the list elements.

# In[19]:


c = '/'.join(a)[18:]
print(c)

# **split( )** function is used to convert a string back to a list. Think of it as the opposite of the **join()** function.

# In[20]:


d = c.split('/')
print(d)

# In **split( )** function one can also specify the number of times you want to split the string or the number of elements the new returned list should conatin. The number of elements is always one more than the specified number this is because it is split the number of times specified.

# In[21]:


e = c.split('/',3)
print(e)
print(len(e))

# **lower( )** converts any capital letter to small letter.

# In[22]:


print(String0)
print(String0.lower())

# **upper( )** converts any small letter to capital letter.

# In[23]:


String0.upper()

# **replace( )** function replaces the element with another element.

# In[24]:


String0.replace('Taj Mahal','Bengaluru')

# **strip( )** function is used to delete elements from the right end and the left end which is not required.

# In[25]:


f = '    hello      '

# If no char is specified then it will delete all the spaces that is present in the right and left hand side of the data.

# In[26]:


f.strip()

# **strip( )** function, when a char is specified then it deletes that char if it is present in the two ends of the specified string.

# In[27]:


f = '   ***----hello---*******     '

# In[28]:


f.strip('*')

# The asterisk had to be deleted but is not. This is because there is a space in both the right and left hand side. So in strip function. The characters need to be inputted in the specific order in which they are present.

# In[29]:


print(f.strip(' *'))
print(f.strip(' *-'))

# **lstrip( )** and **rstrip( )** function have the same functionality as strip function but the only difference is **lstrip( )** deletes only towards the left side and **rstrip( )** towards the right.

# In[31]:


print(f.lstrip(' *'))
print(f.rstrip(' *'))

# ## Dictionaries

# Dictionaries are more used like a database because here you can index a particular sequence with your user defined string.

# To define a dictionary, equate a variable to { } or dict()

# In[32]:


d0 = {}
d1 = dict()
print(type(d0), type(d1))

# Dictionary works somewhat like a list but with an added capability of assigning it's own index style.

# In[33]:


d0['One'] = 1
d0['OneTwo'] = 12 
print(d0)

# That is how a dictionary looks like. Now you are able to access '1' by the index value set at 'One'

# In[35]:


print(d0['One'])

# Two lists which are related can be merged to form a dictionary.

# In[36]:


names = ['One', 'Two', 'Three', 'Four', 'Five']
numbers = [1, 2, 3, 4, 5]

# **zip( )** function is used to combine two lists

# In[37]:


d2 = zip(names,numbers)
print(d2)

# The two lists are combined to form a single list and each elements are clubbed with their respective elements from the other list inside a tuple. Tuples because that is what is assigned and the value should not change.
# 
# Further, To convert the above into a dictionary. **dict( )** function is used.

# In[39]:


a1 = dict(d2)
print(a1)

# ### Built-in Functions

# **clear( )** function is used to erase the entire database that was created.

# In[40]:


a1.clear()
print(a1)

# Dictionary can also be built using loops.

# In[41]:


for i in range(len(names)):
    a1[names[i]] = numbers[i]
print(a1)

# **values( )** function returns a list with all the assigned values in the dictionary.

# In[42]:


a1.values()

# **keys( )** function returns all the index or the keys to which contains the values that it was assigned to.

# In[43]:


a1.keys()

# **items( )** is returns a list containing both the list but each element in the dictionary is inside a tuple. This is same as the result that was obtained when zip function was used.

# In[44]:


a1.items()

# **pop( )** function is used to get the remove that particular element and this removed element can be assigned to a new variable. But remember only the value is stored and not the key. Because the is just a index value.

# In[46]:


a2 = a1.pop('Four')
print(a1)
print(a2)

# In[ ]:



