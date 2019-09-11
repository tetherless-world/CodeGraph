#!/usr/bin/env python
# coding: utf-8

# # Intro
# 
# You don't directly choose the numbers to go into your convolutions for deep learning... instead the deep learning technique determines what convolutions will be useful from the data (as part of model-training). We'll come back to how the model does that soon.
# 
# ![Imgur](https://i.imgur.com/op9Maqr.png)
# 
# But looking closely at convolutions and how they are applied to your image will improve your intuition for these models, how they work, and how to debug them when they don't work.
# 
# **Let's get started.**
# 
# # Exercises
# We'll use some small utilty functions to visualize raw images and some results of your code. Execute the next cell to load the utility functions.

# In[ ]:


# Set up code checking
from learntools.core import binder
binder.bind(globals())
from learntools.deep_learning.exercise_1 import *
print("Setup Complete")

# ### Exercise 1
# 
# In the video, you saw a convolution that detected horizontal lines. That convolution shows up again in the code cell below.
# 
# Run the cell to see a raw image as well as the output from applying this convolution to the image.
# 

# In[ ]:


horizontal_line_conv = [[1, 1], 
                        [-1, -1]]
# load_my_image and visualize_conv are utility functions provided for this exercise
original_image = load_my_image() 
visualize_conv(original_image, horizontal_line_conv)

# Now it's your turn. Instead of a horizontal line detector, you will create a vertical line detector.
# 
# **Replace the underscores with numbers to make a vertical line detector and uncomment both lines of code in the cell below. Then run **

# In[ ]:


vertical_line_conv = ____

q_1.check()
visualize_conv(original_image, vertical_line_conv)

# If you'd like a hint or the solution, uncomment the appropriate line below.

# In[ ]:


#q_1.hint()
#q_1.solution()

# ### Exercise 2
# The convolutions you've seen are 2x2.  But you could have larger convolutions. They could be 3x3, 4x4, etc.  They don't even have to be square. Nothing prevents using a 4x7 convolution.
# 
# Compare the number of visual patterns that can be captured by small convolutions. Which of the following is true?
# 
# - There are more visual patterns that can be captured by large convolutions
# - There are fewer visual patterns that can be captured by large convolutions
# - The number of visual patterns that can be captured by large convolutions is the same as the number of visual patterns that can be captured by small convolutions?
# 
# Once you think you know the answer, check it by uncommenting and running the line below.

# In[ ]:


#q_2.solution()

# # Keep Going
# Now you are ready to **[combine convolutions into powerful models](https://www.kaggle.com/dansbecker/building-models-from-convolutions).** These models are fun to work with, so keep going.
# 
# ---
# **[Deep Learning Course Home Page](https://www.kaggle.com/learn/deep-learning)**
# 
