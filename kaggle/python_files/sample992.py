#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# The Rectified Linear Unit is the most commonly used activation function in deep learning models.  The function returns 0 if it receives any negative input, but for any positive value $x$ it returns that value back. So it can be written as  $f(x) = max(0,x)$.  
# 
# Graphically it looks like this
# 
# ![ReLU image](https://i.imgur.com/gKA4kA9.jpg)
# 
# It's surprising that such a simple function (and one composed of two linear pieces) can allow your model to account for non-linearities and interactions so well.  But the ReLU function works great in most applications, and it is very widely used as a result.
# 
# # Why It Works
# ## Introducing Interactions and Non-linearities
# 
# Activation functions serve two primary purposes:
# 1) Help a model account for **interaction effects**.  
# What is an interactive effect?  It is when one variable A affects a prediction differently depending on the value of B. For example, if my model wanted to know whether a certain body weight indicated an increased risk of diabetes, it would have to know an individual's height.  Some bodyweights indicate elevated risks for short people, while indicating good health for tall people.  So, the **effect of body weight on diabetes risk depends on height**, and we would say that **weight and height have an interaction effect**.
# 
# 2) Help a model account for **non-linear effects.**
# This just means that if I graph a variable on the horizontal axis, and my predictions on the vertical axis, it isn't a straight line.  Or said another way, the effect of increasing the predictor by one is different at different values of that predictor.
# 
# ## How ReLU captures Interactions and Non-Linearities
# ** Interactions:** Imagine a single node in a neural network model.  For simplicity, assume it has two inputs, called A and B.  The weights from A and B into our node are 2 and 3 respectively.  So the node output is $f(2A + 3B)$. We'll use the ReLU function for our f.  So, if $2A + 3B$ is positive, the output value of our node is also $2A + 3B$. If $2A + 3B$ is negative, the output value of our node is 0.
# 
# For concreteness, consider a case where A=1 and B=1.  The output is $2A + 3B$, and if A increases, then the output increases too.  On the other hand, if B=-100 then the output is 0, and if A increases moderately, the output remains 0.  So A might increase our output, or it might not.  It just depends what the value of B is.
# 
# This is a simple case where the node captured an interaction. As you add more nodes and more layers, the potential complexity of interactions only increases.  But you should now see how the activation function helped capture an interaction.
# 
# **Non-linearities:**  A function is non-linear if the slope isn't constant.  So, the ReLU function is non-linear around 0, but the slope is always either 0 (for negative values) or 1 (for positive values).  That's a very limited type of non-linearity.  
# 
# But two facts about deep learning models allow us to create many different types of non-linearities from how we combine ReLU nodes.
# 
# First, most models include a **bias** term for each node.  The bias term is just a constant number that is determined during model training.  For simplicity, consider a node with a single input called A, and a bias.  If the bias term takes a value of 7, then the node output is f(7+A).  In this case, if A is less than -7, the output is 0 and the slope is 0.  If A is greater than -7, then the node's output is 7+A, and the slope is 1.  
# 
# So the bias term allows us to move where the slope changes. So far, it still appears we can have only two different slopes.
# 
# However, real models have many nodes. Each node (even within a single layer) can have a different value for it's bias, so each node can change slope at different values for our input.  
# 
# When we add the resulting functions back up, we get a combined function that changes slopes in many places.
# 
# These models have the flexibility to produce non-linear functions and account for interactions well (if that will giv better predictions).  As we add more nodes in each layer (or more convolutions if we are using a convolutional model) the model gets even greater ability to represent these interactions and non-linearities.
# 
# 
# 
# ## Facilitating Gradient Descent
# This section is more technical than those above it. If you find it difficult, remember that you can have a lot of success using deep learning even without this technical background.
# 
# Historically, deep learning models started off with s-shaped curves (like the tanh function below)
# ![Imgur](https://i.imgur.com/Q1jQejl.png)
# 
# The tanh would seem to have a couple advantages.  Even though it gets close to flat, it isn't completely flat anywhere.  So it's output always reflects changes in it's input, which we might expect to be a good thing.  Secondly, it is non-linear (or curved everywhere).  Accounting for non-linearities is one of the activation function's main purposes.  So, we expect a non-linear function to work well.
# 
# However researchers had great difficulty building models with many layers when using the tanh function.  It is relatively flat except for a very narrow range (that range being about -2 to 2).  The derivative of the function is very small unless the input is in this narrow range, and this flat derivative makes it difficult to improve the weights through gradient descent.  This problem gets worse as the model has more layers.  This was called the **vanishing gradient problem**.
# 
# The ReLU function has a derivative of 0 over half it's range (the negative numbers).  For positive inputs, the derivative is 1.
# 
# When training on a reasonable sized batch, there will usually be some data points giving positive values to any given node.  So the average derivative is rarely close to 0, which allows gradient descent to keep progressing.
# 
# # Alternatives
# 
# There are many similar alternatives which also work well.  The Leaky ReLU is one of the most well known.  It is the same as ReLU for positive numbers.  But instead of being 0 for all negative values, it has a constant slope (less than 1.). 
# 
# That slope is a parameter the user sets when building the model, and it is frequently called $\alpha$.  For example, if the user sets $\alpha = 0.3$, the activation function is `f(x) = max(0.3*x, x)`.  This has the theoretical advantage that, by being influenced by `x` at all values, it may be make more complete use of the information contained in `x`.  
# 
# Their are other alternatives, but both practitioners and researchers have generally found insufficient benefit to justify using anything other than ReLU.
