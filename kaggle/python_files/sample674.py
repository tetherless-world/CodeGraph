#!/usr/bin/env python
# coding: utf-8

# This is day one of a three day event, held during [Kaggle CareerCon 2019](https://www.kaggle.com/careercon2019). Each day we’ll learn about a new part of developing an API and put it into practice. By day 3, you’ll have written and deployed an API of your very own!
# 
#  * **[Day 1: The Basics of Rest APIs – What They Are and How to Design One](https://www.kaggle.com/rtatman/careercon-intro-to-apis).** By the end of this day you’ll have written the OpenAPI specification for your API. 
#  * **[Day 2: How to Make an API for an Existing Python Machine Learning Project](https://www.kaggle.com/rtatman/careercon-making-an-app-from-your-modeling-code).** By the end of this day, you’ll have a Flask app that you can use to serve your model.
#  * **[Day 3: How to deploy your API on your choice of services – Heroku or Google Cloud](https://www.kaggle.com/rtatman/careercon-deploying-apis-on-heroku-appengine/).** By the end of this day, you’ll have deployed your model and will be able to actually use your API! (Note that, in order to use Google Cloud, you’ll need to create a billing account, which requires a credit card. If you don’t have access to a credit, you can still use Heroku.)
# 
# ___

# # Why would a data scientist want to build an API?
# 
# If you’re like me, you probably didn’t build any APIs while you were learning data science. Instead, you might have focused on data cleaning, machine learning, statistics and visualization. But how do you get from “I’ve got a notebook with my modelling code” to “someone can use an app built around my project to draw little hearts around pictures of dogs”?
# 
# Generally **if you’ve built a model that does some task you want people to be able to use it**. If you’re working on a team with software engineers, then they’ll probably take care of putting trained models into production. But even if that’s the case, they’ll probably appreciate it if you don’t just email them a notebook (unless your company is like Netflix and [does everything in notebooks](https://medium.com/netflix-techblog/notebook-innovation-591ee3221233)).
# 
# > "I think the truth is any [software engineer] worth their salt will refuse to put a notebook into production. They'll rewrite it from scratch if need be." - Jeremy Kun, Google Software Engineer
# 
# Rewriting notebooks from scratch, probably in another programming language, isn’t a great use of anyone’s time. If you can give your software engineering colleagues an API instead it will make their life a lot easier, even if they do need to end up re-writing some of it or adding additional functionality. (For example, we’re not going to talk about authentication in these workshops but you probably don’t want to launch a public-facing app without any authentication. 😬) 
# 
# Even if you never need to train models for production, if you have an idea of how APIs work it will be easier for you to use them. (I remember how confused I was the first time I used the Twitter API! I think it was the first time I’d ever had to deal with a JSON file.) Since a *lot* of data and services are only available via API these days, I personally think it’s a good idea for professional data scientists to have a good idea how APIs work. 
# 
# # What is an API?
# 
# API is short for “application programmatic interface”. 
# 
# * **Application** refers to code that’s been written to perform a specific task. An API serves as a way to move information between applications. For example, you may be writing a data science application to get sentiment scores for tweets about your company. In order to do this, you’ll need to get information (tweets about your company) from a different application (Twitter). 
# * **Programmatic** means that you interact with the interface with code. This is different from graphical interfaces, where you have a visual interface. For example, if you use Twitter you can use the Twitter graphical interface to download your own tweets by clicking a button. If you want to download tweets by a number of different people, however, there’s no button to do that. Instead, you’ll need to write code that interacts with the Twitter API and specifies the type and number of tweets you want to download. 
# * **Interface** just means that an API serves as a go-between for two interfaces. **You can think of an API as a postal service.** It defines a set of rules for how to move things around. It also creates the specific addresses you can use to send data to or receive data from. (At least in the US, post offices have little boxes that have their own addresses that you can rent and send things to/from.)
# 
# There are several different ways to design and build APIs but the most commonly used is REST. REST is *also* an acronym; it stands for Representational State Transfer. There are a couple of important concepts that differentiate RESTful APIs. I’ll talk about a few of them here but if you’re *really* excited about RESTful API design, you should check out [Roy Fielding’s dissertation, which outlines the REST design philosophy](https://www.ics.uci.edu/~fielding/pubs/dissertation/rest_arch_style.htm).
# 
# ## Client-server
# 
# REST uses a client-server architecture. This means that information is stored in one place (the server) and that interacting with that data is done via the client. Generally, each server is a single centralized computing resource that serves many clients.
# 
# You’re probably somewhat familiar with this system from using the internet. Your browser (like Chrome, FireFox or Edge) is a client that interacts with the server of the websites you visit (like Kaggle, Twitter or YouTube). So when you interact with a website you’re not actually creating a copy of the entire website and all of its data on your local computer. Instead, you’re sending a series of requests to the server that’s hosting the website and you get back only the data you ask for each time.
# 
# ## Resources 
# 
# How does a client know which data to fetch, through? Or if a client sends data to the server, how is it stored and organized so that you can interact with it later?
# 
# In REST  **each chunk of data or entity is called a resource**. A resource could be anything: a text file, geo-coordinates or a specific customer. (If you’re familiar with object oriented programming, resources are objects.) Resources can have also relationships with one another. For example, each customer might be associated with the geo-coordinates of the stores they've visited. 
# 
# > *But Rachael, what about the “representational” part of REST?* Each resource is stored with additional information about, generally in a JSON file that also contains the resource itself. That file, with the resource and additional information about, is called a representation. To continue with our post office example, if a resource is  a piece of mail, then the representation is the envelope around it with the address and other information. (There’s a [nice discussion of the distinction between resource & representation here](https://lists.w3.org/Archives/Public/public-html/2009Oct/0184.html) if you’re curious.) Assume from here on when I say “resource” I mean “a resource and its representation”.
# 
# ## Methods
# 
# So how do we interact with our resources? REST API’s use methods that allow you to create, interact with and delete resources (you can think of these as functions). For APIs that are served over the web, the most common methods are:
# 
# * GET, which will return one or more resources without changing them
# * POST, which creates new resources
# * PUT, which updates existing resources
# * DELETE, which removes existing resources
# 
# In our example, we’re only going to be using POST. Because we’re not setting up a database for our application that will store data, we’ll need to create a new resource every time we send data to our API.
# 
# # What will we be building? 
# 
# For my example, I’m going to be building an API that will extract keywords and their location (character index) from text I provide to the API. I’m going to specifically be getting the names of popular data science packages, but you could use this technique to find any list of keywords, like names of pet dogs, ice cream flavors or store addresses. You’re also free to develop your own model and build an API to serve that instead. 
# 
# Before we start writing any code, however, I want to make sure I know what my API will look like. I can use this 1) as a guide for how to structure my application and 2) as documentation. That way, when it’s time for someone else to use our API, it’s very easy for us to pass on information on what it does.  
# 
# Here are three questions we want to answer. 
# 
# * What do we want our API to do?
# * What are we passing in to do that thing?
# * What are we getting back when we do that thing?
# 
# For this example, our answers might look something like this.
# 
# * We want our API to tell us where specific keywords appear in our text. 
# * We are passing in a JSON with our text file.
# * We’re getting a JSON with our keyword matches and where they occur in the input text.
# 
# ## Your turn!
# 
# If you’re working on an API that does something different (like draws hearts around any dogs in a picture or classifies rows of a .csv file into “pass” and “fail”) take some time to answer these questions for yourself. 
# 
# It’s possible that you may want your API to do multiple things, in which case you’d want to answer all three questions for each thing you want it to do.

# # Intro to OpenAPI
# 
# Fortunately, we don’t have to rely on just asking ourselves questions to build a specification for our API. There are also some standards we can use to guide our API design. I’ve chosen to use [OpenAPI (formerly known as Swagger)](https://swagger.io/docs/specification/about/) because it’s open source and there’s a nice tooling ecosystem around it. (There are even tools that will build a Flask app for you from your specification!) If your company uses a different system, however, you’ll want to use theirs. 
# 
# Here’s the full specification for our little sample API: 
# 
# ```
# openapi: "3.0.0"
# 
# info:
#   title: "Package name extractor"
#   description: "API that accepts a text string and returns packages names in it."
#   version: "1.0"
# 
# paths:
#   /extractpackages:
#     post:
#       description: "Extract package names"
#       
#       requestBody:
#         description: "Json with single field containing text to extract entities from"
#         required: true
#         content:
#           application/json: {}
#       
#       responses:
#         '200':
#           description: "Returns names & indexs of packages in the provided text"
#           content: 
#            application/json: {}
# 
# ```
# 
# There’s kind of a lot going on there, so let’s break it down piece by piece.
# 
# First, we’re specifying what version of OpenAPI we’re using. I’ve picked 3.0.0 because it’s the most recent major version. You’d probably only want to pick a different version if you were working on a project with an existing specification that was written in an earlier version. 
# 
# ```
# openapi: "3.0.0"
# ```
# 
# Next we’ve got information on our specific API .I’ve given it a name, a short description and a version number. Since this is the first version of my app, I’m calling it 1.0. If I make major changes to my API in the future, I’ll want to update the specification and create a new version. 
# 
# ```
# info:
#   title: "Package name extractor"
#   description: "API that accepts a text string and returns packages names in it."
#   version: "1.0"
# ```
# 
# Finally, we have the meat of the specificion; our methods. Here we have only one method. It’s at the URL [whatever my app’s URL is]/extractpackages. 
# 
# It’s a POST method that extracts package names. We pass in a JSON file and, if everything goes well, get back a JSON file. This is the same information that we provided in the answer to the questions in the previous section, just in a machine-readable format.
# 
# > The information that the client passes to the server is called a “request”  and the information that the server passes back to the client is called a “response”. 
# 
# The “200” is a HTTP response status code. "200" in partiuclar means that the request was accepted and everything’s ok. If we get any other response status code, then our API will return nothing. (The server will probably return send the usual error code it generates for a specififc error, though, like "404" if the server can't find what the client requests.)
# 
# ```
# paths:
#   /extractpackages:
#     post:
#       description: "Extract package names"
#       
#       requestBody:
#         description: "Json with single field containing text to extract entities from"
#         required: true
#         content:
#           application/json: {}
#       
#       responses:
#         '200':
#           description: "Returns names & indexs of packages in the provided text"
#           content: 
#            application/json: {}
# ```
# 
# ## Your turn!
# 
# Now it’s time for you to write your own specification. You might find it easier to use the [Swagger editor](http://editor.swagger.io/), which has lots of nice features for designing OpenAPI specifications.
# 
# I'd recommend copying the example I gave you above and editing it so that it describes the API you want to build. If you'd like feedback from other people, feel free to share a link in the comments! (You can download your specification as a YAML file from the Swagger editor and then upload it as a dataset or to a GitHub repo.) 
# 
# Tomorrow we'll use our specifications to create our API by writing a Flask app. 
