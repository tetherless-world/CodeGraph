#!/usr/bin/env python
# coding: utf-8

# ____
# 
# * **Day 1**: Determining what information should be monitored with a dashboard. [Notebook](https://www.kaggle.com/rtatman/dashboarding-with-notebooks-day-1), [Livestream Recording](https://www.youtube.com/watch?v=QO2ihJS2QLM)
# * **Day 2**: How to create effective dashboards in notebooks, [Python Notebook](https://www.kaggle.com/rtatman/dashboarding-with-notebooks-day-2-python), [R Notebook](https://www.kaggle.com/rtatman/dashboarding-with-notebooks-day-2-r), [Livestream](https://www.youtube.com/watch?v=rhi_nexCUMI)
# * **Day 3**: Running notebooks with the Kaggle API, [Notebook](https://www.kaggle.com/rtatman/dashboarding-with-notebooks-day-3), [Livestream](https://youtu.be/cdEUEe2scNo)
# * **Day 4**: Scheduling notebook runs using cloud services, [Notebook](https://www.kaggle.com/rtatman/dashboarding-with-notebooks-day-4), [Livestream](https://youtu.be/Oujj6nT7etY)
# * **Day 5**: Testing and validation, [Python Notebook](https://www.kaggle.com/rtatman/dashboarding-with-notebooks-day-5), [R Notebook](https://www.kaggle.com/rtatman/dashboarding-with-notebooks-day-5-r), [Livestream](https://www.youtube.com/watch?v=H6DcpIykT8E)
# 
# ____
# 
# 
# Welcome to the first day of Dashboarding with scheduled notebooks. Today we're going to do two things:
# 
# * Pick a dataset to work with
# * Figure out what data we should include in our dashboard
# 
# Today's timeline: 
# 
# * **5 minutes:** Read notebook
# * **5 minutes:** Pick dataset and read over the documentation, determining what the most important information should be
# * **5 minutes:** Start kernel and read in data
# * **5 minutes:** Create one or more visualizations (no need to worry about pretty; quick and dirty will work!)
# 
# # Picking a dataset
# 
# Not every dataset needs to be dashboarded. Dashboards are useful because they make it easy to monitor things that change over time, which means it only makes sense to use datasets that are updated; there's usually no reason to go to all the trouble of building a dashboard for a static dataset when a plain notebook or markdown document will do just as well. 
# 
# The method we're going to be using--scheduling our notebooks rather than continuously updating them--works best for datasets that are batch processed. 
# 
# > **Batch data processing** refers to data processing that happens at a single point in time, usually by running a script. It's opposed to **streaming data processing** which happens continuously. 
# 
# I've put together a list of Kaggle datasets that are batch processed and updated daily for you here. It’s mostly public data that’s provided by cities in the US, but Kaggle’s own public data, Meta Kaggle, is also updated daily. Pick one that you like and create a new Kernel using it as a data source. 
# 
# * [Meta Kaggle](https://www.kaggle.com/kaggle/meta-kaggle)
# * [Procurement Notices](https://www.kaggle.com/theworldbank/procurement-notices)
# * [Chicago Red Light and Speed Camera Data](https://www.kaggle.com/chicago/chicago-red-light-and-speed-camera-data)
# * [Chicago 311 Service Requests](https://www.kaggle.com/chicago/chicago-311-service-requests)
# * [Seattle Road Weather Information Stations](https://www.kaggle.com/city-of-seattle/seattle-road-weather-information-stations)
# * [Seattle Use of Force](https://www.kaggle.com/city-of-seattle/seattle-use-of-force)
# * [Seattle Crisis Data](https://www.kaggle.com/city-of-seattle/seattle-crisis-data)
# * [Los Angeles Parking Citations](https://www.kaggle.com/cityofLA/los-angeles-parking-citations)
# * [What's Happening LA Calendar Dataset](https://www.kaggle.com/cityofLA/what's-happening-la-calendar-dataset)
# * [Oakland Call Center & Public Work Service Requests](https://www.kaggle.com/cityofoakland/oakland-call-center-public-work-service-requests)
# * [NY Bus Breakdown and Delays](https://www.kaggle.com/new-york-city/ny-bus-breakdown-and-delays)
# * [NYPD Motor Vehicle Collisions](https://www.kaggle.com/new-york-city/nypd-motor-vehicle-collisions)
# * [NY Daily Inmates In Custody](https://www.kaggle.com/new-york-city/ny-daily-inmates-in-custody)
# * [NYS Turnstile Usage Data](https://www.kaggle.com/new-york-state/nys-turnstile-usage-data)
# * [NOAA Global Surface Summary of the Day](https://www.kaggle.com/noaa/noaa-global-surface-summary-of-the-day/)
# * [SF Fire Data (Incidents, Violations, and more)](https://www.kaggle.com/san-francisco/sf-fire-data-incidents-violations-and-more)
# * [SF Restaurant Scores - LIVES Standard](https://www.kaggle.com/san-francisco/sf-restaurant-scores-lives-standard)
# 
# # Figure out what data should be dashboarded
# 
# Because we're picking public datasets rather than working from one we've been given by our co-workers, we unfortunately can't use the most effective technique to figure out what information to include: asking whoever gave you the data. 
# 
# > The easiest way to figure out what to include in a dashboard is to ask stakeholders (other people that care about what's in your data and you would want to use the dashboard) what they'd consider the most important information.
# 
# Failing that, there are some general guidelines you can use to figure out what information to include in a dashboard. 
# 
# * *What information is changing relatively quickly (every day or hour)?* Information that only changes every quarter or year probably belong in a report, not a dashboard. 
# * *What information is the most important to your mission?* If you're a company, things like money or users are probably going to be pretty important, but if you're a school district you probably care more about things like attendance or grades.
# * *What will affect the choices you or others will need to make?* Are you running A/B tests and need to choose which model to keep in production based on them? Then it's probably important that you track your metrics and other things that might affect those metrics, like sales that are running at the same time. Is there some outside factor that might affect your business, like the weather forecast next week? Then it might make sense to pull in another dataset and show that as well.
# * *What changes have you made?* If you're tuning parameters or adjusting teaching schedules, you want to track the fact that you've made those changes and also how they've affected outcomes.
# 
# # Your turn!
# 
# Pick a dataset that's updated daily by Kaggle from this list. Imagine you work for the organization that produced it and identify factors in the dataset that might represent:
# 
# * The goals of your organization (like users or measures of pollution)
# * Things that you (or your colleagues) can change to affect those goals (like advertising spending or the number of factory inspections)
# * Thing you can't change but that will affect the outcome (like the school year, or weather conditions)
# 
# You might not find all three in the same dataset, but you should be able to pinpoint at least one. I'd recommend using the summary statistics in the Data tab of the dataset or reading the documentation in the Overview tab to help identify them.
# 
# Then start a kernel on that dataset ([this video has a quick walk-through if you need a quick refresher on how to do this](https://youtu.be/fvF2H85ko9c)) and put together two quick visualizations or summary tables that show two of the factors you've identified in the first step.
# 
# If you like, you can make your kernel public and share a link to it in the comments on this dataset to share with other participants. (And you can take a peek at other people's work to see what they've chosen to look at!) I'll pick a couple that I especially like to highlight as examples. :)
