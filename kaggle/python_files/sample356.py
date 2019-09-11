#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style='white')

import os

# ## Load and parse .csv

# In[2]:


grasp3_1 = pd.read_csv("../input/graphics-for-daa/grasp3-output-1.csv", header=None)
grasp3_2 = pd.read_csv("../input/graphics-for-daa/grasp3-output.csv", header=None)

concat_grasp = pd.concat((grasp3_1, grasp3_2)) # Concat grasp csv

grasp = grasp3_1 
grasp[3] = concat_grasp.groupby(concat_grasp.index).mean()[3] # Mean between duplicated indices
grasp = grasp.drop([1], axis=1)
grasp.columns = ["instance", "alpha", "value"]

print(grasp.info())
grasp.head(10)

# # GRASP3 result value for different 'alpha' parameters

# (Hoped for better results....)

# In[3]:


f, ax1 = plt.subplots(figsize=(10, 5))

for instance_name in grasp['instance'].unique():
    filtered = grasp[grasp['instance'] == 'MDPI1_500.txt']
    sns.pointplot(x='alpha', y='value', data=filtered, color='blue', alpha=0.5)
    

plt.grid()

# # Execution time for each algorithm

# In[4]:


exdata = pd.read_csv("../input/execution-data/data.csv", header=None)
exdata.columns = ['instance', 'algorithm', 'value', 'time']

print(exdata.info())
exdata.head(10)

# In[5]:


algorithm_grouped = exdata[exdata['instance'].str.contains(r'MDPI1.*')] # Only instances of MDPI1
algorithm_grouped = algorithm_grouped[algorithm_grouped['algorithm'] != 'Grasp+VND']

plt.figure(figsize=(20, 10))
           
ax = sns.barplot(x='algorithm', y='time', hue='instance', data=algorithm_grouped)

# Labels
ax.set_title('Execution time of each algorithm for each instance', fontsize=25)
ax.set_xlabel('Algorithm', fontsize=20)
ax.set_ylabel('Time', fontsize=20)
ax.set_yticklabels([f'{x} s' for x in ax.get_yticks()])
plt.setp(ax.get_legend().get_texts(), fontsize='15') # for legend text
plt.setp(ax.get_legend().get_title(), fontsize='18'); # for legend title

# ### Execution time of each algorithm for the "MPDI1_1000.txt" instance without GRASP+VND

# In[6]:


filtered_1000_instances = exdata[exdata['instance'].str.contains(r'1000')]
filtered_1000_instances = filtered_1000_instances[filtered_1000_instances['algorithm'] != 'Grasp+VND']
filtered_1000_instances.head(100)

plt.figure(figsize=(20, 10))
           
ax = sns.barplot(x='instance', y='time', hue='algorithm', data=filtered_1000_instances)

# Labels
ax.set_title('Execution time of each algorithm for the "MPDI1_1000.txt" instance (no GRASP+VND)', fontsize=25)
ax.set_xlabel('Instance', fontsize=20)
ax.set_ylabel('Time', fontsize=20)
ax.set_yticklabels([f'{x} s' for x in ax.get_yticks()])
plt.setp(ax.get_legend().get_texts(), fontsize='15') # for legend text
plt.setp(ax.get_legend().get_title(), fontsize='18'); # for legend title

# ### Execution time of each algorithm for the "MPDI1_1000.txt" instance with GRASP+VND

# In[7]:


filtered_1000_instances = exdata[exdata['instance'].str.contains(r'1000')]
filtered_1000_instances.head(100)

plt.figure(figsize=(20, 10))
           
ax = sns.barplot(x='instance', y='time', hue='algorithm', data=filtered_1000_instances)

# Labels
ax.set_title('Execution time of each algorithm for the "MPDI1_1000.txt" instance', fontsize=25)
ax.set_xlabel('Instance', fontsize=20)
ax.set_ylabel('Time', fontsize=20)
ax.set_yticklabels([f'{x} s' for x in ax.get_yticks()])
plt.setp(ax.get_legend().get_texts(), fontsize='15') # for legend text
plt.setp(ax.get_legend().get_title(), fontsize='18'); # for legend title

# ### Execution time for the TabuAdd algorithm (fastest)

# In[8]:


vnd_and_tabuadd = exdata[exdata['algorithm'] == 'TabuAdd']
vnd_and_tabuadd = vnd_and_tabuadd[vnd_and_tabuadd['instance'].str.contains(r'500|750|1000')]

plt.tight_layout()
plt.figure(figsize=(20, 10))
           
ax = sns.barplot(x='instance', y='time', hue='algorithm', data=vnd_and_tabuadd)

# Labels
ax.set_title('Execution time for the TabuAdd algorithm', fontsize=25)
ax.set_xlabel('Instance', fontsize=20)
ax.set_ylabel('Time', fontsize=20)
ax.set_yticklabels([f'{x} s' for x in ax.get_yticks()])
plt.setp(ax.get_legend().get_texts(), fontsize='15') # for legend text
plt.setp(ax.get_legend().get_title(), fontsize='18'); # for legend title

# ### Execution time for the VNS algorithm

# In[9]:


vns = exdata[exdata['algorithm'] == 'VNS']
vns = vns[vns['instance'].str.contains(r'500|750|1000')]

plt.tight_layout()
plt.figure(figsize=(20, 10))
           
ax = sns.barplot(x='instance', y='time', hue='algorithm', data=vns)

# Labels
ax.set_title('Execution time for the VNS algorithm', fontsize=25)
ax.set_xlabel('Instance', fontsize=20)
ax.set_ylabel('Time', fontsize=20)
ax.set_yticklabels([f'{x} s' for x in ax.get_yticks()])
plt.setp(ax.get_legend().get_texts(), fontsize='15') # for legend text
plt.setp(ax.get_legend().get_title(), fontsize='18'); # for legend title

# ### Execution time for the Grasp+VND algorithm

# In[10]:


vnd_and_tabuadd = exdata[exdata['algorithm'] == 'Grasp+VND']
vnd_and_tabuadd = vnd_and_tabuadd[vnd_and_tabuadd['instance'].str.contains(r'500|750|1000')]

plt.tight_layout()
plt.figure(figsize=(20, 10))
           
ax = sns.barplot(x='instance', y='time', hue='algorithm', data=vnd_and_tabuadd)

# Labels
ax.set_title('Execution time for the Grasp+VND algorithm', fontsize=25)
ax.set_xlabel('Instance', fontsize=20)
ax.set_ylabel('Time', fontsize=20)
ax.set_yticklabels([f'{x} s' for x in ax.get_yticks()])
plt.setp(ax.get_legend().get_texts(), fontsize='15') # for legend text
plt.setp(ax.get_legend().get_title(), fontsize='18'); # for legend title

# ### Comparison: VNS vs Grasp+VND

# In[11]:


vns_and_vnd = exdata[(exdata['algorithm'] == 'VNS') | (exdata['algorithm'] == 'Grasp+VND')]
vns_and_vnd = vns_and_vnd[vns_and_vnd['instance'].str.contains(r'500|750|1000')]

plt.tight_layout()
plt.figure(figsize=(20, 10))
           
ax = sns.barplot(x='instance', y='time', hue='algorithm', data=vns_and_vnd)

# Labels
ax.set_title('Execution time between VNS and Grasp+VND', fontsize=25)
ax.set_xlabel('Instance', fontsize=20)
ax.set_ylabel('Time', fontsize=20)
ax.set_yticklabels([f'{x} s' for x in ax.get_yticks()])
plt.setp(ax.get_legend().get_texts(), fontsize='15') # for legend text
plt.setp(ax.get_legend().get_title(), fontsize='18'); # for legend title

# ### MeanSum comparison between VNS and Grasp+VND

# In[12]:


vns_and_vnd = exdata[(exdata['algorithm'] == 'VNS') | (exdata['algorithm'] == 'Grasp+VND')]
vns_and_vnd = vns_and_vnd[vns_and_vnd['instance'].str.contains(r'500|750|1000')]

plt.tight_layout()
plt.figure(figsize=(20, 10))
           
ax = sns.barplot(x='instance', y='value', hue='algorithm', data=vns_and_vnd)

# Labels
ax.set_title('Best MeanSum returned by VNS and Grasp+VND', fontsize=25)
ax.set_xlabel('Instance', fontsize=20)
ax.set_ylabel('MeanSum', fontsize=20)
plt.setp(ax.get_legend().get_texts(), fontsize='15') # for legend text
plt.setp(ax.get_legend().get_title(), fontsize='18'); # for legend title

# ### Neighbour usage comparison in VND

# In[13]:


neibourg_vnd_data = pd.read_csv("../input/neighbour-usage-vnd/datos_sobre_vecinos.csv")

plt.tight_layout()
plt.figure(figsize=(20, 10))
           
ax = sns.barplot(x='instance', y='usages', hue='neighbour', data=neibourg_vnd_data)

# Labels
ax.set_title('Neighbour usage comparison in VND', fontsize=25)
ax.set_xlabel('Instance', fontsize=20)
ax.set_ylabel('MeanSum', fontsize=20)
plt.setp(ax.get_legend().get_texts(), fontsize='15') # for legend text
plt.setp(ax.get_legend().get_title(), fontsize='18'); # for legend title

# ### Best MeanSum returned by each algorithm for big instances

# In[14]:


algorithm_value = exdata[exdata['instance'].str.contains(r'500|750|1000')]

plt.tight_layout()
plt.figure(figsize=(20, 10))
           
ax = sns.barplot(x='instance', y='value', hue='algorithm', data=algorithm_value)

# Labels
ax.set_title('Best MeanSum returned by each algorithm for big instances', fontsize=25)
ax.set_xlabel('Instance', fontsize=20)
ax.set_ylabel('MeanSum', fontsize=20)
plt.setp(ax.get_legend().get_texts(), fontsize='15') # for legend text
plt.setp(ax.get_legend().get_title(), fontsize='18'); # for legend title

# Comparison between the MultiStart and the ParallelMultiStart

# In[15]:


vnd_and_tabuadd = exdata[(exdata['algorithm'] == 'MultiStart') | (exdata['algorithm'] == 'ParallelMulSt')]
vnd_and_tabuadd = vnd_and_tabuadd[vnd_and_tabuadd['instance'].str.contains(r'500|750|1000')]

plt.tight_layout()
plt.figure(figsize=(20, 10))
           
ax = sns.barplot(x='instance', y='time', hue='algorithm', data=vnd_and_tabuadd)

# Labels
ax.set_title('Execution time between MultiStart and Parallel-MultiStart with k=100', fontsize=25)
ax.set_xlabel('Instance', fontsize=20)
ax.set_ylabel('Time', fontsize=20)
ax.set_yticklabels([f'{x} s' for x in ax.get_yticks()])
plt.setp(ax.get_legend().get_texts(), fontsize='15') # for legend text
plt.setp(ax.get_legend().get_title(), fontsize='18'); # for legend title

# # Comparison between our results and the ETSII results

# In[35]:


etsii_results = pd.read_csv("../input/official-results/resultados_etsii.csv", sep=";")
etsii_results.rename(columns={'result': 'value'}, inplace=True)
etsii_results = etsii_results.iloc[2:]
exdata_only_tabu_rasp = exdata[exdata['algorithm'].str.contains(r'TabuAdd|TabuSwap|Grasp$|GRASP3$')]

concat_grasp = pd.concat((exdata_only_tabu_rasp, etsii_results)) # Concat grasp csv

print(etsii_results)

plt.tight_layout()
plt.figure(figsize=(20, 10))

x=['']
ax = sns.barplot(x='instance', y='value', hue='algorithm', data=concat_grasp)

# Labels
ax.set_title('Comparison between our results and the ETSII results', fontsize=25)
ax.set_xlabel('Instance', fontsize=20)
ax.set_ylabel('MeanSum', fontsize=20)
plt.setp(ax.get_legend().get_texts(), fontsize='15') # for legend text
plt.setp(ax.get_legend().get_title(), fontsize='18'); # for legend title
