#!/usr/bin/env python
# coding: utf-8

# In[ ]:




# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import os
print(os.listdir("../input"))

# First Step: Define PATH e read the .csv file

# In[3]:


PATH = '../input/Pokemon.csv'

df_poke = pd.read_csv(PATH)

# A first visualization at our Dataset

# In[4]:


df_poke.head()

# In[5]:


df_poke.describe(include='all')

# Let's create a dict for coloring our charts. The colors were scrapped from bulbapedia

# In[6]:


pokecolor = {'Grass': '#78C850', 'Normal':'#A8A878','Fire':'#F08030','Fighting':'#C03028','Water':'#6890F0',
            'Flying':'#A890F0','Poison':'#A040A0','Electric':'#F8D030','Ground':'#E0C068','Psychic':'#F85888',
            'Rock':'#B8A038','Ice':'#98D8D8','Bug':'#A8B820','Dragon':'#7038F8','Ghost':'#705898','Dark':'#705848',
            'Steel':'#B8B8D0','Fairy':'#EE99AC'}



# Let's see how many Pokémon of each type exists in this dataset!

# In[7]:


tipo1 = df_poke['Type 1'].value_counts(ascending=True)
tipo1.plot(kind='barh', grid=True, figsize=(15,8),
          xticks= [n for n in range(0,111, 10)],
          color=tipo1.index.map(pokecolor))
plt.xlabel('Número de Pokémons')
plt.ylabel('Tipo Primario')
plt.title('Pokémons agrupados por tipo primario')
plt.minorticks_on()


# Agora vamos juntar os tipos primários com os secundários e vermos quantos existem de cada combinação!

# In[13]:


df_poke['Type1+2'] = df_poke['Type 1'] +" - " +  df_poke['Type 2']
len(df_poke['Type1+2'].value_counts())

# Wow! There are so many combinations! Then let's visualize only pokémon who actually have Type 2?

# In[ ]:


mask1 = pd.isna(df_poke['Type 2'])
pokecomb = df_poke[mask1 == False]['Type 1'].value_counts(ascending=True)
pokecomb.plot(kind='barh', title='Pokémons that have a combination',
              figsize=(15,8), xticks = [n for n in range(0,56, 5)],
              grid=True, color=pokecomb.index.map(pokecolor))
plt.xlabel('Pokémon Number')
plt.ylabel('Primary Type')

# And to count only Legendary Pokémon...

# In[ ]:


lendmask = df_poke['Legendary'] == True
pokelend = df_poke[lendmask]['Type 1'].value_counts(ascending=True)
pokelend.plot(kind='barh',title='Legendaries by Type',
              figsize=(15,8), grid=True, color = pokelend.index.map(pokecolor))
plt.xlabel('Legendary Number')
plt.ylabel('Primary Type')

# Let's see what types are the strongest by the mean of it's Total stats!

# In[ ]:


status_totais = df_poke.groupby('Type 1').Total.mean().sort_values()
status_totais.plot(kind='barh',title='Primary Type Mean Total Stats', color=status_totais.index.map(pokecolor),
                   xticks=[n for n in range(0,601,50)],figsize=(15,8), grid=True)

plt.xlabel('Total Stats')
plt.ylabel('Primary Type')

# Dragon Type Pokémon has the strongest status! Could this be caused by the great number of Dragon Type Legendary Pokémon?!

# In[ ]:


status_totais_nolend = df_poke[lendmask == False].groupby('Type 1').Total.mean().sort_values(ascending=True)
status_totais_nolend.plot(kind='barh',title='Primary Type Mean Total Stats (No Legendaries)',
      xticks= [n for n in range(0,551,50)],figsize=(15,8), grid=True, color = status_totais_nolend.index.map(pokecolor))
plt.xlabel('Total Stats')
plt.ylabel('Primary Type')

# What about Attack now?

# In[ ]:


poke_atk = df_poke.groupby('Type 1').Attack.mean().sort_values()
poke_atk.plot(kind='barh', title='Primary Type Mean Attack', xticks = [n for n in range(0,121, 10)],
              figsize=(15,8), grid=True, color = poke_atk.index.map(pokecolor))
plt.xlabel('Attack')
plt.ylabel('Primary Type')

# Dragons wins again. Let's check without legendary Pokémon!

# In[ ]:


poke_atk_nolend = df_poke[lendmask == False].groupby('Type 1').Attack.mean().sort_values()
poke_atk_nolend.plot(kind='barh', xticks = [n for n in range(0,111,10)],title='Primary Type Mean Attack (No Legendaries)',figsize=(15,8),
                                            color=poke_atk_nolend.index.map(pokecolor),grid=True)
plt.xlabel('Attack')
plt.ylabel('Primary Type')


# Dragons still win! Now let's see the Defense Stat.

# In[ ]:


poke_def = df_poke.groupby('Type 1').Defense.mean().sort_values()
poke_def.plot(kind='barh', title='Primary Type Mean Defense', xticks = [n for n in range(0,131, 10)],figsize=(15,8),
              color = poke_def.index.map(pokecolor),grid=True)
plt.xlabel('Defense')
plt.ylabel('Primary Type')

# Steel wins by far! Let's see if it's still this way without Legendary Pokémon?

# In[ ]:


poke_def = df_poke[lendmask == False].groupby('Type 1').Defense.mean().sort_values()
poke_def.plot(kind='barh', title='Primary Type Mean Defense (No Legendaries)',xticks = [n for n in range(0,131,10)],
              figsize=(15,8), grid=True, color=poke_def.index.map(pokecolor))
plt.xlabel('Defense')
plt.ylabel('Primary Type')

# Steel still wins! What makes sense seeing how many Steel Type Legendaries are in this dataset.

# In[ ]:


poke_hp = df_poke.groupby('Type 1').HP.mean().sort_values()
poke_hp.plot(kind='barh',title='Primary Type Mean HP',xticks = [n for n in range(0,91,10)],
             figsize=(15,8),grid=True, color=poke_hp.index.map(pokecolor))

plt.xlabel('HP')
plt.ylabel('Primary Type')
And Dragon Type wins again! Now without Legendary Pokémon!
# In[ ]:


poke_hp_nolend = df_poke[lendmask == False].groupby('Type 1').HP.mean().sort_values()
poke_hp_nolend.plot(kind='barh',title='Primary Type Mean HP (No Legendaries)',xticks = [n for n in range(0,86, 10)],
                    figsize=(15,8), grid=True, color = poke_hp_nolend.index.map(pokecolor))
plt.xlabel('HP')
plt.ylabel('Primary Type')

# Now we see a significant change! We can clearly see that Legendary Pokémon was heavily influencing on the Dragon Type HP Mean.
# Let's check it's Special Defense now!

# In[ ]:


poke_spdef = df_poke.groupby('Type 1')['Sp. Def'].mean().sort_values()
poke_spdef.plot(kind='barh', title='Primary Type Mean Special Defense',xticks = [n for n in range(0,91,10)],
                figsize=(15,8), grid=True, color=poke_spdef.index.map(pokecolor))
plt.xlabel('Special Defense')
plt.ylabel('Primary Type')

# Again without Legendary!

# In[ ]:


poke_spdef_nolend = df_poke[lendmask == False].groupby('Type 1')['Sp. Def'].mean().sort_values()
poke_spdef_nolend.plot(kind='barh', title='Primary Type Mean Special Defense (No Legendaries)',xticks = [n for n in range(0,91,10)],
                figsize=(15,8), grid=True, color=poke_spdef_nolend.index.map(pokecolor))
plt.xlabel('Special Defense')
plt.ylabel('Primary Type')

# Wow, we can see significant changes in the top 3! Dragons got from first to third place.
# Let's analyze Special Attack now.

# In[ ]:


poke_spatk = df_poke.groupby('Type 1')['Sp. Atk'].mean().sort_values()
poke_spatk.plot(kind='barh',title='Primary Type Mean Special Attack', xticks=[n for n in range(0,101,10)],
               figsize=(15,8), grid=True, color=poke_spatk.index.map(pokecolor))
plt.xlabel('Special Attack')
plt.ylabel('Primary Type')

# Without Legendaries, as usual!

# In[ ]:


poke_spatk_nolend = df_poke[lendmask == False].groupby('Type 1')['Sp. Atk'].mean().sort_values()
poke_spatk_nolend.plot(kind='barh',title='Primary Type Mean Special Attack (No Legendaries)', xticks=[n for n in range(0,101,10)],
               figsize=(15,8), grid=True, color=poke_spatk_nolend.index.map(pokecolor))
plt.xlabel('Special Attack')
plt.ylabel('Primary Type')

# Now the Speed!

# In[ ]:


poke_spd= df_poke.groupby('Type 1')['Speed'].mean().sort_values()
poke_spd.plot(kind='barh',title='Primary Type Mean Speed', xticks=[n for n in range(0,101,10)],
               figsize=(15,8), grid=True, color=poke_spd.index.map(pokecolor))
plt.xlabel('Speed')
plt.ylabel('Primary Type')

# Without Legendary Pokémon:

# In[ ]:


poke_spd= df_poke[lendmask == False].groupby('Type 1')['Speed'].mean().sort_values()
poke_spd.plot(kind='barh',title='Primary Type Mean Speed (No Legendaries)', xticks=[n for n in range(0,101,10)],
               figsize=(15,8), grid=True, color=poke_spd.index.map(pokecolor))
plt.xlabel('Speed')
plt.ylabel('Primary Type')

# Wow, we have only 2 Flying Type Legendaries and it changes so much in the mean!
# 

#  

# It is somehow finished, but i would love to see suggestions for more plotting, if i should use faceting, subplotting, seaborn plots and other things like these. It i my first kernel and i would be very grateful if you would give me your feedback in it. Thanks!
