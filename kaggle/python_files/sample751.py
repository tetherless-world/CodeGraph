#!/usr/bin/env python
# coding: utf-8

# This notebook shows how to build and run concorde TSP solver directly, without using a rather underfeatured wrapper like pyconcorde.

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import sympy

# ## Build concorde

# *Note: Internet must be enabled in kernel environment's settings for this step.*
# 
# Download concorde's source code and build LINKERN - the main tour finding component (Chained Lin-Kernighan). It comes with a convenient command line utility that we'll build and copy for later use.

# In[ ]:


if ! [[ -f ./linkern ]]; then
  wget http://www.math.uwaterloo.ca/tsp/concorde/downloads/codes/src/co031219.tgz
  echo 'c3650a59c8d57e0a00e81c1288b994a99c5aa03e5d96a314834c2d8f9505c724  co031219.tgz' | sha256sum -c
  tar xf co031219.tgz
  (cd concorde && CFLAGS='-Ofast -march=native -mtune=native -fPIC' ./configure)
  (cd concorde/LINKERN && make -j && cp linkern ../../)
  rm -rf concorde co031219.tgz
fi

# The rest of concorde code is mostly dedicated to optimizing a lower bound on TSP length and is not of great practical interest for this competition, so we won't built it.

# ## Prepare input

# In[ ]:


def read_cities(filename='../input/cities.csv'):
    return pd.read_csv(filename, index_col=['CityId'])

cities = read_cities()

# Concorde's EUC_2D norm rounds the distances between cities to the nearest integer ([source](https://github.com/matthelb/concorde/blob/master/UTIL/edgelen.c#L299)) whereas competition metric doesn't. This significantly hurts quality as we get closer to TSP optimum. Let's scale the coordinates up by a few orders of magnitude to work around this problem:

# In[ ]:


cities1k = cities * 1000

# Write out the problem in TSPLIB format:

# In[ ]:


def write_tsp(cities, filename, name='traveling-santa-2018-prime-paths'):
    with open(filename, 'w') as f:
        f.write('NAME : %s\n' % name)
        f.write('COMMENT : %s\n' % name)
        f.write('TYPE : TSP\n')
        f.write('DIMENSION : %d\n' % len(cities))
        f.write('EDGE_WEIGHT_TYPE : EUC_2D\n')
        f.write('NODE_COORD_SECTION\n')
        for row in cities.itertuples():
            f.write('%d %.11f %.11f\n' % (row.Index+1, row.X, row.Y))
        f.write('EOF\n')

write_tsp(cities1k, 'cities1k.tsp')

# ## Run LINKERN

# Flags that we're using:
# 
#   * `-s <seed>` - random seed
#   * `-S <file>` - save the best found tour periodically in this file
#   * `-R 999999999` - repeat LK rounds (almost) indefinitely
#   * `-t <seconds>` - bound by time instead
#   * `-K 1` - a minor tuning: set kick type to "geometric" instead of "random walk"

# In[ ]:


time ./linkern -K 1 -s 42 -S linkern.tour -R 999999999 -t 300 ./cities1k.tsp >linkern.log

# Lengths of best found tours (times scaling factor) during each LK round:

# In[ ]:


pd.read_csv('linkern.csv', index_col=0, names=['TSP tour length']).plot();

# ## Score and submit found tour

# A small OOP wrapper to represent tours:

# In[ ]:


class Tour:
    cities = read_cities()
    coords = (cities.X + 1j * cities.Y).values
    penalized = ~cities.index.isin(sympy.primerange(0, len(cities)))

    def __init__(self, data):
        """Initializes from a list/iterable of indexes or a filename of tour in csv/tsplib/linkern format."""

        if type(data) is str:
            data = self._read(data)
        elif type(data) is not np.ndarray or data.dtype != np.int32:
            data = np.array(data, dtype=np.int32)
        self.data = data

        if (self.data[0] != 0 or self.data[-1] != 0 or len(self.data) != len(self.cities) + 1):
            raise Exception('Invalid tour')

    @classmethod
    def _read(cls, filename):
        data = open(filename, 'r').read()
        if data.startswith('Path'):  # csv
            return pd.read_csv(io.StringIO(data)).Path.values
        offs = data.find('TOUR_SECTION\n')
        if offs != -1:  # TSPLIB/LKH
            data = np.fromstring(data[offs+13:], sep='\n', dtype=np.int32)
            data[-1] = 1
            return data - 1
        else:  # linkern
            data = data.replace('\n', ' ')
            data = np.fromstring(data, sep=' ', dtype=np.int32)
            if len(data) != data[0] + 1:
                raise Exception('Unrecognized format in %s' % filename)
            return np.concatenate((data[1:], [0]))

    def info(self):
        dist = np.abs(np.diff(self.coords[self.data]))
        penalty = 0.1 * np.sum(dist[9::10] * self.penalized[self.data[9:-1:10]])
        dist = np.sum(dist)
        return { 'score': dist + penalty, 'dist': dist, 'penalty': penalty }

    def dist(self):
        return self.info()['dist']

    def score(self):
        return self.info()['score']

    def __repr__(self):
        return 'Tour: %s' % str(self.info())

    def to_csv(self, filename):
        pd.DataFrame({'Path': self.data}).to_csv(filename, index=False)

# Read found TSP tour and score it:

# In[ ]:


tour = Tour('linkern.tour')
tour

# Format it for submission:

# In[ ]:


tour.to_csv('submission.csv')

# ## Plot the tour

# With rainbow gradient to vizualize the progression of the tour:

# In[ ]:


def plot_tour(tour, cmap=mpl.cm.gist_rainbow, figsize=(25, 20)):
    fig, ax = plt.subplots(figsize=figsize)
    n = len(tour.data)

    for i in range(201):
        ind = tour.data[n//200*i:min(n, n//200*(i+1)+1)]
        ax.plot(tour.cities.X[ind], tour.cities.Y[ind], color=cmap(i/200.0), linewidth=1)

    ax.plot(tour.cities.X[0], tour.cities.Y[0], marker='*', markersize=15, markerfacecolor='k')
    ax.autoscale(tight=True)
    mpl.colorbar.ColorbarBase(ax=fig.add_axes([0.125, 0.075, 0.775, 0.01]),
                              norm=mpl.colors.Normalize(vmin=0, vmax=n),
                              cmap=cmap, orientation='horizontal')

plot_tour(tour)

# ## Changelog

# * V1: initial version (1518555.37)
#   * [5 hour run](https://www.kaggle.com/blacksix/concorde-for-5-hours): 1516912.37
# * V2: improved plotting (1518523.84)
# * V3: geometric kick (-K 1), improved tour wrapper, scorer and plotting  (1518375.06)
# * V4: changelog, comments, compile with -Ofast (1518296.59)
# * V5: synced library code to a version from my latest [Flip 'n Roll](https://www.kaggle.com/blacksix/flip-n-roll-fast-python-scorer) kernel, fixed a minor bug in plotting
