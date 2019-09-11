#!/usr/bin/env python
# coding: utf-8

# <img src="http://www.science4all.org/wp-content/uploads/2013/12/Santas-Vehicle-Routing-Problem3.png" alt="drawing" width="600"/>
# 
# This solution uses [Google OR-Tools](https://developers.google.com/optimization/) to optimize the routing for Santa. OR-Tools include a "vehicle routing library" for solving TSPs and a convenient Python interface. Lucky for us, it's all included in Kaggle kernels!
# 
# The start point for the route comes from model-based clustering. You can see from the city plot how the density of points varies quite a bit across the grid. The right clustering approach may work better than starting with a grid or k-means clustering. Also, breaking the cities into clusters can scale to millions of points on a typical machine if the need arises. 
# 
# Here is the overall approach:
# 
# * Divide cities into clusters
# * Get centers for each cluster
# * Find an optimal path across centers
# * Find start and stop points for each cluster
# * Find an optimal segment for each cluster and assemble

# In[ ]:


from tqdm import tqdm
import numpy as np
import pandas as pd
from sympy import sieve
import hvplot.pandas #custom
import colorcet as cc

# ## Clustering 
# 
# #### First Look
# Hexagonal binning is often a good way to quickly see the density of overlapping points. Below is a comparison of all points with a bin plot of prime cities. Prime and non-prime cities appear to be be similarly distributed with a few dark, dense areas.

# In[ ]:


cities = pd.read_csv("../input/cities.csv", index_col=['CityId'])
pnums = list(sieve.primerange(0, cities.shape[0]))
cities['isprime'] = cities.index.isin(pnums)
display(cities.head())

# show all points and density of primes
allpoints = cities.hvplot.scatter('X', 'Y',  width=380, height=350, datashade=True, 
                title='All Cities')
colors = list(reversed(cc.kbc))
primedensity = cities[cities.isprime].hvplot.hexbin(x='X', y='Y', width=420, height=350, 
                cmap=colors, title='Density of Prime Cities').options(size_index='Count', 
                min_scale=0.8, max_scale=0.95)
allpoints + primedensity

# #### Model-based Clustering
# 
# I first tried density-based clustering, but it left too many points that didn't belong to any cluster. In some cases it's ideal to separate out the sparse points, but for TSPs it's not so good. Model-based clustering seems to be a better choice. The number of clusters is set manually. Here I'll choose a relatively high number to ease the burden on the router.

# In[ ]:


from sklearn.mixture import GaussianMixture

mclusterer = GaussianMixture(n_components=350, tol=0.01, random_state=66, verbose=1)
cities['mclust'] = mclusterer.fit_predict(cities[['X', 'Y']].values)
nmax = cities.mclust.max()
print("{} clusters".format(nmax+1))

# Clusters can be of different sizes with model-based clustering. Here's the distribution of cities per cluster and the scatterplot. 

# In[ ]:


histo = cities.hvplot.hist('mclust', ylim=(0,14000), color='tan')

custcolor = cc.rainbow + cc.rainbow
gausses = cities.hvplot.scatter('X', 'Y',  by='mclust', size=5, width=500, height=450, 
                datashade=True, dynspread=True, cmap=custcolor)
display(histo, gausses)

# ## Cluster Centers
# 
# Here are the geometric centers of all the clusters. The big red dot is the North Pole.

# In[ ]:


centers = cities.groupby('mclust')['X', 'Y'].agg('mean').reset_index()
def plot_it(df, dotsize, dotcolor, dotalpha):
    p = df.hvplot.scatter('X', 'Y', size=dotsize, xlim=(0,5100), ylim=(0,3400), width=500,
            height=450, hover_cols=['mclust'], color=dotcolor, alpha=dotalpha)
    return p

cents = plot_it(centers, 30, 'darkblue', 0.5)
npole = plot_it(cities.loc[[0]], 100, 'red', 1)
cents*npole

# ## Optimal Path across Centers
# 
# OR-Tools can find an optimal path across the clusters. Here is a modified version of the code documented on [Google's TSP page](https://developers.google.com/optimization/routing/tsp).

# In[ ]:


#%% imports
from scipy.spatial.distance import pdist, squareform
from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2

#%% functions
def create_mat(df):
    print("building matrix")
    mat = pdist(locations)
    return squareform(mat)

def create_distance_callback(dist_matrix):
    def distance_callback(from_node, to_node):
      return int(dist_matrix[from_node][to_node])
    return distance_callback

status_dict = {0: 'ROUTING_NOT_SOLVED', 
               1: 'ROUTING_SUCCESS', 
               2: 'ROUTING_FAIL',
               3: 'ROUTING_FAIL_TIMEOUT',
               4: 'ROUTING_INVALID'}

def optimize(df, startnode=None, stopnode=None, fixed=False):     
    num_nodes = df.shape[0]
    mat = create_mat(df)
    dist_callback = create_distance_callback(mat)
    search_parameters = pywrapcp.RoutingModel.DefaultSearchParameters()
#     search_parameters.time_limit_ms = int(1000*60*numminutes)
    search_parameters.solution_limit = num_iters 
    search_parameters.first_solution_strategy = (
                                    routing_enums_pb2.FirstSolutionStrategy.SAVINGS)
    search_parameters.local_search_metaheuristic = (
                            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)

    if fixed:
        routemodel = pywrapcp.RoutingModel(num_nodes, 1, [startnode], [stopnode])
    else:
        routemodel = pywrapcp.RoutingModel(num_nodes, 1, startnode)
    routemodel.SetArcCostEvaluatorOfAllVehicles(dist_callback)
    
    print("optimizing {} cities".format(num_nodes)) 
    assignment = routemodel.SolveWithParameters(search_parameters)

    print("status: ", status_dict.get(routemodel.status()))
    print("travel distance: ",  str(assignment.ObjectiveValue()), "\n")
    return routemodel, assignment
    
def get_route(df, startnode, stopnode, fixed): 
    routemodel, assignment = optimize(df, int(startnode), int(stopnode), fixed)
    route_number = 0
    node = routemodel.Start(route_number)
    route = []
    while not routemodel.IsEnd(node):
        route.append(node) 
        node = assignment.Value(routemodel.NextVar(node))
    return route


# In[ ]:


#%% parameters
num_iters=100

# main
nnode = int(cities.loc[0, 'mclust'])
locations = centers[['X', 'Y']].values
segment = get_route(locations, nnode, 0, fixed=False)

# In[ ]:


opoints = centers.loc[segment]
centersline = opoints.hvplot.line('X', 'Y', xlim=(0,5100), ylim=(0,3400), color='green', width=500, 
                            height=450, hover=False) 
gausses*cents*npole*centersline

# The path looks reasonable... 
# 
# ## Start and Stop Points
# The next step is to find good start/stop points for each cluster along the path.
# * Set the start point for cluster[n] as the point closest to the center of cluster[n-1].
# * Set the stop point for cluster[n] as the point closest to the start point of cluster[n+1].
# 
# This idea comes from [Proposed Algorithms to solve Big Data traveling salesman problem](https://www.researchgate.net/publication/326325068_Proposed_Algorithms_to_solve_Big_Data_traveling_salesman_problem). Thank you Research Gate!

# In[ ]:


opoints.reset_index(drop=True, inplace=True) #recall ordered points
cities['clustorder'] = cities.groupby('mclust').cumcount()

# In[ ]:


from sklearn.neighbors import NearestNeighbors

startlist=[0]
neigh = NearestNeighbors(n_neighbors=1, n_jobs=-1)
for i,m in enumerate(opoints.mclust[1:], 0):
    neigh.fit(cities.loc[cities.mclust == m, ['X', 'Y']].values)
    lastcenter = opoints.loc[i, ['X', 'Y']].values.reshape(1, -1)
    closestart = neigh.kneighbors(lastcenter, return_distance=False)
    start = cities.index[(cities.mclust == m) & (cities.clustorder == closestart.item())].values[0]
    startlist.append(start)
opoints['startpt'] = startlist    

# In[ ]:


stoplist = []
for i,m in enumerate(opoints.mclust, 1):
    neigh.fit(cities.loc[cities.mclust == m, ['X', 'Y']].values)
    if m != opoints.mclust.values[-1]:
        nextstartnode = opoints.loc[i, 'startpt']
    else: 
        nextstartnode = 0
    nextstart = cities.loc[nextstartnode, ['X', 'Y']].values.reshape(1, -1)
    closestop = neigh.kneighbors(nextstart, return_distance=False)
    stop = cities.index[(cities.mclust == m) & (cities.clustorder == closestop.item())].values[0]
    stoplist.append(stop)
opoints['stoppt'] = stoplist 

display(cities.head(), opoints.head())

#  Here's what  the new path looks like. The points are all on cluster borders which makes sense.

# In[ ]:


coords = cities.loc[opoints.stoppt, ['X', 'Y', 'mclust']]
stops = plot_it(coords, 30, 'darkblue', 0.5)
stopsline = coords.hvplot.line('X', 'Y', xlim=(0,5100), ylim=(0,3400), color='green', width=500, 
                            height=450, hover=False) 
stopsline*npole*gausses*stops

# ## Optimal Segments for each Cluster
# The next step is to iterate through clusters with the solver. The closed form of the solver with start and stop points is being difficult. For now I'll use start points only and take the hit of doubling back to the next cluster.
# 

# In[ ]:


num_iters = 100
seglist = []
total_clusts = cities.shape[0]
for i,m in enumerate(opoints.mclust):
    district = cities[cities.mclust == m]
    print("begin cluster {}, {} of {}".format(m, i, opoints.shape[0]-1))

    clstart = opoints.loc[i, 'startpt']
    nnode = district.loc[clstart, 'clustorder']
    clstop = opoints.loc[i, 'stoppt']
    pnode = district.loc[clstop, 'clustorder']
    locations = district[['X', 'Y']].values
    
    segnodes = get_route(locations, nnode, pnode, fixed=False) #output is type list
    ord_district =  district.iloc[segnodes]
    segment = ord_district.index.tolist()
    seglist.append(segment)

seglist.append([0])
path = np.concatenate(seglist)

# In[ ]:


bestpath = cities.loc[path, ['X', 'Y']]
clustline = bestpath.hvplot.line('X', 'Y', xlim=(0,5100), ylim=(0,3400), width=500, 
                            height=450, datashade=True, dynspread=True) 
clustline*npole

# In[ ]:


def score_it(path):
    path_df = cities.reindex(path).reset_index()
    path_df['step'] = np.sqrt((path_df.X - path_df.X.shift())**2 + (path_df.Y - path_df.Y.shift())**2)
    path_df['step_adj'] = np.where((path_df.index) % 10 != 0, path_df.step, path_df.step + 
                            path_df.step*0.1*(~path_df.CityId.shift().isin(pnums)))
    return path_df.step_adj.sum()

display(score_it(path))

sub = pd.read_csv('../input/sample_submission.csv')
sub['Path'] = path
sub.to_csv('submission.csv', index=False)
sub.head()

# Not bad, even with the wasted moves although it does take a while to run. It would be interesting to see what improvements come with a fixed-route model, fewer clusters and/or longer run times. 
# 
# P.S. Check out [this discussion](https://www.kaggle.com/c/traveling-santa-2018-prime-paths/discussion/73800) for a solid alternative to creating subproblems (clusters)!

# In[ ]:



