#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
import pandas as pd
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

# In[ ]:


For the Traveling Santa 2018 competition. All computation is done
within the Kernel, including building the list of primes and creating the TSPLIB file.
The downloaded C codes (in gzipped tar files) are left in the Kernel.

Note: The prime_thread code, built on top of the linkern code from Concorde, uses as a
random seed the current time. The behavior of the Kernel is random.

# In[ ]:


cities = pd.read_csv('../input/cities.csv', index_col=['CityId'], nrows=None)
cities_1000 = cities * 1000

# In[ ]:


def write_tsp(nodes, filename, name='Santa Prime Paths'):
    # From https://www.kaggle.com/blacksix/concorde-for-5-hours.
    with open(filename, 'w') as f:
        f.write('NAME : %s\n' % name)
        f.write('COMMENT : %s\n' % name)
        f.write('TYPE : TSP\n')
        f.write('DIMENSION : %d\n' % len(nodes))
        f.write('EDGE_WEIGHT_TYPE : EUC_2D\n')
        f.write('NODE_COORD_SECTION\n')
        for row in nodes.itertuples():
            f.write('%d %.11f %.11f\n' % (row.Index + 1, row.X, row.Y))
        f.write('EOF\n')

write_tsp(cities_1000, 'santa197769.tsp')

# In[ ]:


rm -r LKH-2.0.9
rm LKH-2.0.9.t*
wget http://akira.ruc.dk/~keld/research/LKH/LKH-2.0.9.tgz

# In[ ]:


tar xzvf LKH-2.0.9.tgz
cd LKH-2.0.9
make
mv LKH ..
cd ..
rm -r LKH-2.0.9

# In[ ]:


ls

# In[ ]:


def write_parameters(parameters, filename):
    # From https://www.kaggle.com/jsaguiar/lkh-solver
    with open(filename, 'w') as f:
        for param, value in parameters:
            f.write("{} = {}\n".format(param, value))
    print("Parameters saved as", filename)

parameters0 = [
    ("PROBLEM_FILE", "santa197769.tsp"),
    ("TOUR_FILE", "santa.new0.tour"),
    ('CANDIDATE_SET_TYPE', 'POPMUSIC'),
    ('INITIAL_PERIOD', 100),
    ('INITIAL_TOUR_ALGORITHM', 'GREEDY'),
    ('MAX_TRIALS', 20),
    ('MOVE_TYPE', 5),
    ('PATCHING_C', 5),
    ('PATCHING_A', 1),
    ('RECOMBINATION', 'GPX2'),
    ('RUNS',1)
]
parameters7 = [
    ("PROBLEM_FILE", "santa197769.tsp"),
    ("TOUR_FILE", "santa.rohenew.tour"),
    ('CANDIDATE_SET_TYPE', 'POPMUSIC'),
    ('INITIAL_PERIOD', 100),
    ('INITIAL_TOUR_ALGORITHM', 'GREEDY'),
    ('SUBPROBLEM_TOUR_FILE', 'santa.new0.tour'),
    ('SUBPROBLEM_SIZE', '50000 ROHE BORDERS COMPRESSED'),
    ('MAX_TRIALS', 100),
    ('MOVE_TYPE', 5),
    ('PATCHING_C', 5),
    ('PATCHING_A', 1),
    ('RECOMBINATION', 'GPX2'),
    ('RUNS',1)
]
write_parameters(parameters0, "par0.par")
write_parameters(parameters7, "par7.par")

# In[ ]:


./LKH par0.par

# In[ ]:


./LKH par7.par

# In[ ]:


def write_xy(nodes, filename):
    with open(filename, 'w') as f:
        f.write('%d\n' % len(nodes))
        for row in nodes.itertuples():
            f.write('%.12f %.12f\n' % (row.X, row.Y))
        f.write('EOF\n')

write_xy(cities, 'kaggle.xy')

# In[ ]:


wget http://www.math.uwaterloo.ca/tsp/pm/gen_primes.c
gcc -o gen_primes gen_primes.c -lm
./gen_primes
head primes.txt
rm gen_primes*

# In[ ]:


wget http://www.math.uwaterloo.ca/tsp/pm/PM_1.tgz
tar xzvf PM_1.tgz
cd PM_1
make prime_thread
mv prime_thread ..
cd ..
rm -r PM_1

# In[ ]:


ls

# In[ ]:


./prime_thread -P primes.txt -Z santa.rohenew.tour -o kick.tour -t7200.0 kaggle.xy

# In[ ]:


rm -r PM-LKH*
wget http://www.math.uwaterloo.ca/tsp/pm/PM-LKH-3b.tgz
tar xzvf PM-LKH-3b.tgz
mv PM-LKH-3b PM-LKH
cd PM-LKH
make
rm -r PMSRC_DIV

# In[ ]:


cp kaggle.xy PM-LKH/
cp santa197769.tsp PM-LKH/
cp primes.txt PM-LKH/primes_list

# In[ ]:


cd PM-LKH
./run_Segment_Optimization ../kick.tour 10000 post.tour

# In[ ]:


cd PM-LKH
mv submission.csv ../submission.csv
rm -r DIV
rm -r DIV_TOURS

# In[ ]:


cd PM-LKH
./run_Segment_Optimization post.tour 5000 post2.tour

# In[ ]:


cd PM-LKH
mv submission.csv ../submission.csv
rm -r DIV
rm -r DIV_TOURS

# In[ ]:


cd PM-LKH
./run_Segment_Optimization post2.tour 7500 post3.tour

# In[ ]:


cd PM-LKH
mv submission.csv ../submission.csv
rm -r DIV
rm -r DIV_TOURS

# In[ ]:


cd PM-LKH
./run_Segment_Optimization post3.tour 6250 post4.tour

# In[ ]:


cd PM-LKH
mv submission.csv ../submission.csv
mv post4.tour ../post4.tour
rm -r DIV
rm -r DIV_TOURS


# In[ ]:


Final run of the linkern-based code to search for global impprovemnts.

# In[ ]:


./prime_thread -P primes.txt -Z post4.tour -t3000.0 kaggle.xy
