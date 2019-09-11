#!/usr/bin/env python
# coding: utf-8

# In[129]:


import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as const
import os
import math

# Parameters
ximin=-8
ximax=-ximin
Nsteps=1001
nu_0=4.0
ϵ=-4

ξ_vector=np.linspace(ximin,ximax,Nsteps)
h=ξ_vector[1]-ξ_vector[0]
nu=-nu_0*np.exp(-ξ_vector**2)

#choosing arbritrary point for evaluating the energy value
def findOddEnergy(ϵ_min,ϵ_max):
    
    while(abs(ϵ_min-ϵ_max)>1e-8):
        
        ϵ_avg=(ϵ_min+ϵ_max)/2
        k=math.sqrt(-ϵ_avg)
        ϕ=[]
        ϕ.append(1)
        ϕ.append(np.exp(k*h))

        for i in range(2,Nsteps):
            ϕ.append(((2+h**2*(nu[i]-ϵ_avg))*ϕ[i-1])-ϕ[i-2])

        ϕ_950=(((2+h**2*(nu[Nsteps-50]-ϵ_avg))*ϕ[Nsteps-50-1])-ϕ[Nsteps-50-2])

        if(ϕ_950)<0:
            ϵ_max=ϵ_avg
        else:
            ϵ_min=ϵ_avg
            
    plt.plot(ϕ, '-')
    plt.show()

    return ϵ_avg;

def findEvenEnergy(ϵ_min,ϵ_max):
    
    while(abs(ϵ_min-ϵ_max)>1e-10):
        
        ϵ_avg=(ϵ_min+ϵ_max)/2
        k=math.sqrt(-ϵ_avg)
        ϕ=[]
        ϕ.append(1)
        ϕ.append(np.exp(k*h))

        for i in range(2,Nsteps):
            ϕ.append(((2+h**2*(nu[i]-ϵ_avg))*ϕ[i-1])-ϕ[i-2])

        ϕ_950=(((2+h**2*(nu[Nsteps-50]-ϵ_avg))*ϕ[Nsteps-50-1])-ϕ[Nsteps-50-2])

        if((ϕ_950)<0):
            ϵ_min=ϵ_avg
        else:
            ϵ_max=ϵ_avg
        
    plt.plot(ϕ, '-')
    plt.show()

    return ϵ_avg;

groundStateEnergy = findOddEnergy(-2.5,-2)
print(groundStateEnergy)

firstExcitedStateEnergy = findEvenEnergy(groundStateEnergy,0)
print(firstExcitedStateEnergy)

secondExcitedStateEnergy = findOddEnergy(firstExcitedStateEnergy,0)
print(secondExcitedStateEnergy)


# 1. The ground state wave function has 0 nodes
# 2. The first excited state has 1 node
# 3.

# In[ ]:




# 

# In[214]:


# Parameters
ximin=-16
ximax=-ximin
Nsteps=2001
nu_0=2
ϵ=-4
ξ_0=12

ξ_vector=np.linspace(ximin,ximax,Nsteps)
h=ξ_vector[1]-ξ_vector[0]
nu=-nu_0*(np.exp(-(ξ_vector-ξ_0)**2)+np.exp(-(ξ_vector+ξ_0)**2))

plt.plot(nu, '-')
plt.show()

groundStateEnergy = findOddEnergy(-2,-.799500)
print(groundStateEnergy)
excitedStateEnergy = findEvenEnergy(groundStateEnergy,0)
print(excitedStateEnergy)
secondExcitedStateEnergy = findOddEnergy(excitedStateEnergy,0)
print(secondExcitedStateEnergy)

# In[215]:


# Parameters
ximin=-16
ximax=-ximin
Nsteps=2001
nu_0=2
ϵ=-4
ξ_0=1.5

ξ_vector=np.linspace(ximin,ximax,Nsteps)
h=ξ_vector[1]-ξ_vector[0]
nu=-nu_0*(np.exp(-(ξ_vector-ξ_0)**2)+np.exp(-(ξ_vector+ξ_0)**2))

nu=-nu_0*(np.exp(-(ξ_vector-ξ_0)**2)+np.exp(-(ξ_vector+ξ_0)**2))

plt.plot(nu, '-')
plt.show()

groundStateEnergy = findOddEnergy(-2,-1)
print(groundStateEnergy)
excitedStateEnergy = findEvenEnergy(groundStateEnergy,0)
print(excitedStateEnergy)
secondExcitedStateEnergy = findOddEnergy(excitedStateEnergy,0)
print(secondExcitedStateEnergy)

# In[218]:


# Parameters
ximin=-16
ximax=-ximin
Nsteps=2001
nu_0=4
ϵ=-4
ξ_0=1.1

ξ_vector=np.linspace(ximin,ximax,Nsteps)
h=ξ_vector[1]-ξ_vector[0]
nu=-nu_0*(np.exp(-(ξ_vector-ξ_0)**2)+np.exp(-(ξ_vector+ξ_0)**2))

plt.plot(nu, '-')
plt.show()

groundStateEnergy = findOddEnergy(-2,0)
print(groundStateEnergy)
excitedStateEnergy = findEvenEnergy(groundStateEnergy,0)
print(excitedStateEnergy)


# In[219]:


ξ_0=1.3

ξ_vector=np.linspace(ximin,ximax,Nsteps)
h=ξ_vector[1]-ξ_vector[0]
nu=-nu_0*(np.exp(-(ξ_vector-ξ_0)**2)+np.exp(-(ξ_vector+ξ_0)**2))

plt.plot(nu, '-')
plt.show()

groundStateEnergy = findOddEnergy(-2,0)
print(groundStateEnergy)
excitedStateEnergy = findEvenEnergy(groundStateEnergy,0)
print(excitedStateEnergy)


# In[220]:


ξ_0=1.5

ξ_vector=np.linspace(ximin,ximax,Nsteps)
h=ξ_vector[1]-ξ_vector[0]
nu=-nu_0*(np.exp(-(ξ_vector-ξ_0)**2)+np.exp(-(ξ_vector+ξ_0)**2))

plt.plot(nu, '-')
plt.show()

groundStateEnergy = findOddEnergy(-2,0)
print(groundStateEnergy)
excitedStateEnergy = findEvenEnergy(groundStateEnergy,0)
print(excitedStateEnergy)


# In[ ]:



