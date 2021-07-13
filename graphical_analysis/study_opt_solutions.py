#!/usr/bin/env python
# coding: utf-8

# In[72]:


import numpy as np
from numpy import *

# this module of numpy allows to mask certain arrays 
import numpy.ma as ma

from sympy.utilities.iterables import multiset_permutations

import itertools
from itertools import permutations

import copy

import matplotlib.pyplot as plt

from operator import itemgetter
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# to draw tables
from prettytable import PrettyTable 

import os


# # We read the optimal solutions from the 'many11_32' folder and the 'collect_test_solutions.csv' file. All the solutions scored 11.32. 

# In[16]:


solutions = dict()
index = 1


# We read the solutions from the 'many11_32' folder
your_path = 'many11_32'
files = os.listdir(your_path)
for file in files:
    
    solutions[index] = []
    
    if os.path.isfile(os.path.join(your_path, file)):
        f = open(os.path.join(your_path, file),'r')
        for x in f:
            x = x.replace("\n", "")
            solutions[index].append(int(x))
        f.close()
        
    index += 1

# we read the solutions stored in the file 'collect_test_solutions.csv'
f = open('collect_test_solutions.csv').readlines()
for line in f:
    
    for i in range(1,10):
        if line[:3]== f'{i}00' or line[:4]== f'1000':
            
            solutions[index] = []
            
            start = -1
            end = -1
            for w,ww in enumerate(line):
                if line[w]=="[":
                    start = w
                elif line[w]=="]":
                    end = w
            string = list(line[start+1:end].split(","))
            solutions[index] += list(map(int,string))
            
            index += 1
            
# we gather the solutions in classes, we want to erase eventual duplicates
    
classes = dict()

classes[1] = solutions[1]

jindex = 1

flag = False

for i in range(2, len(solutions)+1):
    
    flag = False
    
    for j in range(1, jindex +1):
        
        if classes[j] == solutions[i]:
                    
            flag = True
            
    if flag == False:
        
        jindex += 1
        
        classes[jindex] = solutions[i]
               
print("number of solutions:",len(solutions))        
print("number of different solutions:",len(classes))


# ## we analyze the lenght of each solution and the average length of the optimal solutions:
# 

# In[36]:


num_nodes = []
#print('lista',num_nodes)
number = 0 
for i in range(1, len(classes)+1):
    
    number = classes[i][1:].index(1)
    #print(number)
    num_nodes.append(number)
    #print('lista',num_nodes)
    
  
print(num_nodes)
print(len(num_nodes))  
print(set(num_nodes))
    


# # we just discovered that all the 554 solutions visit 35 nodes. Now we want to verify which are these nodes...

# In[71]:


which_numbers = dict()



for i in range(1, len(classes)+1):
    
    for h in range(1,36):
        numero = classes[i][h]
        which_numbers[numero] = 1

nodes = which_numbers.keys()    
print("which are these nodes?", nodes)
print("How many are these nodes? ", len(nodes) )


# # We just discovered that the 35 nodes visited are always the same nodes. Now we want to verify in which position they happen in percentage...

# In[67]:


percentages = dict()

for node in nodes:
    
    percentages[node] = np.zeros(35)
    
    for i in range(1, len(classes)+1):
        
        where = classes[i].index(node)
        
        percentages[node][where-1] += 1
        
    percentages[node] = np.round(percentages[node]/len(classes),3)
    #print('LUNGHEZZA', len(percentages[node]))
    
print(percentages)



# # We print the table with the percentages of the visiting positions of each node

# In[69]:


# Hereby we write the table!

first_line = ["Numbers and positions"]
for i in range(1,36):
    first_line.append(i)
    
TablePercentages = PrettyTable(first_line)

for node in nodes:
    new_line = np.append(node,percentages[node])
    TablePercentages.add_row(new_line) 

print(TablePercentages)
f_Table = open("TablePercentages.txt", "a")
print(TablePercentages, file = f_Table)
f_Table.close() 


# In[ ]:




