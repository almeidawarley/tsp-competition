import numpy as np
from baseline_surrogate.demo_surrogate import check_surrogate_solution

import env

import instance

from generator.op.timewindows import TWGenerator as TW

from op_utils.op import dist_l2_closest_integer

from op_utils.op import make_dist_matrix

#from generator.op.timewindows import get_adjacency_matrix

from env import Env

#TWclass = TWGenerator()

#return TWclass.get_adjacency_matrix

def nn_algo(init_node, cost_matrix, n_nodes):
    """
    Nearest Neighbour algorithm
    """
    cost_matrix = cost_matrix.copy()

    for i in range(0, n_nodes):
        cost_matrix[i][i] = np.inf

    tour = [init_node]

    for _ in range(n_nodes - 1):
        node = tour[-1]-1
        min_index = np.argmin(cost_matrix[node])
        for t in tour:
            cost_matrix[min_index ][t-1] = np.inf
            cost_matrix[t-1][min_index] = np.inf
        tour.append(min_index + 1)
    tour.append(init_node)
    return tour

n_nodes = 55
points= np.arange(n_nodes)
instancee = env.Env(n_nodes, seed=3119615)
#cost_matrix=make_dist_matrix(points, dist=dist_l2_closest_integer, to_integer=True, rd=4)
#cost_matrix=instancee
cost_matrix=instance.make_instance(n_nodes, seed=3119615)[1]
times=instance.make_instance(n_nodes, seed=3119615)[0]
print('coordinates, time windows, prizes and tour time',times)
print('distance matrix', cost_matrix)
solution=nn_algo(1, cost_matrix, n_nodes)
check_surrogate_solution(solution)

def nn_algo_1_2(init_node, cost_matrix, n_nodes, times):
    """
    Nearest Neighbour algorithm
    """
    cost_matrix = cost_matrix.copy()
    
    

    for i in range(0, n_nodes):
        cost_matrix[i][i] = np.inf

    tour = [init_node]

    for _ in range(n_nodes - 1):
        node = tour[-1]-1
        
        min_index = np.argmin(cost_matrix[node])
        
        # we delete those nodes "min_index" which has a closed window BEFORE the opening of "node" or that has an opening AFTER the closing of 1
        if times[min_index][4]<times[node][3] or times[min_index][3]>times[min_index][6]:
        
          for t in tour:
              cost_matrix[min_index ][t-1] = np.inf
              cost_matrix[t-1][min_index] = np.inf
 
          
        else:
          for t in tour:
              cost_matrix[min_index ][t-1] = np.inf
              cost_matrix[t-1][min_index] = np.inf
          tour.append(min_index + 1)
          
    tour.append(init_node)
    for i in range(1, n_nodes+1):
      if i not in tour:
        tour.append(i)
    
    return tour

print('modified nearest neighborhood')
solution2=nn_algo_1_2(1, cost_matrix, n_nodes, times)
check_surrogate_solution(solution2)


# first furthest node neighborhood: we start from the furthest node from 1 which is accessible and then we tour with a neirest neighborhood algo
def fn_algo(init_node, cost_matrix, n_nodes, times):
    """
    Nearest Neighbour algorithm
    """
    cost_matrix = cost_matrix.copy()
    
    
    # we temporarily define the distance from a node to itself as 0, then, after we have chosen the furthest feasible node, we modify it to infinite
    for i in range(0, n_nodes):
        cost_matrix[i][i] = 0

    tour = [init_node]
  
    node = 0
        
    
    
    while len(tour)<2:
    
        max_index = np.argmax(cost_matrix[node])
        
        # we delete those nodes "min_index" which has a closed window BEFORE the opening of "node" or that has an opening AFTER the closing of 1
        if times[max_index][4]<times[node][3] or times[max_index][3]>times[max_index][6]:
          
            for t in tour:
                cost_matrix[t-1][max_index] = 0
 
          
        else:
            for t in tour:
                cost_matrix[max_index ][t-1] = np.inf
                cost_matrix[t-1][max_index] = np.inf
            tour.append(max_index + 1)


    for i in range(0, n_nodes):
          cost_matrix[i][i] = np.inf


    for _ in range(n_nodes - 2):
        node = tour[-1]-1
        
        min_index = np.argmin(cost_matrix[node])
        
        # we delete those nodes "min_index" which has a closed window BEFORE the opening of "node" or that has an opening AFTER the closing of 1
        if times[min_index][4]<times[node][3] or times[min_index][3]>times[min_index][6]:
        
          for t in tour:
              cost_matrix[min_index ][t-1] = np.inf
              cost_matrix[t-1][min_index] = np.inf
 
          
        else:
          for t in tour:
              cost_matrix[min_index ][t-1] = np.inf
              cost_matrix[t-1][min_index] = np.inf
          tour.append(min_index + 1)
          
    tour.append(init_node)
    for i in range(1, n_nodes+1):
      if i not in tour:
        tour.append(i)
    
    return tour

print('first node is the furthest feasible node, then we apply nearest neighbor')
solution3=fn_algo(1, cost_matrix, n_nodes, times)
check_surrogate_solution(solution3)


def sn_algo(init_node, cost_matrix, n_nodes, times):
    """
    Nearest Neighbour algorithm
    """
    cost_matrix = cost_matrix.copy()
    
    

    for i in range(0, n_nodes):
        cost_matrix[i][i] = np.inf

    tour = [init_node]

    for _ in range(n_nodes - 1):
        node = tour[-1]-1
        
        min_index0 = np.argmin(cost_matrix[node])
        dist=cost_matrix[node][min_index0]
        cost_matrix[node][min_index0 ]= np.inf
        min_index = np.argmin(cost_matrix[node])
        cost_matrix[node][min_index0 ]= dist
        
        # we delete those nodes "min_index" which has a closed window BEFORE the opening of "node" or that has an opening AFTER the closing of 1
        if times[min_index][4]<times[node][3] or times[min_index][3]>times[min_index][6]:
        
          for t in tour:
              cost_matrix[min_index ][t-1] = np.inf
              cost_matrix[t-1][min_index] = np.inf
 
          
        else:
          for t in tour:
              cost_matrix[min_index ][t-1] = np.inf
              cost_matrix[t-1][min_index] = np.inf
          tour.append(min_index + 1)
          
    tour.append(init_node)
    for i in range(1, n_nodes+1):
      if i not in tour:
        tour.append(i)
    
    return tour

print('second nearest neighborhood')
solution4=sn_algo(1, cost_matrix, n_nodes, times)
check_surrogate_solution(solution4)