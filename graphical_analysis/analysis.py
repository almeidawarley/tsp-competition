import env
import numpy as np
import math
import copy
import time
from pathlib import Path
import matplotlib.pyplot as plt

def load_instance(identifier):
    # Load instance from file

    instance = env.Env(from_file=True,
                       x_path='data/valid/instances/{}.csv'.format(identifier),
                       adj_path='data/valid/adjs/adj-{}.csv'.format(identifier))

    return instance


def load_validation(use_test):
    # Load validation instance
    if use_test:
        instance = env.Env(65, seed=6537855)  # test phase
    else:
        instance = env.Env(55, seed=3119615)  # validation phase
    return instance

def adjust_instance(instance):
    # Perform instance adjustements

    instance.nodes = list(range(1, instance.n_nodes + 1))
    instance.rewards = instance.x[:, -2]
    instance.maximum = instance.x[:, -1][1]
    instance.opening = instance.x[:, -4]
    instance.closing = instance.x[:, -3]

    return instance

def check_performance(instance, solution, iterations=10 ** 4):
    # Check solution performance

    # Create average variables
    avg_time = 0
    avg_reward = 0
    avg_penalty = 0
    percentage = 0

    counter = 0
    while counter < iterations:
        # Call black-box simulator
        time, reward, penalty, feasible = instance.check_solution(solution)

        # Update average variables
        avg_time += time
        avg_reward += reward
        avg_penalty += penalty
        percentage += 1 if feasible else 0

        counter += 1

    avg_time /= iterations
    avg_reward /= iterations
    avg_penalty /= iterations
    percentage /= iterations
    avg_objective = avg_reward + avg_penalty

    return avg_time, avg_reward, avg_penalty, percentage, avg_objective

def solution_analysis(instance, sol):
    n_nodes = len(instance.nodes)

    #Order nodes not explored in solution from earliest opening time to latest opening time
    index_2nd_1 = [i for i, n in enumerate(sol) if n == 1][1]
    not_chosen_nodes = [i for i in sol[-(len(sol)-index_2nd_1):] if i != 1]
    openings = []
    for node in not_chosen_nodes:
        openings.append(instance.x[node - 1, -4])
    not_chosen_nodes = [x for _,x in sorted(zip(openings, not_chosen_nodes))]
    sol[-(len(sol)-index_2nd_1-1):] = not_chosen_nodes

    #Check performance of the solution
    avg_time, avg_reward, avg_penalty, percentage, avg_objective = check_performance(instance, sol)
    print('\n\n\nSolution: ', sol)
    print('Objective: ', avg_objective)
    print('Time: ', avg_time)
    print('Reward: ', avg_reward)
    print('Penalty: ', avg_penalty)
    print('Percentage: ', percentage)
    print('Maximum Time: ', instance.maximum)

    #Plot options
    f = plt.figure()
    f.set_figwidth(20)
    f.set_figheight(10)
    #To get information about every node in the solution
    openings = []
    closings = []
    rewards = []
    distance_node_1 = []
    Tmax = instance.maximum = instance.x[:, -1][1]
    visit_order = 0
    color = 'green'; label='Time window of nodes in solution'
    #Iterate through all the nodes of the solution
    for node in sol:
        #By entering this condition, it means from this point we iterate in nodes we do not explore in this solution, so we change the labels for the plot
        if visit_order > 0 and node == 1:
            color = 'red'; label='Time window of impossible nodes'
        #Get information about every node in the solution
        openings.append(instance.x[node - 1, -4])
        closings.append(instance.x[node - 1, -3])
        rewards.append(instance.x[node - 1, -2])
        distance_node_1.append(instance.adj[0, node - 1])
        #Enter this condition to print the information about the node on the plot
        if visit_order>0 and node!=1:
            plt.plot([0, distance_node_1[visit_order]], [visit_order, visit_order], color='blue', label='Distance to node 1 and reward')
            plt.annotate(str(rewards[visit_order]), (openings[visit_order]-22, visit_order-0.5), color='blue', fontsize=6)
            plt.plot([openings[visit_order], closings[visit_order]], [visit_order, visit_order], color=color, label=label)
            plt.annotate(str(node), (closings[visit_order]+10, visit_order-0.5), color=color, fontsize=6)
        visit_order += 1

    #Add labels, and maximal time information
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.xlabel('Time', fontsize=18)
    plt.ylabel('Order of visited nodes', fontsize=18)
    plt.plot([Tmax, Tmax], [0, len(sol)], color='black')
    plt.annotate('Tmax', (Tmax+10, 20.5), color='black', fontsize=6)
    plt.savefig('analysis.png', dpi=1200)
    plt.show()


if __name__ == "__main__":
  
    instance_number = 1
    use_validation_or_test = True # True for validation and test phase, otherwise use of instance_number
    use_test = True # True for test phase only

    #Best solution found for validation instance
    #sol = [1, 35, 11, 18, 19, 25, 16, 8, 26, 44, 2, 54, 3, 20, 9, 21, 40, 5, 49, 23, 7, 37, 41, 55, 4, 51, 30, 53, 1, 6, 10, 12, 13, 14, 15, 17, 22, 24, 27, 28, 29, 31, 32, 33, 34, 36, 38, 39, 42, 43, 45, 46, 47, 48, 50, 52]
    #Best solution found for test instance
    sol = [1, 32, 45, 55, 41, 47, 49, 5, 44, 23, 6, 57, 2, 16, 42, 60, 33, 46, 11, 43, 19, 64, 13, 29, 9, 35, 65, 22, 62, 7, 63, 4, 24, 30, 40, 48, 1, 3, 8, 10, 12, 14, 15, 17, 18, 20, 21, 25, 26, 27, 28, 31, 34, 36, 37, 38, 39, 50, 51, 52, 53, 54, 56, 58, 59, 61]


    if not (use_validation_or_test):
        instance = load_instance("instance" + str(instance_number).zfill(4))
    else:
        instance = load_validation(use_test)
    instance = adjust_instance(instance)

    #Run the solution analysis on the instance. This will save a analysis.png file
    solution_analysis(instance, sol)
