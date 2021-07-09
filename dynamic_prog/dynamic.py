import env
import numpy as np
import math
import copy
import time
from pathlib import Path


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


def export_solution(solution, timestamp, score, instance_name, name='no_name'):
    """
    Export solution to .out file
    """
    # Write solution information to .out file
    with open(
            'dyn_solutions/' + 'instance_' + instance_name + '_' + name + '_time_' + timestamp + '_score_' + str(score),
            'w') as output:
        for node in solution:
            output.write('{}\n'.format(node))
    return name


# Iterate through all the nodes n of the graph, and try the solution [1,n,1].
# If expected objective of this solution is <0 after Monte-Carlo simulations, we forget this node
# And will never consider it since it will have a negative objective whenever we visit it.
# This function returns a list of the good nodes only.
def to_explore(instance):
    n_nodes = len(instance.nodes)
    aller_retour = {}
    for i in range(2, n_nodes + 1):
        sol = [1, i, 1]
        end_sol = [j for j in range(2, n_nodes + 1)];
        end_sol.remove(i)
        sol += end_sol
        MonteCarlo = 1000  # Number of Monte Carlo samples. Higher number means less noise.
        obj = 0  # Objective averaged over Monte Carlo samples, to be used for surrogate modelling
        for _ in range(MonteCarlo):
            obj_cost, rewards, pen, feas = instance.check_solution(sol)
            obj = obj + (rewards + pen) / MonteCarlo
        aller_retour[i] = obj

    to_explore = [node if value > 0 else 0 for (node, value) in aller_retour.items()]
    to_explore = [node for node in to_explore if node != 0]
    to_explore.insert(0, 1)
    print("\nThe function to_explore removed " + str(
        n_nodes - len(to_explore)) + " nodes of the graph which initially contained " + str(n_nodes) + ".")
    print("We remain with " + str(len(to_explore)) + " nodes, so we deleted " + str(
        round(100 * (n_nodes - len(to_explore)) / n_nodes, 2)) + "%.")

    return to_explore


# Compute matrix E and TT. E is a N* . N* . Tmax matrix (where N* is the number of good nodes returned by the function to_explore), and an entry E[n_1, n_2, t]
# of the matrix stores the expected objective for leaving from node n_i to n_j at time t. TT is a N* . N* . n_samples_for_TT matrix that stores travel time samples
# for going from node n_i to n_j without any delay caused by time windows.
# n_mc_for_E is the desired number of MC simulations to do before storing the average into E. n_samples_for_TT is the number of travel time samples from edges to save.
def E_and_TT_matrices(instance, dic_good_nodes, n_mc_for_E, n_samples_for_TT):
    n_nodes = len(instance.nodes)
    n_good_nodes = len(dic_good_nodes)
    print("\nStarting to compute E and TT matrices on ", n_good_nodes, "nodes.")

    T = math.floor(instance.maximum)
    E = np.zeros((len(to_explore), len(to_explore), T))
    TT = np.zeros((len(to_explore), len(to_explore), n_samples_for_TT))

    iteration = 0
    for index_i, n_i in dic_good_nodes.items():
        start = time.time()

        instance_new = copy.deepcopy(instance)
        instance_new.adj[0, n_i - 1] = 0  # Distance on edge 1-to-n_i to 0.
        instance_new.x[n_i - 1, -2] = 0  # Reward at n_i to 0

        for index_j, n_j in dic_good_nodes.items():

            # Keep 0 in the matrix on the diagonal
            if n_j == n_i:
                continue

            sol = [1, n_i, n_j, 1]
            # If looking at edge 1-to-n_j, solution has different form
            if n_i == 1:
                sol = [1, n_j, 1]
            elif n_j == 1:
                sol = [1, n_i, 1]
            rest = [i for i in range(1, n_nodes + 1) if i not in sol]
            sol = sol + rest

            # COMPUTE MATRIX TT

            # Save travel time samples from n_i to n_j
            instance_new.x[n_i - 1, -4] = 0  # Opening time at n_i to 0
            o_j = instance_new.x[n_j - 1, -4]  # Save opening time at n_j
            instance_new.x[n_j - 1, -4] = 0  # Opening time at n_j to 0
            d_j_1 = instance_new.adj[n_j - 1, 0] = 0  # Save distance on edge n_j-to-1
            instance_new.adj[n_j - 1, 0] = 0  # Distance on edge n_j-to-1 to 0
            for i in range(n_samples_for_TT):
                tour_time, rewards, pen, feas = instance_new.check_solution(sol)
                TT[index_i, index_j, i] = tour_time

            # COMPUTE MATRIX E

            instance_new.x[n_j - 1, -4] = o_j     # Restore opening time at n_j for computing E
            instance_new.adj[n_j - 1, 0] = d_j_1  # Restore distance on edge n_j-to-1

            for t in range(T):
                instance_new.x[n_i - 1, -4] = t  # Opening time at n_i to t
                # Simulate edge n_i-to-n_j by leaving node n_i at time t and store expected objective in E
                e = 0
                for _ in range(n_mc_for_E):
                    tour_time, rewards, pen, feas = instance_new.check_solution(sol)
                    e += (rewards + pen)

                E[index_i, index_j, t] = e / n_mc_for_E

                # Shortcut: If t>4 and the last 4 values of E[index_i, index_j] are the same, then fill the rest of matrix with the same value.
                if t > 4 and E[index_i, index_j, t - 3] == E[index_i, index_j, t - 2] and E[index_i, index_j, t - 2] == \
                        E[index_i, index_j, t - 1] and E[index_i, index_j, t - 1] == E[index_i, index_j, t]:
                    E[index_i, index_j, t + 1:] = list(E[index_i, index_j, t] * np.ones(T - t - 1))
                    break

        del (instance_new)
        end = time.time()
        print("Node #", iteration + 1, "/", len(to_explore), "completed in ", end - start, " seconds")
        iteration += 1
    return E, TT


# Same as E_and_TT_matrices, but for E, it computes an additional dimension: we look at solutions of the form [1, n_i, n_j, n_k, 1] instead of [1, n_i, n_j, 1].
def E_2(instance, dic_good_nodes, n_mc_for_E):
    n_nodes = len(instance.nodes)
    n_good_nodes = len(dic_good_nodes)
    print("\nStarting to compute E and TT matrices on ", n_good_nodes ** 2, "nodes.")

    T = math.floor(instance.maximum)
    E = np.zeros((len(to_explore), len(to_explore), len(to_explore), T))

    iteration = 0
    for index_i, n_i in dic_good_nodes.items():
        start = time.time()

        instance_new = copy.deepcopy(instance)
        instance_new.adj[0, n_i - 1] = 0  # Distance on edge 1-to-n_i to 0.
        instance_new.x[n_i - 1, -2] = 0  # Reward at n_i to 0

        for index_j, n_j in dic_good_nodes.items():
            start_2 = time.time()

            for index_k, n_k in dic_good_nodes.items():

                # Solution to test
                if n_i == 1 and n_j != 1 and n_k != 1:
                    sol = [1, n_j, n_k, 1]
                elif n_i != 1 and n_j == 1 and n_k != 1:
                    sol = [1, n_i, 1]
                elif n_i != 1 and n_j != 1 and n_k == 1:
                    sol = [1, n_i, n_j, 1]
                elif n_i == 1 and n_j == 1 and n_k != 1:
                    sol = [1, 1]
                elif n_i == 1 and n_j != 1 and n_k == 1:
                    sol = [1, n_j, 1]
                elif n_i != 1 and n_j == 1 and n_k == 1:
                    sol = [1, n_i, 1]
                elif n_i == 1 and n_j == 1 and n_k == 1:
                    sol = [1, 1]
                elif n_i != 1 and n_j != 1 and n_k != 1:
                    sol = [1, n_i, n_j, n_k, 1]
                sol = sol[1:-1]
                sol = list(dict.fromkeys(sol))
                sol = [1] + sol + [1]
                rest = [i for i in range(1, n_nodes + 1) if i not in sol]
                sol = sol + rest

                for t in range(T):
                    instance_new.x[n_i - 1, -4] = t  # Opening time at n_i to t

                    # Simulate edge n_i-to-n_j by leaving node n_i at time t and store expected objective in E
                    e = 0
                    for _ in range(n_mc_for_E):
                        tour_time, rewards, pen, feas = instance_new.check_solution(sol)
                        e += (rewards + pen)

                    E[index_i, index_j, index_k, t] = e / n_mc_for_E

                    # Shortcut: If t>4 and the last 4 values of E[index_i, index_j] are the same, then fill the rest of matrix with the same value.
                    if t > 4 and E[index_i, index_j, index_k, t - 3] == E[index_i, index_j, index_k, t - 2] and E[
                        index_i, index_j, index_k, t - 2] == \
                            E[index_i, index_j, index_k, t - 1] and E[index_i, index_j, index_k, t - 1] == E[
                        index_i, index_j, index_k, t]:
                        E[index_i, index_j, index_k, t + 1:] = list(
                            E[index_i, index_j, index_k, t] * np.ones(T - t - 1))
                        break
            end_2 = time.time()
            print("Node #", iteration + 1, "/", len(to_explore) ** 2, "completed in ", end_2 - start_2, " seconds")
            iteration += 1
        del (instance_new)
        end = time.time()
        print("Node #", iteration + 1, "/", len(to_explore) ** 2, "completed in ", end - start, " seconds")
    return E


# From a set of travel time samples stored in TT for a specific edge, will return the nth biggest value, where n is determined by ratio_biggest.
# For example, if ratio_biggest=0.5, the mean value will be returned. If ratio_biggest=0.9 and we have 200 samples, the 180th sample will be returned.
def TT_value_to_use(ratio_biggest, samples):
    samples.sort()
    return samples[int(np.floor(ratio_biggest * len(samples)))]


# From a node n_i at time current_time, lists all the nodes where we have time to go to come back to n_1 before Tmax.
# Two ways are proposed to compute this list:
# - We can use a single value (for example the mean with ratio_biggest=0.5) from the travel time samples stored in TT to estimate if we can come back in time.
# - We can use many travel time samples stored in TT and see how many time we success to come back in time. If the ratio we come back on time is bigger than
#   min_ratio_for_checkup, then we can go to the node.
def checkup(instance, current_time, n_i, dic_good_nodes, TT, n_samples_to_consider=100, min_ratio_for_checkup=0.9,
            use_single_value=False, ratio_biggest=0.5):
    checked = []
    Tmax = math.floor(instance.maximum)
    index_i = list(dic_good_nodes.keys())[list(dic_good_nodes.values()).index(n_i)]

    for index_j, n_j in dic_good_nodes.items():
        # To estimate the travel time, we will consider only one value from the samples
        if use_single_value:
            # If current_time + max(travel_time_from_n_i_to_n_j, opening_time_window_n_j) + travel_time_from_n_j_to_n_1 <= Tmax then we can go to n_j
            if current_time + max(TT_value_to_use(ratio_biggest, TT[index_i, index_j, :]), instance.x[n_j - 1, -4]) \
                    + TT_value_to_use(ratio_biggest, TT[index_j, 0, :]) <= Tmax:
                checked.append(n_j)
        # We consider n_samples_to_consider to approximate well the travel times from n_i to n_j, and n_j to n_1
        else:
            # Shortcut: if we can come back to node 1 with longest travel times, then we can go to node n_j
            if current_time + max(max(TT[index_i, index_j, :]), instance.x[n_j - 1, -4]) + max(
                    TT[index_j, 0, :]) <= Tmax:
                checked.append(n_j)
            else:
                counter_1 = 0
                for tt_1 in TT[index_i, index_j, :n_samples_to_consider]:
                    counter_2 = 0
                    for tt_2 in TT[index_j, 0, :n_samples_to_consider]:
                        if current_time + max(tt_1, instance.x[n_j - 1, -4]) + tt_2 <= Tmax:
                            counter_2 = counter_2 + 1 / n_samples_to_consider
                    if counter_2 > min_ratio_for_checkup:
                        counter_1 = counter_1 + 1 / n_samples_to_consider
                # If current_time + max(tt, opening_time_window_n_j) + tt' <= Tmax at least 100*min_ratio_for_checkup % of the time, then we can go to n_j
                if counter_1 > min_ratio_for_checkup:
                    checked.append(n_j)

    # print("\nOn the ", len(dic_good_nodes), " possible nodes to go to from node n_", n_i, " only ", len(checked),
    #      "allow us to come back to node n_1 before Tmax.")
    return checked


# Returns a list of n_samples_for_TT travel time samples for the path [n_i, n_j, n_k] when leaving n_i at time t and stopping when arriving at n_k.
# n_i, n_j and n_k must be different nodes.
def get_TT_2(instance, dic_good_nodes, n_i, n_j, n_k, t, n_samples_for_TT):
    # All the nodes of the path must be different

    instance_new = copy.deepcopy(instance)
    instance_new.adj[0, n_i - 1] = 0  # Distance on edge 1-to-n_i to 0.
    instance_new.x[n_i - 1, -4] = t  # Opening time at n_i to t
    instance_new.adj[n_k - 1, 0] = 0  # Distance on edge n_k-to-1 to 0.
    instance_new.x[n_k - 1, -4] = 0  # Opening time at n_k to 0
    instance_new.x[:, -1] = [float('inf')] * n_nodes  # T_max at infinity (no penalty for a too long tour)
    samples = []
    for i in range(n_samples_for_TT):
        if n_i == 1 and n_j != 1 and n_k != 1:
            sol = [1, n_j, n_k, 1]
        elif n_i != 1 and n_j == 1 and n_k != 1:
            sol = [1, n_i, 1]
        elif n_i != 1 and n_j != 1 and n_k == 1:
            sol = [1, n_i, n_j, 1]
        elif n_i == 1 and n_j == 1 and n_k != 1:
            sol = [1, 1]
        elif n_i == 1 and n_j != 1 and n_k == 1:
            sol = [1, n_j, 1]
        elif n_i != 1 and n_j == 1 and n_k == 1:
            sol = [1, n_i, 1]
        elif n_i == 1 and n_j == 1 and n_k == 1:
            sol = [1, 1]
        elif n_i != 1 and n_j != 1 and n_k != 1:
            sol = [1, n_i, n_j, n_k, 1]
        sol = sol[1:-1]
        sol = list(dict.fromkeys(sol))
        sol = [1] + sol + [1]
        rest = [i for i in range(1, n_nodes + 1) if i not in sol]
        sol = sol + rest
        tour_time, rewards, pen, feas = instance_new.check_solution(sol)
        samples.append(tour_time)
    return samples


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

# Returns the solution found by the one-step look-ahead
def dyn_sol_dim_1(current_time, sol, dic_good_nodes, E, TT, verbose=False):

    # Validation:
    # P = 25: 7.831972000000508
    # P = 50: 8.234618000000543
    # P = 75: 8.149999999999686

    # Test:
    # P = 50:  7.970000000000866 [1, 55, 45, 47, 41, 5, 49, 57, 6, 16, 60, 42, 33, 11, 46, 43, 64, 19, 13, 29, 7, 65, 35, 9, 22, 62, 1, 2, 3, 4, 8, 10, 12, 14, 15, 17, 18, 20, 21, 23, 24, 25, 26, 27, 28, 30, 31, 32, 34, 36, 37, 38, 39, 40, 44, 48, 50, 51, 52, 53, 54, 56, 58, 59, 61, 63]
    # P = 75:  8.29970000000156
    # P = 100: 8.429999999998653

    p = 100

    iteration = 0
    n_i = sol[-1]
    obj = 0
    inv_dic_good_nodes = {v: k for k, v in dic_good_nodes.items()}

    while True:

        iteration += 1
        if verbose:
            print('Iteration: ', iteration)
            print('    Ni: ', n_i)
            print('    Current time: ', current_time)

        check = list(dic_good_nodes.values())
        check = [c for c in check if c not in sol]
        check_id = [c - 1 for c in check]
        if verbose:
            print('    Check: ', check)
        check_indices = []
        for c in check:
            check_indices.append(inv_dic_good_nodes[c])
        if verbose:
            print('    Check indices: ', check_indices)
        check_is_not_empty = (check_indices != [])

        if check_is_not_empty:

            index_i = inv_dic_good_nodes[n_i]
            if verbose:
                print('    E[Ni,:,current_time]', E[index_i, :, current_time])
            comp_reward = E[index_i, check_indices, current_time]
            if verbose:
                print('    Competitive Reward: ', comp_reward)
            if verbose:
                print('    TT[Ni,:,:]: ', TT[index_i,check_indices,:])
            weight = np.percentile(TT[index_i, check_indices, :], p, axis=1)
            weight = [w + current_time for w in weight]
            if verbose:
                print('    Opening: ', instance.opening[check_id])
            weight = np.maximum(weight, instance.opening[check_id])
            weight = [w - current_time for w in weight]
            if verbose:
                print('    Weight: ', weight)
            comp_reward_weighted = [i / max(j, 1) for i, j in zip(comp_reward, weight)]
            if verbose:
                print('    Weighted Competitive Reward: ', comp_reward_weighted)
            index_j = check_indices[np.argmax(comp_reward_weighted)]
            n_j = dic_good_nodes[index_j]
            if verbose:
                print('    Nj: ', n_j)
                print('    Reward: ', E[index_i, index_j, current_time])

            sol_temp = copy.deepcopy(sol)
            sol_temp.append(n_j)
            sol_temp.append(1)
            rest_temp = [i for i in range(1, n_nodes + 1) if i not in sol_temp]
            sol_temp = sol_temp + rest_temp
            _, _, _, _, avg_obj = check_performance(instance, sol_temp)
            instance_new = copy.deepcopy(instance)
            instance_new.adj[n_j - 1, 0] = 0
            avg_time, _, _, _, _ = check_performance(instance_new, sol_temp)
            current_time = int(np.floor(avg_time))
            print('New current time: ', current_time)
            print('New objective: ', avg_obj)
            print('Increase of objective: ', avg_obj - obj)
            if current_time < instance.maximum and (avg_obj - obj) > 0:
                sol.append(n_j)
                obj = copy.deepcopy(avg_obj)
                n_i = n_j
            else:
                break
            if verbose:
                print('    Solution: ', sol, '\n')
        else:
            break

    sol.append(1)
    rest = [i for i in range(1, n_nodes + 1) if i not in sol]
    sol = sol + rest
    return sol

#Returns the solution found by the two-steps look-ahead
def dyn_sol_dim_2(current_time, sol, dic_good_nodes, E, verbose=False):
    # Validation phase with old code:
    # P = 25:
    # P = 50:  5.400000000000936
    # P = 75:
    # P = 100:

    # Test phase:
    # P = 0 :
    # P = 25:
    # P = 50:  3.1599999999998665 [1, 55, 41, 57, 43, 46, 19, 64, 13, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 42, 44, 45, 47, 48, 49, 50, 51, 52, 53, 54, 56, 58, 59, 60, 61, 62, 63, 65]
    # P = 75:
    # P = 100: 2.3499999999997385 [1, 32, 41, 43, 19, 64, 13, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 33, 34, 35, 36, 37, 38, 39, 40, 42, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 65]

    p = 50

    iteration = 0
    n_i = sol[-1]
    obj = 0
    inv_dic_good_nodes = {v: k for k, v in dic_good_nodes.items()}

    while True:

        iteration += 1
        if verbose:
            print('Iteration: ', iteration)
            print('    Ni: ', n_i)
            print('    Current time: ', current_time)

        check = list(dic_good_nodes.values())
        check = [c for c in check if c not in sol]
        check_id = [c - 1 for c in check]
        if verbose:
            print('    Check: ', check)
        check_indices = []
        for c in check:
            check_indices.append(inv_dic_good_nodes[c])
        if verbose:
            print('    Check indices: ', check_indices)
        check_is_not_empty = (check_indices != [])

        if check_is_not_empty:

            index_i = inv_dic_good_nodes[n_i]
            if n_i == 1:
                print('    E[Ni,:,:,current_time]: ', E[index_i, :, :, current_time])
            comp_reward = E[index_i, check_indices, : , current_time][:, check_indices]
            if n_i == 1:
                print('    Competitive Reward: ', comp_reward)

            TT_i = np.ones((len(dic_good_nodes), len(dic_good_nodes)))*1000
            for index_j in check_indices:
                for index_k in check_indices:
                    TT_i[index_j, index_k] = \
                        np.percentile(
                            get_TT_2(instance,
                                     dic_good_nodes,
                                     n_i,
                                     dic_good_nodes[index_j],
                                     dic_good_nodes[index_k],
                                     current_time,
                                     1000)
                            , p)
            TT_i = TT_i[check_indices,:][:,check_indices]
            if n_i == 1:
                print('    TT_i: ', TT_i)
            weight = TT_i + current_time
            if n_i == 1:
                print('    TT_i + current_time: ', weight)
            opening_i = np.zeros((len(dic_good_nodes), len(dic_good_nodes)))
            for index_j in check_indices:
                for index_k in check_indices:
                    opening_i[index_j, index_k] = instance.opening[dic_good_nodes[index_k]-1]
            opening_i = opening_i[check_indices,:][:,check_indices]
            if n_i == 1:
                print('    opening_i: ', opening_i)
            weight = np.maximum(weight, opening_i)
            weight = weight - current_time
            if n_i == 1:
                print('    Weight: ', weight)
            comp_reward_weighted = comp_reward/np.maximum(weight,1)
            if n_i == 1:
                print('    Weighted Competitive Reward: ', comp_reward_weighted)
                print('    Maximum: ', np.max(comp_reward_weighted))
            index_j = check_indices[np.unravel_index(comp_reward_weighted.argmax(), comp_reward_weighted.shape)[0]]
            n_j = dic_good_nodes[index_j]
            if verbose:
                print('    Nj: ', n_j)
                print('    Reward: ', E[index_i, index_j, index_k, current_time])
            sol_temp = copy.deepcopy(sol)
            sol_temp.append(n_j)
            sol_temp.append(1)
            rest_temp = [i for i in range(1, n_nodes + 1) if i not in sol_temp]
            sol_temp = sol_temp + rest_temp
            _, _, _, _, avg_obj = check_performance(instance, sol_temp)
            instance_new = copy.deepcopy(instance)
            instance_new.adj[n_j - 1, 0] = 0
            avg_time, _, _, _, _ = check_performance(instance_new, sol_temp)
            current_time = int(np.floor(avg_time))
            print('    New current time: ', current_time)
            print('    New objective: ', avg_obj)
            print('    Increase of objective: ', avg_obj - obj)
            if current_time < instance.maximum and (avg_obj - obj) > 0:
                sol.append(n_j)
                obj = copy.deepcopy(avg_obj)
                n_i = n_j
            else:
                break
            if verbose:
                print('    Solution: ', sol, '\n')
        else:
            break

    sol.append(1)
    rest = [i for i in range(1, n_nodes + 1) if i not in sol]
    sol = sol + rest
    return sol


if __name__ == "__main__":

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    instance_number = 1
    use_validation_or_test = True # True for validation or test phase, otherwise use of instance_numebr
    use_test = False # True for test phase only
    precomputed = True  # Use E and TT already stored for that instance
    E_dimensions = 2 # Number of steps to look-ahead (can be 1 or 2)

    if not (use_validation_or_test):
        instance = load_instance("instance" + str(instance_number).zfill(4))
    else:
        instance = load_validation(use_test)
    instance = adjust_instance(instance)

    n_nodes = len(instance.nodes)

    to_explore = to_explore(instance)
    dic_good_nodes = {i: to_explore[i] for i in range(0, len(to_explore))}

    print("\nRemaining nodes are:", dic_good_nodes)

    # Compute E and TT if it has not been done before for this instance
    if not (precomputed):
        if E_dimensions == 1:
            E, TT = E_and_TT_matrices(instance=instance,
                                      dic_good_nodes=dic_good_nodes,
                                      n_mc_for_E=100,
                                      # Number of MC simulations to do before storing the average into E
                                      n_samples_for_TT=100  # Travel time samples from edges to save
                                      )
        elif E_dimensions == 2:
            E = E_2(instance=instance,
                    dic_good_nodes=dic_good_nodes,
                    n_mc_for_E=100 # Number of MC simulations to do before storing the average into E
                    )
        Path("./store_E_and_TT").mkdir(parents=True, exist_ok=True)
        if not (use_validation_or_test):
            np.save("./store_E_and_TT/E_test_" + "instance_" + str(instance_number).zfill(4) + "_E_dim_" + str(
                E_dimensions), E)
            if E_dimensions == 1:
                np.save("./store_E_and_TT/TT_test_" + "instance_" + str(instance_number).zfill(4), TT)
        else:
            if use_test:
                np.save("./store_E_and_TT/E_test_" + "instance_" + "test" + "_E_dim_" + str(E_dimensions), E)
                if E_dimensions == 1:
                    np.save("./store_E_and_TT/TT_test_" + "instance_" + "test", TT)
            else:
                np.save("./store_E_and_TT/E_test_" + "instance_" + "valid" + "_E_dim_" + str(E_dimensions), E)
                if E_dimensions == 1:
                    np.save("./store_E_and_TT/TT_test_" + "instance_" + "valid", TT)
    else:
        if not (use_validation_or_test):
            E = np.load("./store_E_and_TT/E_test_" + "instance_" + str(instance_number).zfill(4) + "_E_dim_" + str(
                E_dimensions) + ".npy")
            if E_dimensions == 1:
                TT = np.load("./store_E_and_TT/TT_test_" + "instance_" + str(instance_number).zfill(4) + ".npy")
        else:
            if use_test:
                E = np.load("./store_E_and_TT/E_test_" + "instance_" + "test" + "_E_dim_" + str(E_dimensions) + ".npy")
                if E_dimensions == 1:
                    TT = np.load("./store_E_and_TT/TT_test_" + "instance_" + "test" + ".npy")
            else:
                E = np.load("./store_E_and_TT/E_test_" + "instance_" + "valid" + "_E_dim_" + str(E_dimensions) + ".npy")
                if E_dimensions == 1:
                    TT = np.load("./store_E_and_TT/TT_test_" + "instance_" + "valid" + ".npy")

    check = list(dic_good_nodes.values())
    check_id = [c - 1 for c in check]
    print('Instance.x: ', instance.x[check_id,:])
    print('Instance.adj: ', instance.adj[check_id,:][:,check_id])

    if E_dimensions == 1:
        current_time = int(0)
        sol = [1]

        sol = dyn_sol_dim_1(current_time=current_time,
                            sol=sol,
                            dic_good_nodes=dic_good_nodes,
                            E=E,
                            TT=TT,
                            verbose=True)

        avg_time, avg_reward, avg_penalty, percentage, avg_objective = check_performance(instance, sol)

        print('\n\n\nSolution: ', sol)
        print('Objective: ', avg_objective)
        print('Time: ', avg_time)
        print('Reward: ', avg_reward)
        print('Penalty: ', avg_penalty)
        print('Percentage: ', percentage)
        print('Maximum Time: ', instance.maximum)

    elif E_dimensions == 2:
        current_time = int(0)
        sol = [1]

        sol = dyn_sol_dim_2(current_time=current_time,
                            sol=sol,
                            dic_good_nodes=dic_good_nodes,
                            E=E,
                            verbose=True)

        avg_time, avg_reward, avg_penalty, percentage, avg_objective = check_performance(instance, sol)

        print('\n\n\nSolution: ', sol)
        print('Objective: ', avg_objective)
        print('Time: ', avg_time)
        print('Reward: ', avg_reward)
        print('Penalty: ', avg_penalty)
        print('Percentage: ', percentage)
        print('Maximum Time: ', instance.maximum)

    Path("./dyn_solutions").mkdir(parents=True, exist_ok=True)
    if not (use_validation_or_test):
        export_solution(sol, timestamp, avg_objective, str(instance_number).zfill(4), name='no_name')
    else:
        export_solution(sol, timestamp, avg_objective, 'valid', name='no_name')


