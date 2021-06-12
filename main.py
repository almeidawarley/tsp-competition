import cplex as cp
import random as rd
import time as tm
import numpy as np
import env

def name_xvariable(i, j):
    # Name x variable for arc (i, j)
    return 'x_' + str(i) + '_' + str(j)

def name_tvariable(i):
    # Name t variable for node i
    return 't_' + str(i)

def decide_weights(instance, i, j):
    if i == j or (i == 0 and j == 1) or (i == 1 and j == 0):
        return 0
    else:
        if j == 0:
            return 0
        else:
            return instance.rewards[j-1]
    # An arc (k, k) should have zero weight
    # Note that nodes 0 and 1 are the same
    if i == j or (i == 0 and j == 1) or (i == 1 and j == 0):
        return 0
    # Calculate the weight for arc(i, j) 
    # Adjust the indices for nodes 0 and 1
    elif i == 0:
        return instance.rewards[j-1] / instance.times[i][j-1]
    elif j == 0:
        return instance.rewards[j] / instance.times[i-1][j]
    # Highest reward possible for time equal 0
    elif instance.times[i-1][j-1] == 0:
        return 1
    # Standard calculation for other arcs
    else:
        return instance.rewards[j-1] / instance.times[i-1][j-1]

def create_variables(instance, solver):
    # Create decision variables
    names = []
    coefficients = []
    types = ['C' for i in instance.nodes for j in instance.nodes]
    uppers = [1 for i in instance.nodes for j in instance.nodes]
    lowers = [0 for i in instance.nodes for j in instance.nodes]

    for i in instance.nodes:
        for j in instance.nodes:
            names.append(name_xvariable(i, j))
            coefficients.append(decide_weights(instance, i, j))

    solver.variables.add(obj = coefficients, ub = uppers, lb = lowers, types = types, names = names)

    names = []
    coefficients = []
    types = ['C' for i in instance.nodes]
    lowers = [0 for i in instance.nodes]

    for i in instance.nodes:
        names.append(name_tvariable(i))
        coefficients.append(0)

    solver.variables.add(obj = coefficients, lb = lowers, types = types, names = names)

def create_constraint1(instance, solver):
    # Flow constraint per node
    rows = []
    senses = []
    rhs = []
    names = []

    for i in instance.nodes:
        senses.append('E')
        names.append('flow_' + str(i))
        coefficients = []
        variables = []

        # Check whether node i is the origin or destination depot
        if i == 1:
            rhs.append(1)
        elif i == 0:
            rhs.append(-1)
        else:
            rhs.append(0)

        for k in instance.nodes:
            if i != k:
                # Block direct transition between nodes 0 and 1
                if not ((i == 0 and k == 1) or (i == 1 and k == 0)):
                    coefficients.append(1)
                    variables.append(name_xvariable(i, k))
                    coefficients.append(-1)
                    variables.append(name_xvariable(k, i))
        rows.append([variables,coefficients])

    solver.linear_constraints.add(lin_expr = rows, senses = senses, rhs = rhs, names = names)

def create_constraint2(instance, solver):
    # Depart from node
    rows = []
    senses = []
    rhs = []
    names = []

    for i in instance.nodes:
        senses.append('L')
        names.append('dpt_' + str(i))
        coefficients = []
        variables = []
        rhs.append(1)

        for k in instance.nodes:
            coefficients.append(1)
            variables.append(name_xvariable(i, k))
        rows.append([variables,coefficients])

    solver.linear_constraints.add(lin_expr = rows, senses = senses, rhs = rhs, names = names)

def create_constraint3(instance, solver):
    # Arrival at node
    rows = []
    senses = []
    rhs = []
    names = []

    for i in instance.nodes:
        senses.append('L')
        names.append('arr_' + str(i))
        coefficients = []
        variables = []
        rhs.append(1)

        for k in instance.nodes:
            coefficients.append(1)
            variables.append(name_xvariable(k, i))
        rows.append([variables,coefficients])

    solver.linear_constraints.add(lin_expr = rows, senses = senses, rhs = rhs, names = names)

def create_constraint4(instance, solver, M = 10000):
    # Time constraints
    rows = []
    senses = []
    rhs = []
    names = []

    for i in instance.nodes:
        for j in instance.nodes:
            if i != j:
                senses.append('G')
                names.append('tmp_' + str(i) + '_' + str(j))
                coefficients = []
                variables = []
                rhs.append(1 - M)
                
                coefficients.append(1)
                variables.append(name_tvariable(j))
                coefficients.append(-1)
                variables.append(name_tvariable(i))
                coefficients.append(-1 *  M)
                variables.append(name_xvariable(i, j))

                rows.append([variables,coefficients])

    solver.linear_constraints.add(lin_expr = rows, senses = senses, rhs = rhs, names = names)

def create_restriction(instance, solver, size):
    # Restrict size of the tour
    rows = []
    senses = []
    rhs = []
    names = []
    
    senses.append('L')
    names.append('rst')
    coefficients = []
    variables = []
    rhs.append(size)

    for i in instance.nodes:
        for j in instance.nodes:
            coefficients.append(1)
            variables.append(name_xvariable(i, j))
    rows.append([variables,coefficients])

    solver.linear_constraints.add(lin_expr = rows, senses = senses, rhs = rhs, names = names)

def cut_infeasible(solver, solution, cuts):
    # Cut infeasible solutions
    rows = []
    senses = []
    rhs = []
    names = []
    
    senses.append('L')
    names.append('feas_' + str(cuts))
    coefficients = []
    variables = []
    
    counter = 0
    index = 1
    while solution[index] != 1:
        coefficients.append(1)
        variables.append(name_xvariable(solution[index - 1], solution[index]))
        index += 1
        counter += 1
    rhs.append(counter - 1)
    rows.append([variables,coefficients])

    solver.linear_constraints.add(lin_expr = rows, senses = senses, rhs = rhs, names = names)

def cut_impossible(instance, solver):
    # Cut impossible nodes
    rows = []
    senses = []
    rhs = []
    names = []

    for i in instance.nodes:

        # True if it is not possible to reach the node stright from depot
        first = instance.times[0][i-1] > instance.closing[i-1]
        # True if it is not possible to reach the depot stright from node
        second = instance.opening[i-1] + instance.times[i-1][0] > instance.maximum
        # if first:
            # print('Node ', i, ' attends the 1st criteria')
        # if second:
            # print('Node ', i, ' attends the 2nd criteria')
            
        if i != 0 and (first or second):
            senses.append('E')
            names.append('imp_' + str(i))
            coefficients = []
            variables = []
            rhs.append(0)

            for k in instance.nodes:
                coefficients.append(1)
                variables.append(name_xvariable(i, k))
                if i != k:
                    coefficients.append(1)
                    variables.append(name_xvariable(k, i))
            rows.append([variables,coefficients])

    solver.linear_constraints.add(lin_expr = rows, senses = senses, rhs = rhs, names = names)

def cut_improbable(instance, solver):
    # Cut improbable arcs
    rows = []
    senses = []
    rhs = []
    names = []

    for i in instance.nodes:
        for j in instance.nodes:

            # True if leaving the earliest from node i cannot reach node j in time
            flag = instance.opening[i-1] + instance.times[i-1][j-1] > instance.closing[j-1]
            # if flag:
                # print('Arc (', i, ',', j, ') attends the 3rd criteria')

            if not(i == j or (i == 0 and j == 1) or (i == 1 and j == 0)) and flag:
                senses.append('E')
                names.append('imp_' + str(i) + '_' + str(j))
                coefficients = []
                variables = []
                rhs.append(0)
                coefficients.append(1)
                variables.append(name_xvariable(i, j))
                rows.append([variables,coefficients])

    solver.linear_constraints.add(lin_expr = rows, senses = senses, rhs = rhs, names = names)

def build_model(instance):
    # Build model
    solver = cp.Cplex()
    solver.objective.set_sense(solver.objective.sense.maximize)
    # solver.set_log_stream(None)
    # solver.set_results_stream(None)

    create_variables(instance, solver)
    create_constraint1(instance, solver)
    create_constraint2(instance, solver)
    create_constraint3(instance, solver)
    # create_constraint4(instance, solver)
    cut_impossible(instance, solver)
    cut_improbable(instance, solver)

    return solver

def run_model(instance, solver, size):
    # Run model    
    create_restriction(instance, solver, size)
    solver.solve()
    objective = solver.solution.get_objective_value()
    solution = {variable: value for (variable, value) in zip(solver.variables.get_names(), solver.solution.get_values())}
    print('Objective: ', objective)
    solver.write('model.lp')
    solver.linear_constraints.delete('rst')
    return solution


def simulate_solution(instance, solution, flag = False):
    # Simulate solution
    time, reward, penalty, feasibility = instance.check_solution(solution)
    if flag:
        print('Time: ', time)
        print('Reward: ', reward)
        print('Penalty: ', penalty)
        print('Feasibility: ', feasibility)
    return reward, feasibility


def format_solution(instance, solution):
    # Format solution
    temporary = {}
    for variable, value in solution.items():
        if value > 0.1 and 'x' in variable:
            _, o, d = variable.split('_')
            o = int(o)
            d = int(d)
            temporary[o] = d

    formatted = []
    index = 1
    formatted.append(1)
    while temporary[index] != 0:
        formatted.append(temporary[index])
        index = temporary[index]
    formatted.append(1)

    for i in instance.nodes:
        if i != 0:
            if i not in formatted:
                formatted.append(i)

    return formatted


# Instance data
instance = env.Env(from_file = True,  
    x_path = 'data/valid/instances/instance0001.csv', 
    adj_path = 'data/valid/adjs/adj-instance0001.csv')

# Random data
# instance = env.Env(55, seed=3119615) 

# Instance adjustements
instance.nodes = list(range(0, instance.n_nodes + 1))
instance.rewards = instance.x[:, -2]
instance.maximum = instance.x[:, -1][1]
instance.opening = instance.x[:, -4]
instance.closing = instance.x[:, -3]

# Estimate time
weights = np.random.rand(instance.n_nodes, instance.n_nodes)
instance.times = weights * instance.adj
solver = build_model(instance)
solver.write('model.lp')

# Iterative approach
best_solution = []
best_reward = -1
start = tm.time()
size = 2
counter = 0
iterations = 100
cuts = 0

while counter < iterations and size < instance.n_nodes:
    solution = run_model(instance, solver, size)
    for variable, value in solution.items():
        if 'x' in variable and value > 0:
            print(variable, ',', value)
    solution = format_solution(instance, solution)
    # print('Solution: ', solution)
    reward, feasibility = simulate_solution(instance, solution)
    if feasibility:
        # print('Solution feasible')
        if reward > best_reward:
            best_solution = solution
            best_reward = reward
        size += 1
    else:
        # print('Solution infeasible')
        cut_infeasible(solver, solution, cuts)
        cuts += 1
    counter += 1

print('Counter: ', counter)
print('Size: ', size)
print('Best reward: ', best_reward)
print('Best solution: ', best_solution)

end = tm.time()
print('Time: ', end - start)

a = input('wait...')

solver.MIP_starts.delete()
solver.linear_constraints.add(lin_expr = [
    # [['x_1_11'],[1]],
    [['x_11_7'],[1]],
    [['x_7_16'],[1]],
    [['x_16_2'],[1]],
    [['x_2_13'],[1]],
    [['x_13_5'],[1]]
   # [['x_5_0'],[1]]
],
    senses = ['E', 'E', 'E', 'E', 'E'], rhs = [1,1,1,1,1])
solver.write('model.lp')
solver.solve()
solution = {variable: value for (variable, value) in zip(solver.variables.get_names(), solver.solution.get_values())}
solution = format_solution(instance, solution)
print(solution)
