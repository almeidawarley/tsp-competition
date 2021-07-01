import env
import uuid as ui
import numpy as np
import cplex as cp
import time as tm

def name_variablex(i, j):
    # Name variable x for arc (i,j)

    return 'x_' + str(i) + '_' + str(j)

def name_variablet(i):
    # Name variable t for node i

    return 't_' + str(i)

def calculate_weight(instance, i, j):
    # Calculate weight for arc (i,j)

    # If it is a self loop, set weight 0
    if i == j:
        return 0
    # if it is a valid arc, retrieve reward
    else:
        return instance.rewards[j-1]

def create_variables(instance, solver):
    # Create decision variables

    # Create auxiliary vectors
    names = []
    coefficients = []
    types = ['B' for i in instance.nodes for j in instance.nodes]
    uppers = [1 for i in instance.nodes for j in instance.nodes]
    lowers = [0 for i in instance.nodes for j in instance.nodes]

    # Create variable x for each arc (i,j)
    for i in instance.nodes:
        for j in instance.nodes:
            names.append(name_variablex(i, j))
            coefficients.append(calculate_weight(instance, i, j))

    solver.variables.add(obj = coefficients, ub = uppers, lb = lowers, types = types, names = names)

    # Create auxiliary vectors
    M = instance.n_nodes
    names = []
    coefficients = []
    types = ['C' for i in instance.nodes]
    uppers = [M for i in instance.nodes]
    lowers = [0 for i in instance.nodes]

    # Create variable t for each node i
    for i in instance.nodes:
        names.append(name_variablet(i))
        coefficients.append(0)

    solver.variables.add(obj = coefficients, ub = uppers, lb = lowers, types = types, names = names)

def create_flow_constraint(instance, solver):
    # Flow constraint per node

    # Create auxiliary vectors
    rows = []
    senses = []
    rhs = []
    names = []

    # Create flow constraint for each node i
    for i in instance.nodes:
        senses.append('E')
        names.append('flw_' + str(i))
        coefficients = []
        variables = []
        rhs.append(0)

        for k in instance.nodes:
            # Avoid self loop variables
            if i != k:
                coefficients.append(1)
                variables.append(name_variablex(i, k))
                coefficients.append(-1)
                variables.append(name_variablex(k, i))
        rows.append([variables,coefficients])

    solver.linear_constraints.add(lin_expr = rows, senses = senses, rhs = rhs, names = names)

def create_depart_constraint(instance, solver):
    # Depart constraint from node

    # Create auxiliary vectors
    rows = []
    senses = []
    rhs = []
    names = []

    # Create depart constraint for each node i
    for i in instance.nodes:
        senses.append('L')
        names.append('dpt_' + str(i))
        coefficients = []
        variables = []
        rhs.append(1)

        for k in instance.nodes:
            coefficients.append(1)
            variables.append(name_variablex(i, k))
        rows.append([variables,coefficients])

    solver.linear_constraints.add(lin_expr = rows, senses = senses, rhs = rhs, names = names)

def create_arrival_constraint(instance, solver):
    # Arrival constraint at node

    # Create auxiliary vectors
    rows = []
    senses = []
    rhs = []
    names = []

    # Create arrival constraint for each node i
    for i in instance.nodes:
        senses.append('L')
        names.append('arr_' + str(i))
        coefficients = []
        variables = []
        rhs.append(1)

        for k in instance.nodes:
            coefficients.append(1)
            variables.append(name_variablex(k, i))
        rows.append([variables,coefficients])

    solver.linear_constraints.add(lin_expr = rows, senses = senses, rhs = rhs, names = names)

def create_start_constraint(instance, solver):
    # Tour start at depot

    # Create auxiliary vectors
    rows = []
    senses = []
    rhs = []
    names = []

    # Create tour start constraint
    senses.append('E')
    names.append('str')
    coefficients = []
    variables = []
    rhs.append(1)

    for k in instance.nodes:
        coefficients.append(1)
        variables.append(name_variablex(1,k))
    rows.append([variables,coefficients])

    solver.linear_constraints.add(lin_expr = rows, senses = senses, rhs = rhs, names = names)

def create_end_constraint(instance, solver):
    # Tour end at depot

    # Create auxiliary vectors
    rows = []
    senses = []
    rhs = []
    names = []

    # Create tour end constraint
    senses.append('E')
    names.append('dne')
    coefficients = []
    variables = []
    rhs.append(1)

    for k in instance.nodes:
        coefficients.append(1)
        variables.append(name_variablex(k,1))
    rows.append([variables,coefficients])

    solver.linear_constraints.add(lin_expr = rows, senses = senses, rhs = rhs, names = names)

def create_ordering_constraint(instance, solver):
    # Ordering constraints
    # t_j - t_i \geq 1 + (1 - x_ij) M

    # Create auxiliary vectors
    rows = []
    senses = []
    rhs = []
    names = []

    M = instance.n_nodes

    # Create ordering constraints for each arc (i,j)
    for i in instance.nodes:
        for j in instance.nodes:
            # Avoid self loop variables
            # Avoid also the depot
            if j != 1 and i != j:
                senses.append('G')
                names.append('tmp_' + str(i) + '_' + str(j))
                coefficients = []
                variables = []
                rhs.append(1 - M)
                
                coefficients.append(1)
                variables.append(name_variablet(j))
                coefficients.append(-1)
                variables.append(name_variablet(i))
                coefficients.append(-1 *  M)
                variables.append(name_variablex(i, j))

                rows.append([variables,coefficients])

    solver.linear_constraints.add(lin_expr = rows, senses = senses, rhs = rhs, names = names)

def create_restriction(instance, solver, size):
    # Tour size constraint

    # Create auxiliary vectors
    rows = []
    senses = []
    rhs = []
    names = []

    # Create tour size constraint    
    senses.append('L')
    names.append('rst')
    coefficients = []
    variables = []
    rhs.append(size)

    for i in instance.nodes:
        for j in instance.nodes:
            coefficients.append(1)
            variables.append(name_variablex(i, j))
    rows.append([variables,coefficients])

    solver.linear_constraints.add(lin_expr = rows, senses = senses, rhs = rhs, names = names)

def cut_infeasible(solver, solution, identifier):
    # Cut an infeasible solution

    # Create auxiliary vectors
    rows = []
    senses = []
    rhs = []
    names = []

    # Create infeasible solution cut    
    senses.append('L')
    names.append('inf_' + str(identifier))
    coefficients = []
    variables = []

    # Parse the solution vector accordingly    
    index = 0
    # Until the depot appears, add x variables to the constraint
    while solution[index + 1] != 1:
        coefficients.append(1)
        variables.append(name_variablex(solution[index], solution[index + 1]))
        index += 1
    rhs.append(index - 1)
    rows.append([variables,coefficients])

    solver.linear_constraints.add(lin_expr = rows, senses = senses, rhs = rhs, names = names)

def cut_feasible(solver, solution, identifier):
    # Cut a feasible solution

    # Create auxiliary vectors
    rows = []
    senses = []
    rhs = []
    names = []

    # Create feasible solution cut    
    senses.append('L')
    names.append('fea_' + str(identifier))
    coefficients = []
    variables = []

    # Parse the solution vector accordingly    
    index = 0
    # Add x variables to constraint
    while solution[index + 1] != 1:
        coefficients.append(1)
        variables.append(name_variablex(solution[index], solution[index + 1]))
        index += 1
    coefficients.append(1)
    variables.append(name_variablex(solution[index], solution[index + 1]))
    rhs.append(index)
    rows.append([variables,coefficients])

    solver.linear_constraints.add(lin_expr = rows, senses = senses, rhs = rhs, names = names)

def cut_unreachable(instance, solver):
    # Cut (likely) unreachable nodes and arcs

    # Create auxiliary vectors
    rows = []
    senses = []
    rhs = []
    names = []

    counter = 0

    # Create unreachable cut
    coefficients = []
    variables = []
    senses.append('L')
    names.append('unr')
    rhs.append(0)

    # Parse the instance nodes accordingly
    for i in instance.nodes:
        for j in instance.nodes:

            # True if it is not possible to reach the node straight from depot
            first = instance.times[0][i-1] > instance.closing[i-1]
            # True if it is not possible to reach the depot straight from node
            second = instance.opening[i-1] + instance.times[i-1][0] > instance.maximum
            # True if leaving the earliest from node i cannot reach node j in time
            third = instance.opening[i-1] + instance.times[i-1][j-1] > instance.closing[j-1]

            # If the node has an unreachable property, cut it
            # If the arc has an unreachable property, cut it
            if first or second or third:
                counter += (first or second)
                variable = name_variablex(i, j)
                if variable not in variables:
                    variables.append(variable)
                    coefficients.append(1)

    # Calculation to obtain number of unreachable nodes
    counter = counter/instance.n_nodes

    rows.append([variables,coefficients])

    solver.linear_constraints.add(lin_expr = rows, senses = senses, rhs = rhs, names = names)
    return counter

def relax_unreachable(instance, solver, step):
    # Relax unreachable cut

    print('Relaxing the unreachable cut constraint...')
    rhs = solver.linear_constraints.get_rhs('unr')
    solver.linear_constraints.set_rhs('unr', rhs + step)


def build_model(instance):
    # Build model P(s)

    # Create solver instance
    solver = cp.Cplex()

    # Set solver parameters
    solver.objective.set_sense(solver.objective.sense.maximize)
    # solver.parameters.mip.tolerances.mipgap.set(0.1)
    solver.set_log_stream(None)
    solver.set_results_stream(None)

    # Create variables and constraints
    create_variables(instance, solver)
    create_flow_constraint(instance, solver)
    create_depart_constraint(instance, solver)
    create_arrival_constraint(instance, solver)
    create_start_constraint(instance, solver)
    create_end_constraint(instance, solver)
    create_ordering_constraint(instance, solver)
    
    # Create unreachable cuts according to the instance
    unreachable = cut_unreachable(instance, solver)

    return solver, unreachable

def run_model(instance, solver, size, path = 'dummy.lp'):
    # Run model P(s)

    # Add tour size constraint
    create_restriction(instance, solver, size)

    # Run the solver
    try:
        solver.solve()
    except:
        return {}

    # Retrieve the solution
    objective = solver.solution.get_objective_value()
    solution = {variable: value for (variable, value) in 
        zip(solver.variables.get_names(), solver.solution.get_values())}
    # print('Objective: ', objective)

    # Export the model
    solver.write(path)

    # Remove tour size constraint
    solver.linear_constraints.delete('rst')

    objective = round(objective, 4)

    return objective, solution


def check_performance(instance, solution, iterations = 10 ** 4, flag = False):
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

    # Print relevant information
    if flag:
        print('Solution: ', solution)
        print('Performance for', iterations, 'iterations')
        print('Objective: ', avg_objective)
        print('Time: ', avg_time)
        print('Reward: ', avg_reward)
        print('Penalty: ', avg_penalty)
        print('Percentage: ', percentage)

    avg_objective = round(avg_objective, 4)
    avg_reward = round(avg_reward, 4)
    avg_penalty = round(avg_penalty, 4)

    return avg_objective, avg_reward, avg_penalty, percentage


def format_solution(instance, solution):
    # Format solution from model P(s)

    # Create auxiliary dictionary
    temporary = {}
    for variable, value in solution.items():
        if value > 0.1 and 'x' in variable:
            _, o, d = variable.split('_')
            o = int(o)
            d = int(d)
            temporary[o] = d

    # Parse the auxiliary dictionary
    formatted = []
    index = 1
    formatted.append(1)
    while temporary[index] != 1:
        formatted.append(temporary[index])
        index = temporary[index]
    formatted.append(1)

    # Add remaining nodes to the solution
    for i in instance.nodes:
        if i not in formatted:
            formatted.append(i)

    return formatted

def adapt_weights(instance, solver, history):
    # Adapt objective coefficients based on historical data

    # Export historical data to a file
    with open('historical.txt', 'w') as output:
        for entry in history:
            output.write('{} - {}\n'.format(entry[1], entry[0]))

    # Create auxiliary vectors
    updates = []

    # Calculate new weights for each arc (i,j)
    for i in instance.nodes:
        for j in instance.nodes:
            if i != j:

                # Retrieve the name and the coefficient of arc (i, j)
                variable = name_variablex(i, j)
                weight = calculate_weight(instance, i, j)

                counter = 0
                additive = 0

                for execution in history:
                    if check_membership(execution[0], i, j):
                        # Save how many tours have this arc
                        counter += 1
                        # Retrieve the size of the current tour
                        size = execution[0][1:].index(1) + 1
                        # Distribute the penalty equally among arcs
                        additive += execution[1]/size

                # If penalty is less than zero, adapt weights
                if additive < 0 and counter > 0:
                    weight += additive/counter
                    updates.append((variable, weight))

    solver.objective.set_linear(updates)

def check_membership(solution, i, j):
    # Check membership of an arc to a certain solution

    index = 0
    finished = False

    while not finished:
        if solution[index] == i and solution[index + 1] == j:
            return True
        if index != 0 and solution[index + 1] == 1:
            finished = True
        else:
            index += 1
    return False

def calculate_bound(instance, size):

    solver, unreachable = build_model(instance)
    bound, _ = run_model(instance, solver, size, 'bound.lp')

    bound = round(bound, 4)

    return bound

def tracker_approach(instance, iterations = 10 ** 3, mode = 'w', threshold = 0.8, tolerance = 0.05):
    # Run tracker approach

    assert mode in ['w', 'x', 'a', 'r']

    # Global variables
    best_solution = []
    best_objective = -1 * np.inf

    # Estimate times
    if mode == 'w':
        weights = 1
    elif mode == 'x':
        weights = 0.9
    elif mode == 'a':
        weights = 0.5
    else:
        weights = np.random.rand(instance.n_nodes, instance.n_nodes)
    instance.times = weights * instance.adj

    # Build model P(s)
    solver, unreachable = build_model(instance)

    # Iterative approach
    size = 2
    counter = 0
    feasible = True

    # Historical data
    history = []

    # Maximum size of the tour
    maximum = instance.n_nodes + 1
    # Frontier size of the tour
    frontier = maximum - unreachable 

    # Save start time
    start = tm.time()

    # Iterate until reaching maximum number of iterations or maximum size of the tour or the model is no longer feasible
    while counter < iterations and size < maximum and feasible:

        # Obtain solution from model with current size
        _, solution = run_model(instance, solver, size, 'model.lp')
        #print('Raw solution: ', solution)

        # If the model remains feasible, keep runing the iterative approach
        feasible = len(solution) != 0

        if feasible:

            # Format solution in an understandable manner
            solution = format_solution(instance, solution)
            # print('Formatted solution: ', solution)
            bound = calculate_bound(instance, size)
            
            # Check solution performance
            objective, reward, penalty, percentage = check_performance(instance, solution, 100)
            history.append([solution, penalty])
            
            # Store current solution if it is the best one yet
            if objective >= best_objective:
                best_solution = solution
                best_objective = objective
            
            # If the solution is infeasible most of the time, cut infeasible solution
            if percentage < threshold:
                # print('Solution infeasible')
                cut_infeasible(solver, solution, counter)
            else:
                # print('Solution feasible')
                cut_feasible(solver, solution, counter)

            # Relax unreachable cut according to slack
            if size > frontier:
                relax_unreachable(instance, solver, 1)            
            
            adapt_weights(instance, solver, history)

            gap = (bound - best_objective) / bound
            if gap < tolerance:
                size += 1
            else:
                print('Bound not enoughly closed...')

            counter += 1
            print('Iteration #{}: {} [Objective: {}, Size: {}, Bound: {}]'.format(counter, best_solution, best_objective, size, bound))
        
        # If the model no longer feasible, log information and output the best solution
        else:
            print('The model is no longer feasible due to the tracked information')

    # Save end time
    end = tm.time()

    # Performance summary
    print('Mode: {}'.format(mode))
    print('Feasible: {}'.format(feasible))
    print('Counter: {} out of {}'.format(counter, iterations))
    print('Size: {} out of {}'.format(size, maximum))
    print('Best objective: {}'.format(best_objective))
    print('Best solution: {}'.format(best_solution))
    print('Total time: {}'.format(end - start))

    export_solution(best_solution)

    return best_solution

def export_solution(solution):
    # Export solution to .out file

    name = str(ui.uuid4())[0:8] + '.out'
    with open('solutions/{}'.format(name), 'w') as output:
        for node in solution:
            output.write('{}\n'.format(node))

    print('Exported to file {}'.format(name))
    
    return name

def load_instance(identifier):
    # Load instance from file

    instance = env.Env(from_file = True,  
        x_path = 'data/valid/instances/{}.csv'.format(identifier), 
        adj_path = 'data/valid/adjs/adj-{}.csv'.format(identifier))

    return instance

def load_validation():
    # Load validation instance

    instance = env.Env(55, seed = 3119615)

    return instance

def adjust_instance(instance):
    # Perform instance adjustements

    instance.nodes = list(range(1, instance.n_nodes + 1))
    instance.rewards = instance.x[:, -2]
    instance.maximum = instance.x[:, -1][1]
    instance.opening = instance.x[:, -4]
    instance.closing = instance.x[:, -3]

    return instance

if __name__ == "__main__":
    # instance = load_instance('instance0001')
    instance = load_validation()
    instance = adjust_instance(instance)
    solution = tracker_approach(instance, 3 * 10 ** 3)