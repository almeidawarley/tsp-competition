import sys
import numpy as np
import cplex as cp
import time as tm
import main as mn

def name_x(i, j):
    """
    Name variable x for arc (i, j)
    """

    return 'x_' + str(i) + '_' + str(j)

def name_t(i):
    """
    Name variable t for node i
    """

    return 't_' + str(i)

def retrieve_coefficient(instance, i, j):
    """
    Retrieve (deterministic) coefficient for arc (i, j)
    """

    # If it is a self loop, set coefficient to zero
    if i == j:
        return 0
    # Otherwise, set coefficient to the reward of node j
    else:
        return instance.rewards[j - 1]

def create_variables(instance, solver):
    """
    Create decision variables for the surrogate model
    """

    # Create auxiliary vectors
    names = []
    coefficients = []
    types = ['B' for i in instance.nodes for j in instance.nodes]
    uppers = [1 for i in instance.nodes for j in instance.nodes]
    lowers = [0 for i in instance.nodes for j in instance.nodes]

    # Create variable x per arc (i, j)
    for i in instance.nodes:
        for j in instance.nodes:
            names.append(name_x(i, j))
            coefficients.append(retrieve_coefficient(instance, i, j))

    solver.variables.add(obj = coefficients, ub = uppers, lb = lowers, types = types, names = names)

    # Create auxiliary vectors
    M = instance.n_nodes
    names = []
    coefficients = []
    types = ['C' for i in instance.nodes]
    uppers = [M for i in instance.nodes]
    lowers = [0 for i in instance.nodes]

    # Create variable t per node i
    for i in instance.nodes:
        names.append(name_t(i))
        coefficients.append(0)

    solver.variables.add(obj = coefficients, ub = uppers, lb = lowers, types = types, names = names)

def create_flow_constraint(instance, solver):
    """
    Flow constraint at nodes
    """

    # Create auxiliary vectors
    rows = []
    senses = []
    rhs = []
    names = []

    # Create flow constraint per node i
    for i in instance.nodes:
        senses.append('E')
        names.append('flw_' + str(i))
        rhs.append(0)
        coefficients = []
        variables = []
        for k in instance.nodes:
            # Ignore self loops
            if i != k:
                coefficients.append(1)
                variables.append(name_x(i, k))
                coefficients.append(-1)
                variables.append(name_x(k, i))
        rows.append([variables,coefficients])

    solver.linear_constraints.add(lin_expr = rows, senses = senses, rhs = rhs, names = names)

def create_depart_constraint(instance, solver):
    """
    Depart constraint from nodes
    """

    # Create auxiliary vectors
    rows = []
    senses = []
    rhs = []
    names = []

    # Create depart constraint per node i
    for i in instance.nodes:
        senses.append('L')
        names.append('dpt_' + str(i))
        rhs.append(1)
        coefficients = []
        variables = []
        for k in instance.nodes:
            coefficients.append(1)
            variables.append(name_x(i, k))
        rows.append([variables,coefficients])

    solver.linear_constraints.add(lin_expr = rows, senses = senses, rhs = rhs, names = names)

def create_arrival_constraint(instance, solver):
    """
    Arrival constraint at nodes
    """

    # Create auxiliary vectors
    rows = []
    senses = []
    rhs = []
    names = []

    # Create arrival constraint per node i
    for i in instance.nodes:
        senses.append('L')
        names.append('arr_' + str(i))
        rhs.append(1)
        coefficients = []
        variables = []
        for k in instance.nodes:
            coefficients.append(1)
            variables.append(name_x(k, i))
        rows.append([variables,coefficients])

    solver.linear_constraints.add(lin_expr = rows, senses = senses, rhs = rhs, names = names)

def create_start_constraint(instance, solver):
    """
    Start constraint at the depot
    """

    # Create auxiliary vectors
    rows = []
    senses = []
    rhs = []
    names = []

    # Create start constraint at depot
    senses.append('E')
    names.append('str')
    rhs.append(1)
    coefficients = []
    variables = []
    for k in instance.nodes:
        coefficients.append(1)
        variables.append(name_x(1, k))
    rows.append([variables,coefficients])

    solver.linear_constraints.add(lin_expr = rows, senses = senses, rhs = rhs, names = names)

def create_end_constraint(instance, solver):
    """
    End constraint at the depot
    """

    # Create auxiliary vectors
    rows = []
    senses = []
    rhs = []
    names = []

    # Create end constraint at depot
    senses.append('E')
    names.append('dne')
    rhs.append(1)
    coefficients = []
    variables = []
    for k in instance.nodes:
        coefficients.append(1)
        variables.append(name_x(k, 1))
    rows.append([variables,coefficients])

    solver.linear_constraints.add(lin_expr = rows, senses = senses, rhs = rhs, names = names)

def create_order_constraint(instance, solver):
    """
    Order constraint for arcs
    """

    # Create auxiliary vectors
    rows = []
    senses = []
    rhs = []
    names = []

    M = instance.n_nodes

    # Create order constraint per arc (i, j)
    for i in instance.nodes:
        for j in instance.nodes:
            # Ignore self loops
            # Ignore return to depot
            if i != j and j != 1:
                senses.append('G')
                names.append('ord_' + str(i) + '_' + str(j))
                rhs.append(1 - M)
                coefficients = []
                variables = []
                coefficients.append(1)
                variables.append(name_t(j))
                coefficients.append(-1)
                variables.append(name_t(i))
                coefficients.append(-1 *  M)
                variables.append(name_x(i, j))
                rows.append([variables,coefficients])

    solver.linear_constraints.add(lin_expr = rows, senses = senses, rhs = rhs, names = names)

def create_size_constraint(instance, solver, size):
    """
    Route size constraint
    """

    # Create auxiliary vectors
    rows = []
    senses = []
    rhs = []
    names = []

    # Create route size constraint
    senses.append('L')
    names.append('siz')
    rhs.append(size)
    coefficients = []
    variables = []
    for i in instance.nodes:
        for j in instance.nodes:
            coefficients.append(1)
            variables.append(name_x(i, j))
    rows.append([variables,coefficients])

    solver.linear_constraints.add(lin_expr = rows, senses = senses, rhs = rhs, names = names)

def cut_infeasible(solver, route, identifier):
    """
    Cut for an infeasible solution
    """

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
    index = 0
    while route[index + 1] != 1:
        # Until the depot appears, add arc (i,j) to the cut
        coefficients.append(1)
        variables.append(name_x(route[index], route[index + 1]))
        index += 1
    rhs.append(index - 1)
    rows.append([variables,coefficients])

    solver.linear_constraints.add(lin_expr = rows, senses = senses, rhs = rhs, names = names)

def cut_feasible(solver, route, identifier):
    """
    Cut for a feasible solution
    """

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
    index = 0
    while route[index + 1] != 1:
        # Until the depot appears, add arc (i,j) to the cut
        coefficients.append(1)
        variables.append(name_x(route[index], route[index + 1]))
        index += 1
    # Add return to the depot to the cut
    coefficients.append(1)
    variables.append(name_x(route[index], route[index + 1]))
    rhs.append(index)
    rows.append([variables,coefficients])

    solver.linear_constraints.add(lin_expr = rows, senses = senses, rhs = rhs, names = names)

def cut_impossible(instance, solver):
    """
    Cut for (certainly) impossible nodes
    """

    # Create auxiliary vectors
    rows = []
    senses = []
    rhs = []
    names = []

    # Find impossible nodes
    impossible = []
    for i in instance.nodes:
        if i != 1:
            route = [1, i, 1]
            for j in instance.nodes:
                if j not in route:
                    route.append(j)
            score, _, _, _ = mn.check_performance(instance, route)  
            if score <= 0:
                impossible.append(i)

    # Create unreachable cut
    senses.append('L')
    names.append('imp')
    rhs.append(0)
    coefficients = []
    variables = []
    for i in impossible:
        for k in instance.nodes:
            variable = name_x(i, k)
            if variable not in variables:
                variables.append(variable)
                coefficients.append(1)
            variable = name_x(k, i)
            if variable not in variables:
                variables.append(variable)
                coefficients.append(1)
    rows.append([variables,coefficients])

    solver.linear_constraints.add(lin_expr = rows, senses = senses, rhs = rhs, names = names)

    print('There are {} impossible nodes: {}'.format(len(impossible), impossible))

    return impossible

def cut_unreachable(instance, solver):
    """
    Cut for (likely) unreachable arcs
    """

    # Create auxiliary vectors
    rows = []
    senses = []
    rhs = []
    names = []

    unreachable = []
    # Create unreachable cut
    senses.append('L')
    names.append('unr')
    rhs.append(0)
    coefficients = []
    variables = []
    for i in instance.nodes:
        for j in instance.nodes:
            # If leaving the earliest from node i cannot reach node j in time, cut arc (i, j)
            if instance.opening[i-1] + instance.times[i-1][j-1] > instance.closing[j-1]:
                variables.append(name_x(i, j))
                coefficients.append(1)
                unreachable.append([i, j])
    rows.append([variables,coefficients])

    solver.linear_constraints.add(lin_expr = rows, senses = senses, rhs = rhs, names = names)

    # print('There are {} unreachable arcs: {}'.format(len(unreachable), unreachable))

    return unreachable

def build_model(instance):
    """
    Build the surrogate model for the tracker approach
    """ 

    # Create solver instance
    solver = cp.Cplex()

    # Set solver parameters
    solver.objective.set_sense(solver.objective.sense.maximize)
    # solver.parameters.mip.tolerances.mipgap.set(0.1)
    # solver.parameters.threads.set(1)
    solver.set_results_stream(None)
    solver.set_log_stream(None)

    # Create variables and constraints
    create_variables(instance, solver)
    create_flow_constraint(instance, solver)
    create_depart_constraint(instance, solver)
    create_arrival_constraint(instance, solver)
    create_start_constraint(instance, solver)
    create_end_constraint(instance, solver)
    create_order_constraint(instance, solver)
    
    # Create impossible cuts
    impossible = cut_impossible(instance, solver)
    # Create unreachable cuts
    unreachable = cut_unreachable(instance, solver)

    return solver, len(impossible)

def run_model(instance, solver, size, path = 'dummy.lp'):
    """
    Run the surrogate model for the tracker approach
    """ 

    # Add route size constraint
    create_size_constraint(instance, solver, size)

    try:
        # Run the solver
        solver.solve()
        # Retrieve the solution
        objective = solver.solution.get_objective_value()
        solution = {variable: value for (variable, value) in 
            zip(solver.variables.get_names(), solver.solution.get_values())}
    except:
        return -1 * np.inf, {}    

    # Export the surrogate model
    solver.write(path)

    # Remove route size constraint
    solver.linear_constraints.delete('siz')

    objective = round(objective, 10)

    return objective, solution

def format_solution(instance, solution):
    """
    Format solution from the surrogate model as a route
    """

    # Create auxiliary dictionary
    temporary = {}
    for variable, value in solution.items():
        if value > 0.1 and 'x' in variable:
            _, i, j = variable.split('_')
            i = int(i)
            j = int(j)
            temporary[i] = j

    # Parse auxiliary dictionary
    route = []
    route.append(1)
    index = 1
    while temporary[index] != 1:
        route.append(temporary[index])
        index = temporary[index]
    route.append(1)

    # Add remaining nodes to the route
    for i in instance.nodes:
        if i not in route:
            route.append(i)

    return route

def adapt_coefficients(instance, solver, history, route, penalty):
    """
    Adapt coefficients of the surrogate model based on historical data
    """

    # Create auxiliary vectors
    updates = []

    # Parse arcs in the route
    arcs = mn.retrieve_arcs(route)
    for i, j in arcs:
        history[i][j]['weights'] += penalty / len(arcs)
        history[i][j]['occurrences'] += 1
        if history[i][j]['weights'] < 0:
            coefficient = retrieve_coefficient(instance, i, j)
            coefficient += history[i][j]['weights'] / history[i][j]['occurrences']
            variable = name_x(i, j)
            updates.append((variable, coefficient))
    
    solver.objective.set_linear(updates)

def calculate_bound(instance, solver, size):
    """
    Calculate upper bound for a route size
    """

    variables = solver.variables.get_names()
    coefficients = solver.objective.get_linear()

    # Restore coefficients to standard values
    updates = []
    for i in instance.nodes:
        for j in instance.nodes:
            updates.append((name_x(i, j), retrieve_coefficient(instance, i, j)))

    solver.objective.set_linear(updates)

    # Retrieve a bound value
    bound, _ = run_model(instance, solver, size, 'bound.lp')

    # Restore coefficients to updated values
    updates = []
    for index, _ in enumerate(variables):
        updates.append((variables[index], coefficients[index]))

    solver.objective.set_linear(updates)
    
    bound = round(bound, 10)

    return bound

def tracker_approach(instance, iterations = 10 ** 3, simulations = 100, feasibility_threshold = 0.8, gap_threshold = 0.05):
    """
    Run the surrogate-based tracker approach
    """

    # Estimate travel times        
    instance.times = instance.adj

    # Build the surrogate model
    solver, blocked = build_model(instance)    

    # Historical data    
    history = {}
    for i in instance.nodes:
        history[i] = {}
        for j in instance.nodes:
            history[i][j] = {}
            history[i][j]['weights'] = 0
            history[i][j]['occurrences'] = 0

    # Approach variables
    best_route = []
    best_score = -1 * np.inf
    size = 2
    counter = 0
    feasible = True
    # Maximum route size
    maximum = instance.n_nodes + 1 - blocked

    # Calculate initial upper bound
    bound = calculate_bound(instance, solver, size)

    # Save start time
    start = tm.time()

    # Iterate until (a) maximum number of iterations, (b) the gap has been closed, (c) the surrogate model is no longer feasible
    while counter < iterations and best_score < bound and feasible:

        # Obtain approximate solution from the surrogate model
        approx, solution = run_model(instance, solver, size, 'model.lp')
        #print('Solution: ', solution)

        feasible = len(solution) != 0

        # Keep running if the surrogate model remains feasible
        if feasible:

            # Format solution as a route
            route = format_solution(instance, solution)
            # print('Route: ', route)
            
            # Check route performance
            score, reward, penalty, percentage = mn.check_performance(instance, route, simulations)
            
            # Store current route if it is the best one yet
            if score >= best_score:
                best_route = route
                best_score = score
            
            # If the route is infeasible most of the time, cut infeasible solution
            if percentage < feasibility_threshold:
            # if score < best_score / 2:
                # print('Route infeasible')
                cut_infeasible(solver, route, counter)
            # Otherwise, cut feasible solution because it has been visited already
            else:
                # print('Route feasible')
                cut_feasible(solver, route, counter)
            
            # Adapt coefficients based on historical data
            adapt_coefficients(instance, solver, history, route, penalty)

            # Calculate gap based on the best route
            gap = (bound - best_score) / bound
            if gap < gap_threshold and size < maximum:
                size += 1
                # Calculate new upper bound
                bound = calculate_bound(instance, solver, size)
            else:
                size += 0

            # Print summary of the iteration
            counter += 1
            print('Candidate route at iteration #{}: {} [Score: {}, Approximation: {}]'
                .format(counter, route, score, approx))
            print('Superior route at iteration #{}: {} [Score: {}, Size: {}, Bound: {}]'
                .format(counter, best_route, best_score, size, bound))
        
        # End algorithm if the surrogate model is no longer feasible
        else:
            print('The surrogate model is no longer feasible due to the tracked information')

    # Save end time
    end = tm.time()

    # Calculate running time
    total_time = round(end - start, 2)

    # Export route to a .out file
    path = mn.export_route(best_route)

    # Print summary of the algorithm
    print('> Parameters:')
    print('| # iterations (K): {}'.format(iterations))
    print('| # simulations (M): {}'.format(simulations))
    print('| Feasibility threshold (f): {}'.format(feasibility_threshold))
    print('| Gap threshold (g): {}'.format(gap_threshold))
    print('> Variables:')
    print('| Iteration counter (k): {}'.format(counter))
    print('| Bound for size {}(u^k): {}'.format(size, bound))
    print('| Current route size (s): {}'.format(size))
    print('| Maximum route size: {}'.format(maximum))
    print('> Results:')
    print('| Best route: {}'.format(best_route))
    print('| Best score: {}'.format(best_score))
    print('| Total time: {}'.format(total_time))
    print('| Route file: {}'.format(path))

    return best_route

if __name__ == "__main__":

    if len(sys.argv) > 1:
        instance = mn.load_instance(sys.argv[1])
    else:
        # instance = mn.load_validation()
        instance = mn.load_competition()
    instance = mn.adjust_instance(instance)
    route = tracker_approach(instance, 
        iterations = 1000,
        simulations = 100,
        feasibility_threshold = 0.8,
        gap_threshold = 0.05)