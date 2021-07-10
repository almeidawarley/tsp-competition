import env
import uuid as ui

def check_performance(instance, route, simulations = 10 ** 4, flag = False):
    """
    Check the performance of a route
    """

    # Create average variables
    avg_time = 0
    avg_reward = 0
    avg_penalty = 0
    avg_feasibility = 0

    counter = 0
    while counter < simulations:

        # Call black-box simulator
        time, reward, penalty, feasible = instance.check_solution(route)

        # Update average variables
        avg_time += time
        avg_reward += reward
        avg_penalty += penalty
        avg_feasibility += 1 if feasible else 0

        counter += 1

    avg_time /= simulations
    avg_reward /= simulations
    avg_penalty /= simulations
    avg_feasibility /= simulations
    avg_score = avg_reward + avg_penalty

    # Print relevant information if flag is true
    if flag:
        print('Route: ', route)
        print('Simulations: ', simulations)
        print('Average score: ', avg_score)
        print('Average time: ', avg_time)
        print('Average reward: ', avg_reward)
        print('Average penalty: ', avg_penalty)
        print('Average feasibility: ', avg_feasibility)

    avg_score = round(avg_score, 10)
    avg_reward = round(avg_reward, 10)
    avg_penalty = round(avg_penalty, 10)

    return avg_score, avg_reward, avg_penalty, avg_feasibility

def retrieve_arcs(route):
    """
    Retrieve a list of arcs from a route
    """

    arcs = []
    # Parse route until the depot appears
    index = 0
    while route[index + 1] != 1:
        # Append current arc accordingly
        arcs.append([route[index], route[index + 1]])
        index += 1
    # Append final arc to the depot
    arcs.append([route[index], route[index + 1]])

    return arcs

def export_route(route):
    """
    Export a route to a .out file
    """

    # Create unique identifier for .out file
    name = str(ui.uuid4())[0:8] + '.out'
    # Write route information to .out file
    with open('solutions/{}'.format(name), 'w') as output:
        for node in route:
            output.write('{}\n'.format(node))
    
    return name

def load_instance(identifier):
    """
    Load instance from file
    """

    instance = env.Env(from_file = True,  
        x_path = 'data/valid/instances/{}.csv'.format(identifier), 
        adj_path = 'data/valid/adjs/adj-{}.csv'.format(identifier))

    return instance

def load_validation():
    """
    Load validation instance
    """

    instance = env.Env(55, seed = 3119615)

    return instance

def load_competition():
    """
    Load competition instance (i.e., test instance)
    """

    instance = env.Env(65, seed = 6537855)
    
    return instance

def adjust_instance(instance):
    """
    Adjust instance
    """

    instance.nodes = list(range(1, instance.n_nodes + 1))
    instance.rewards = instance.x[:, -2]
    instance.maximum = instance.x[:, -1][1]
    instance.opening = instance.x[:, -4]
    instance.closing = instance.x[:, -3]

    return instance