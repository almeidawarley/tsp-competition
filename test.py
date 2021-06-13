import env
import os

def check_solution(instance, solution, flag = False):
    # Simulate solution
    time, reward, penalty, feasibility = instance.check_solution(solution)
    if flag:
        print('Time: ', time)
        print('Reward: ', reward)
        print('Penalty: ', penalty)
        print('Feasibility: ', feasibility)
    return time, reward, penalty, feasibility

# instance = env.Env(from_file = True,  
#    x_path = 'data/valid/instances/instance0001.csv', 
#    adj_path = 'data/valid/adjs/adj-instance0001.csv')
instance = env.Env(55, seed=3119615)

folder = 'solutions/'

runs = 10 ** 3
for entry in os.listdir(folder):
    if '.out' in entry:
        solution = []
        path = os.path.join(folder, entry)
        with open(path) as content:
            for line in content:
                line = line.replace(',', '')
                solution.append(int(line))
        print('Path: ', path)
        print('Solution: ', solution)
        average = 0
        counter = 0
        while counter < runs:
            _, reward, penalty, _ = check_solution(instance, solution)
            average += reward + penalty
            counter += 1
        print('Objective: ', average/runs)
