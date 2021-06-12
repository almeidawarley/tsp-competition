import env

def simulate_solution(instance, solution, flag = True):
    # Simulate solution
    time, reward, penalty, feasibility = instance.check_solution(solution)
    if flag:
        print('Solution: ', solution)
        print('Time: ', time)
        print('Reward: ', reward)
        print('Penalty: ', penalty)
        print('Feasibility: ', feasibility)
    return reward, feasibility

instance = env.Env(from_file = True,  
    x_path = 'data/valid/instances/instance0001.csv', 
    adj_path = 'data/valid/adjs/adj-instance0001.csv')
solutions = [
    [1, 19, 3, 14, 20, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18],
    [1, 11, 7, 16, 2, 13, 5, 1, 3, 4, 6, 8, 9, 10, 12, 14, 15, 17, 18, 19, 20]
]
runs = 1

for solution in solutions:
    simulate_solution(instance, solution)