import main as mn
import env
import os

instance = mn.load_validation()
instance = mn.adjust_instance(instance)

folder = 'solutions/'
ref_objective = 5
ref_percentage = 0.75

print('Listing solutions with objective greater than {} and feasible at least {}%'
    .format(round(ref_objective, 2), round(100*ref_percentage, 0)))

for entry in os.listdir(folder):
    if '.out' in entry:
        solution = []
        path = os.path.join(folder, entry)
        with open(path) as content:
            for line in content:
                line = line.replace(',', '')
                solution.append(int(line))
        objective, reward, percentage = mn.check_performance(instance, solution)
        if objective >= ref_objective and percentage >= ref_percentage:
            print('Solution in {} has objective {} and is feasible {}% of the time'
                .format(path, round(objective, 2), round(100*percentage,0)))
