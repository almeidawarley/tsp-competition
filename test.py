import main as mn
import env
import os
import sys

# Load validation instance
instance = mn.load_validation()
instance = mn.adjust_instance(instance)

# Set important parameters
folder = 'solutions/'
ref_objective = 5
ref_percentage = 0.75
ref_solution = ''

# If there is an argument, output information for the solution specified in the argument
if len(sys.argv) > 1:
    ref_solution = sys.argv[1]
    print('Listing information for solution {}'.format(ref_solution))
# If there is not an argument, output information for solution with certain characteristics
else:
    print('Listing all solutions with objective greater than {} and feasible at least {}%'
        .format(round(ref_objective, 2), round(100*ref_percentage, 0)))

# Parse files in the solution directory
for entry in os.listdir(folder):
    if '.out' in entry and ref_solution in entry:
        
        # Read solution from file
        solution = []
        path = os.path.join(folder, entry)
        with open(path) as content:
            for line in content:
                line = line.replace(',', '')
                solution.append(int(line))

        # Check performance for 10^4 iterations
        objective, reward, penalty, percentage = mn.check_performance(instance, solution)

        # If solution has some characteristics or has been specificed, print information
        if (objective >= ref_objective and percentage >= ref_percentage) or ref_solution != '' :
            print('Solution: {} \nObjective: {} \nFeasibility: {}% \nReward: {} \nPenalty: {}'
                .format(path, round(objective, 6), round(100*percentage, 2), round(reward, 6), round(penalty, 6)))
