# AI for TSP Competition
# Repository of the team Margaridinhas for the surrogate track (track 1)

## Generation of the solutions with the highest score
The generation of the set of solutions with the highest score was done in two steps. First, the tracker approach has been called with a few different parameters to generate different good solutions for the test instance. More specifically, the good solutions listed below, which are available in the [solutions](https://github.com/almeidawarley/tsp_competition/blob/master/solutions) folder, have been generated through the tracker approach with the following parameters.

```
4555df95.out: {simulations: 100, iterations: 1000, feasibility_threshold: 0.2, gap_threshold: 0.05, score: 9.8333}
6335cd16.out: {simulations: 1000, iterations: 1000, feasibility_threshold: 0.6, gap_threshold: 0.1, score: 10.232395}
07ea8acd.out: {simulations: 100, iterations: 1000, feasibility_threshold: 0.9, gap_threshold: 0.05, score: 10.625693}
21c21320.out: {simulations: 100, iterations: 1000, feasibility_threshold: 0.4, gap_threshold: 0.05, score: 10.661035}
9274c720.out: {simulations: 100, iterations: 1000, feasibility_threshold: 0.7, gap_threshold: 0.05, score: 10.674276}
dd2b6c5d.out: {simulations: 100, iterations: 1000, feasibility_threshold: 0.5, gap_threshold: 0.05, score: 10.818903}
22cb4ff1.out: {simulations: 100, iterations: 1000, feasibility_threshold: 1, gap_threshold: 0.05, score: 10.819456}
cfdbad6a.out: {simulations: 1000, iterations: 1000, feasibility_threshold: 0.6, gap_threshold: 0.05, score: 10.835396}
6d651a12.out: {simulations: 1000, iterations: 1000, feasibility_threshold: 0.8, gap_threshold: 0.05, score: 10.84941}
fbd8cbfb.out: {simulations: 100, iterations: 1000, feasibility_threshold: 0.8, gap_threshold: 0.05, score: 11.039958}
1c116019.out: {simulations: 100, iterations: 1000, feasibility_threshold: 0.3, gap_threshold: 0.05, score: 11.066891}
c40d86d0.out: {simulations: 1000, iterations: 1000, feasibility_threshold: 0.8, gap_threshold: 0.1, score: 11.197023}
c923a5b2.out: {simulations: 100, iterations: 1000, feasibility_threshold: 0.6, gap_threshold: 0.05, score: 11.215753}
26633496.out: {simulations: 100, iterations: 10000, feasibility_threshold: 0.8, gap_threshold: 0.2, score: 11.282869}
```

The genetic algorithm will take every solution in the `.out` format that are found in a folder with name `warley` as a warm start. This mean we simply copied the files that had very good scores, listed above, from the [solutions](https://github.com/almeidawarley/tsp_competition/blob/master/solutions) folder into the `warley` folder. Here is an example :

```
cp solutions\4555df95.out warley\
```

Then, we inputed the following parameters in the main function of the `genetic_algo.py` file and ran it.

```
  # PARAMETERS
  nodes_num = 65
  seed = 6537855
  generation_num = 5
  population_num = 25600
  parents_num = 200
  warm_dico_sol_lb = 11.3
  monte_carlo = 100
  dico_filename = None
  save_under = time.strftime("%Y%m%d-%H%M%S") + "-env-" + str(nodes_num) + "-" + str(seed) \
    + "_pop-" + str(population_num) + "-" + str(parents_num) + "_gen-"
```

For subsequent runs, we initialised the dictionary parameter in order to take advantage of the previous runs : `dico_filename = os.path.join(os.getcwd(), "20210707-125553-env-65-6537855_pop-25600-200_gen-4.json")`.

## Tracker approach
Open [tracker.py](https://github.com/almeidawarley/tsp_competition/blob/master/tracker.py) and choose the desired value for the parameters of the tracker approach. There are four parameters: *iterations*, which determines the maximum number of iterations of the tracker approach; *simulations*, which determines the number of simulations per iteration; *feasibility_threshold*, which determines the feasibility threshold among a *simulations* number of simulations; and *gap_threshold*, which determines the gap threshold taking into consideration the upper bound at a certain iteration. The tracker approach loads by default the competition instance, i.e., the instance used in the test phase of the competition. However, it is possible to run it for other instances by changing the code accordingly. The tracker approach exports solutions to the [solutions](https://github.com/almeidawarley/tsp_competition/blob/master/solutions) folder. The details of this algorithm can be found in section 2 of the [documentation](https://github.com/almeidawarley/tsp_competition/blob/master/TSP_competition%20-%20detailed%20documentation%20-%20Margaridinhas.pdf).

## Genetic approach
Open [genetic_algo.py](https://github.com/almeidawarley/tsp_competition/blob/master/genetic_algo.py) and choose the desired value for the parameters of the genetic approach. 

```
Params for the algorithm
  nodes_num (int)          : number of nodes in the problem instance
  seed (int)               : random seed for the problem instance
  generation_num (int)     : number of population generations, used as stopping criterion (number of loops).
  population_num (int)     : number of individus in population (to be attained after the reproduction step).
  parents_num (int)        : number of parents in population (survivors after the selection step).
  warm_dico_sol_lb (float) : minimal score tolerated in order to consider a solution from the dico in the warmstart.
  monte_carlo (int)        : number of consecutive evaluations of a solution performed with check_solution during evaluation step.
  dico_filename (str)      : name of preexisting dico (put None if does not want to use one) to perform warmstart.
  save_under (str)         : name of files to save the next generations' dicos.
  mutation_proba (float)   : Percentage, probability that mutation occurs to an individu during the mutation step.
```

 A sample of solutions found by the genetic approach can be found in the [genetic_algo](https://github.com/almeidawarley/tsp_competition/blob/master/genetic_algo/) folder, together with the final dictionary used for the test phase. The genetic algorithm uses the [genetic_operators.py](https://github.com/almeidawarley/tsp_competition/blob/master/genetic_operators.py) and [mean_dico.py](https://github.com/almeidawarley/tsp_competition/blob/master/mean_dico.py) files. The former contains a class implementing all the genetic operators for the selection, reproduction and mutation steps, while the latter contains the utility functions necessary to save the solution check results. The latter also contains a very useful main function to find all solutions above a certain lower bound in the dictionary.

## Dynamical programming approach:
Open [dynamic_prog/dynamic.py](https://github.com/almeidawarley/tsp_competition/blob/master/dynamic_prog/dynamic.py) and choose the desired instance using variables *instance_number*, *use_validation_or_test* and *use_test*. Then, set variable *precomputed* to *False* to compute matrices E and TT for the first time for that particular instance, and that will be stored in subfolder *dynamic_prog/store_E_and_TT*. Once those tables are stored, it will be possible to run again the same method on the same instance with *precomputed* to *True* to avoid recomputing E and TT. Finally, the last parameter is the number of steps to look ahead in the algorithm, encoded in the variable *E_dimensions*. The details of this algorithm is detailed in the appendix of the [documentation](https://github.com/almeidawarley/tsp_competition/blob/master/TSP_competition%20-%20detailed%20documentation%20-%20Margaridinhas.pdf). The solutions will be stored in the subfolder *dynamic_prog/dyn_solutions*.

## Graphical analysis of solutions:
Open [graphical_analysis/analysis.py](https://github.com/almeidawarley/tsp_competition/blob/master/graphical_analysis/analysis.py) and choose the desired instance and solution in the main() (by changing the variables *instance_number*, *use_validation_or_test*, *use_test*, and *sol*). Then run the file. An image analysis.png will be created. Finally, in the same folder, you may find a script [graphical_analysis/study_opt_solutions.py](https://github.com/almeidawarley/tsp_competition/blob/master/graphical_analysis/study_opt_solutions.py) that would enable you to assess the differences among 554 different optimal solutions that could be extracted from the folder 'many11_32' and from the file 'collect_test_solutions'. Through this script it is also possible to print a table that shows the distribution of the nodes in an optimal path. We found that all the optimal paths found visit the same 35 nodes in a different order. 
