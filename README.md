# AI for TSP Competition
# Repository of the team Margaridinhas for the surrogate track (track 1)

## Graphical analysis of solutions:
Open [graphical_analysis/analysis.py](https://github.com/almeidawarley/tsp_competition/blob/master/graphical_analysis/analysis.py) and choose the desired instance and solution in the main() (by changing the variables *instance_number*, *use_validation_or_test*, *use_test*, and *sol*). Then run the file. An image analysis.png will be created.

## Dynamical programming approach:
Open [dynamic_prog/dynamic.py](https://github.com/almeidawarley/tsp_competition/blob/master/dynamic_prog/dynamic.py) and choose the desired instance using variables *instance_number*, *use_validation_or_test* and *use_test*. Then, set variable *precomputed* to *False* to compute matrices E and TT for the first time for that particular instance, and that will be stored in subfolder *dynamic_prog/store_E_and_TT*. Once those tables are stored, it will be possible to run again the same method on the same instance with *precomputed* to *True* to avoid recomputing E and TT. Finally, the last parameter is the number of steps to look ahead in the algorithm, encoded in the variable *E_dimensions*. The details of this algorithm is detailed in the appendix of the [documentation](https://github.com/almeidawarley/tsp_competition/blob/master/AI4TSP_competition_track_1_Margaridinhas.pdf). The solutions will be stored in the subfolder *dynamic_prog/dyn_solutions*.

## Tracker approach
Open [tracker.py](https://github.com/almeidawarley/tsp_competition/blob/master/tracker.py) and choose the desired value for the parameters of the tracker approach. There are four parameters: *iterations*, which determines the maximum number of iterations of the tracker approach; *simulations*, which determines the number of simulations per iteration; *feasibility_threshold*, which determines the feasibility threshold among a *simulations* number of simulations; and *gap_threshold*, which determines the gap threshold taking into consideration the upper bound at a certain iteration. The tracker approach loads by default the competition instance, i.e., the instance used in the test phase of the competition. However, it is possible to run it for other instances by changing the code accordingly. The tracker approach exports solutions to the [solutions](https://github.com/almeidawarley/tsp_competition/blob/master/solutions) folder. The details of this algorithm can be found in section 2 of the [documentation](https://github.com/almeidawarley/tsp_competition/blob/master/AI4TSP_competition_track_1_Margaridinhas.pdf).

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

(Add here an explanation on how to call the genetic algorithm)