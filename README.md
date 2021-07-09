# AI for TSP Competition
# Repository of the team Margaridinhas for the surrogate track (track 1)

## Graphical analysis of solutions:
Open [graphical_analysis/analysis.py](https://github.com/almeidawarley/tsp_competition/blob/master/graphical_analysis/analysis.py) and choose the desired instance and solution in the main() (by changing the variables *instance_number*, *use_validation_or_test*, *use_test*, and *sol*). Then run the file. An image analysis.png will be created.

## Dynamical programming approach:
Open [dynamic_prog/dynamic.py](https://github.com/almeidawarley/tsp_competition/blob/master/dynamic_prog/dynamic.py) and choose the desired instance using variables *instance_number*, *use_validation_or_test* and *use_test*. Then, set variable *precomputed* to *False* to compute matrices E and TT for the first time for that particular instance, and that will be stored in subfolder *dynamic_prog/store_E_and_TT*. Once those tables are stored, it will be possible to run again the same method on the same instance with *precomputed* to *True* to avoid recomputing E and TT. Finally, the last parameter is the number of steps to look ahead in the algorithm, encoded in the variable *E_dimensions*. The details of this algorithm is detailed in the appendix of the [documentation](https://github.com/almeidawarley/tsp_competition/blob/master/AI4TSP_competition_track_1_Margaridinhas.pdf). The solutions will be stored in the subfolder *dynamic_prog/dyn_solutions*.
