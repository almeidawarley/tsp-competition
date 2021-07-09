import mean_dico as DO
import genetic_operators as GO

import numpy as np
import os
import random
import time

from env import Env


class GeneticAlgo:
  """
  Class that contains the functions required to operate the genetic algorithm loop.

  There is the initialisation step, then the evaluation step, the selection step, 
  the reproduction step and the mutation step. After the mutation, loop these steps from
  the evaluation step, on and on, until the stopping criterion is reached.

  So far, the stopping criterion is a number of loops.

  The result of this algorithm is stored in a json dico file that contains a dictionary with mean_dico format.
  """

  def __init__(self, nodes_num, prob_instance, population_num=4992, parents_num=158, mutation_proba=0.05, dico_filename=None, warmstart=True, wslb=6.0, monte_carlo=0):
    """
    Constructor that needs the number of nodes in the problem instance and the problem instance to use.

    Optional parameters :
      population_num (int) :   Number of individus to reach with the reproduction step.
      parents_num (int) :      Number of individus to be left after the selection step.
      mutation_proba (float) : Percentage, probability that mutation occurs to an individu during the mutation step.
      dico_filename (str) :    A filename to save the dico under a fancier name than the default datetime.
                               Is also used (if it exist) as a source of warm individus.
      warmstart (bool) :       If warmstart is true, will use content of preexisting dico_filename to grab a few
                               individus for the warm start. It will also use any solution contained in a folder called « warley ».
      wslb (float) :           Warm start lower bound : will grab from dico_filename solutions that are above wslb.
      monte_carlo (int) :      Number of times to consecutively evaluate an individu with check_solution before
                               storing in the dico.
    """
    self.N = nodes_num
    self.P = population_num
    self.K = parents_num
    self.monte_carlo = monte_carlo
    self.prob_instance = prob_instance
    self.mutation_proba = mutation_proba
    self.dico = DO.MeanDico(self.N, dico_filename)
    self.operators = GO.GeneticOperators(self.N)
    self.corps, self.queue = self.parseInvalidNodes()
    self.individus = self.initialisation(warmstart, wslb)


  def parseInvalidNodes(self):
    """
    Find the nodes that are unattainable during the tour and stock them apart.
    Returns the list of attainable nodes «corps» and the list of unattainable nodes «queue»
    """
    # corps = []
    # queue = []
    # for node in self.prob_instance.x:
    #   if node[3] >= node[6]:
    #     queue.append(node[0])
    #   else:
    #     corps.append(node[0])
    # return corps, queue

    # Warley made prospection so we know nodes that are not helping
    corps = [1, 2, 4, 5, 6, 7, 9, 11, 13, 16, 19, 22, 23, 24, 29, 30, 32, 33, 35, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 55, 57, 60, 62, 63, 64, 65]
    queue = [3, 8, 10, 12, 14, 15, 17, 18, 20, 21, 25, 26, 27, 28, 31, 34, 36, 37, 38, 39, 50, 51, 52, 53, 54, 56, 58, 59, 61]
    return corps, queue


  def initialisation(self, warmstart=True, wslb=6.0):
    """
    Initialise the population until there are self.P individus.
    Return a list of individus (an individu is a list comprising a permutation of the corps nodes).
    The corps nodes are forming a complete route when put in between the base [1] and the queue
    (unattainable nodes).
    """
    # TODO : Initialise the population with something fancier than random individus.
    individus = []
    if warmstart is True:
      # If parameter activated, download first the good individus suggested by Warley.
      # Also, pick the best mean and abs. solutions from the dico.
      individus += self.warmDicoSolutions(wslb) + self.warmWarleySolutions()
      if len(individus) > self.P:
        individus = individus[:self.P]
    while len(individus) < self.P:
      ind = self.corps[:]
      random.shuffle(ind)
      if ind[0] == 1:
        ind = self.operators.Permutation1(ind)
      assert len(ind) == len(self.corps), "Individu does not have the right size :" + str(len(ind))
      individus.append(ind)
    return individus


  def warmDicoSolutions(self, wslb=6.0):
    """
    Reads from the dico the best solutions found in the mean case (they have been evaluated
    at least « monte_carlo » times).
    Those solutions have to be better than a certain lower bound wslb.
    Return the individus list.
    """
    warm = self.dico.selectEntries(wslb, self.tourToIndividu)
    return warm


  def warmWarleySolutions(self):
    """
    Reads from a folder named warley the best solutions found by Warley and adds
    them to an array in the individu format. That means the base [1], the queue nodes
    and the nodes after the return to base [1] must be removed.
    Return the individus list.
    """
    warm = []
    directory = os.path.join(os.getcwd(), "warley")
    for entry in os.scandir(directory):
      if entry.path.endswith(".out") and entry.is_file():
        a_file = open(entry, "r")
        lines = a_file.read().splitlines()
        assert len(lines) == self.N +1, "Bad solution in Warley folder named " + entry.name
        warm.append(self.tourToIndividu(lines, True))
    return warm


  def tourToIndividu(self, tour, isString=False):
    """
    Function that transforms a full tour (with departure from base and all nodes written)
    into an individu (comprising only the corps part).
    """
    ind = []
    if isString and int(tour[0]) == 1:
      tour = tour[1:]
    for node in tour:
      if isString:
        node = int(node)
      if not node in self.queue:
        ind.append(node)
    assert len(ind) == len(self.corps), "tourToIndividu spotted bad individu of isString == " + str(isString) + " " + str(len(ind))
    return ind


  def evaluation(self):
    """
    Evaluate the population individus and add their pointage to the dico.
    Store the candidates to the selection step (with the tours, pointages and fitness levels)
    in self.individus.
    """
    candidates = []
    base = [1]
    for individu in self.individus:
      tour = base + individu + self.queue
      # Objective averaged over Monte Carlo samples, to be used for surrogate modelling
      obj = 0
      for _ in range(self.monte_carlo):
        obj_cost, rewards, pen, feas = self.prob_instance.check_solution(tour)
        obj = obj + (rewards + pen) / self.monte_carlo
      fitness = self.dico.writeEntry(individu, obj, self.monte_carlo)
      candidates.append([individu, obj, fitness])
    self.individus = candidates


  def selection(self):
    """
    Select from the candidates the lucky ones that will become parents for the next generation
    of population.
    Necessitates individus with candidate format (tours, pointages and fitness levels) to work.
    """
    select_op = "GPM"
    # TODO : Try various selection operators
    # First try : BTS
    if select_op == "BTS":
      self.individus = self.operators.BTS(self.individus, self.K)

    # Second try : GPM
    elif select_op == "GPM":
      self.individus = self.operators.GPM(
        self.individus, self.K, self.dico.selectCandidates(6.0, self.tourToIndividu))


  def reproduction(self):
    """
    Select two parents at random from the individus and cross them for reproduction purpose.
    """
    # TODO : Try various crossing operators
    children = []
    while len(children) < self.P:
      a = np.random.randint(0, len(self.individus))
      b = np.random.randint(0, len(self.individus))
      while a == b:
        b = np.random.randint(0, len(self.individus))
      child1, child2 = self.operators.NWOX(self.individus[a], self.individus[b])
      children.append(child1)
      children.append(child2)    
    self.individus = children


  def mutation(self):
    """
    Have individus mutate with self.mutation_proba probability.
    """
    for i in range(0, self.P):
      if np.random.random() <= self.mutation_proba:
        self.individus[i] = self.operators.Permutation(self.individus[i])
      if self.individus[i][0] == 1:
        self.individus[i] = self.operators.Permutation1(self.individus[i])
  

  def save_progress(self, dico_filename=None):
    """
    Save the current dico state.
    """
    self.dico.dump(dico_filename)




if __name__ == '__main__':
  """
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
  """
  # PARAMETERS
  nodes_num = 65
  seed=6537855
  generation_num = 3
  population_num = 25600
  parents_num = 200
  warm_dico_sol_lb = 11.3
  monte_carlo = 100
  dico_filename = os.path.join(os.getcwd(), "20210707-125553-env-65-6537855_pop-25600-200_gen-4.json")
  save_under = time.strftime("%Y%m%d-%H%M%S") + "-env-" + str(nodes_num) + "-" + str(seed) \
    + "_pop-" + str(population_num) + "-" + str(parents_num) + "_gen-"

  # INIT PHASE
  time1 = time.time()
  print("---------------------------------------- \n Genetic Algorithm \n ---------------------------------------- \n")
  env = Env(nodes_num, seed=seed)

  darwin = GeneticAlgo(
    nodes_num, env, population_num, parents_num,
    mutation_proba=0.05, dico_filename=dico_filename,
    warmstart=True, wslb=warm_dico_sol_lb, monte_carlo=monte_carlo)
  darwin.evaluation()
  time_elapsed1 = time.time() - time1
  print("Environment creation and population initialisation done in : ", time_elapsed1)
  print("Starting evolution process")

  # EVOLUTION PHASE
  for i in range(0, generation_num):
    time2 = time.time()

    darwin.selection()
    time_sel = time.time()
    print("End of selection    " + str(i), time_sel - time2)

    darwin.reproduction()
    time_rep = time.time()
    print("End of reproduction " + str(i), time_rep - time_sel)

    darwin.mutation()
    time_mut = time.time()
    print("End of mutation     " + str(i), time_mut - time_rep)

    darwin.evaluation()
    time_eva = time.time()
    print("End of evaluation   " + str(i), time_eva - time_mut)

    darwin.save_progress(save_under + str(i) + ".json")
    best_mean_tour, best_mean_pts, n_of_eval = darwin.dico.getBestEntry()
    print("Best mean tour so far at gen " + str(i) + " with " 
      + str(best_mean_pts) + " pts, tested " + str(n_of_eval) + " times.")
    print(*([1] + best_mean_tour), sep = ", ")
    print("\n ---------------------------------------- \n")
