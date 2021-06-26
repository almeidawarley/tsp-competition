import dico as DO
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
  """

  def __init__(self, nodes_num, prob_instance, population_num=4992, parents_num=158, mutation_proba=0.05, dico_filename=None, warmstart=True, wslb=6.0):
    """
    Constructor that needs the number of nodes in the problem instance. 

    Optional parameters are the population_num to reach with the reproduction step, the parents_num
    to be left after the selection step, and the dico_filename to save under a fancier name than the
    default datetime.
    """
    self.N = nodes_num
    self.P = population_num
    self.K = parents_num
    self.prob_instance = prob_instance
    self.mutation_proba = mutation_proba
    self.dico = DO.Dico(self.N, dico_filename)
    self.operators = GO.GeneticOperators(self.N)
    self.corps, self.queue = self.parseInvalidNodes()
    self.individus = self.initialisation(warmstart, wslb)


  def parseInvalidNodes(self):
    """
    Find the nodes that are unattainable during the tour and stock them apart.
    Returns the list of attainable nodes «corps» and the list of unattainable nodes «queue»
    """
    corps = []
    queue = []
    for node in self.prob_instance.x:
      if node[3] >= node[6]:
        queue.append(node[0])
      else:
        corps.append(node[0])
    return corps, queue


  def initialisation(self, warmstart=True, wslb=6.0):
    """
    Initialise the population until there are self.P individus.
    Return a list of individus (an individu is a list comprising a permutation of the corps nodes).
    """
    # TODO : Initialise the population with something fancier than random individus.
    individus = []
    if warmstart is True:
      # If parameter activated, download first the good individus suggested by Warley.
      # Also, pick the best mean and abs. solutions from the dico.
      individus += self.warmDicoSolutions(wslb) + self.warmWarleySolutions()
    while len(individus) < self.P:
      ind = self.corps[:]
      random.shuffle(ind)
      if ind[0] == 1:
        ind = self.operators.Permutation1(ind)
      individus.append(ind)
    return individus


  def warmDicoSolutions(self, wslb=6.0):
    """
    Reads from the dico the best solutions found both in the abs. case and in the mean case.
    Those solutions have to be better than a certain lower bound wslb.
    Return the individus list.
    """
    warm = self.dico.selectEntries(wslb, self.tourToIndividu)
    return warm


  def warmWarleySolutions(self):
    """
    Reads from a folder named warley the best solutions found by Warley and adds
    them to an array in the individu format. That means the first one, the queue nodes
    and the nodes after the second one must be removed.
    Return the individus list.
    """
    warm = []
    directory = os.path.join(os.getcwd(), "warley")
    for entry in os.scandir(directory):
      if entry.path.endswith(".out") and entry.is_file():
        a_file = open(entry, "r")
        lines = a_file.read().splitlines()
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
      obj_cost, rewards, pen, feas = self.prob_instance.check_solution(tour)
      fitness = self.dico.writeEntry(individu, rewards + pen)
      candidates.append([individu, rewards + pen, fitness])
    self.individus = candidates


  def selection(self):
    """
    Select from the candidates the lucky ones that will become parents for the next generation
    of population.
    Necessitates individus with candidate format (tours, pointages and fitness levels) to work.
    """
    # TODO : Try various selection operators
    self.individus = self.operators.BTS(self.individus, self.K)


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
  nodes_num : number of nodes in the problem instance
  seed : random seed
  generation_num : number of population generation, used as stopping criterion.
  """
  # PARAMETERS
  nodes_num = 55
  seed = 3119615
  generation_num = 50
  population_num = 40000
  parents_num = 625
  warm_dico_sol_lb = 5.0
  dico_filename = os.path.join(os.getcwd(), "env-55-3119615_pop-40000-625_gen-49.json")
  save_under = time.strftime("%Y%m%d-%H%M%S") + "-env-" + str(nodes_num) + "-" + str(seed) \
    + "_pop-" + str(population_num) + "-" + str(parents_num) + "_gen-"

  # INIT PHASE
  time1 = time.time()
  print("---------------------------------------- \n Genetic Algorithm \n ---------------------------------------- \n")
  env = Env(nodes_num, seed=seed)

  darwin = GeneticAlgo(
    nodes_num, env, population_num, parents_num,
    mutation_proba=0.05, dico_filename=dico_filename, warmstart=True, wslb=warm_dico_sol_lb)
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

    if (i+1) % 10 == 0:
      darwin.save_progress(save_under + str(i) + ".json")
    best_mean_tour, best_mean_pts, best_tour, best_pts = darwin.dico.getBestEntry()
    print("Best mean tour so far at gen " + str(i) + " with " + str(best_mean_pts) + " pts.")
    print(*([1] + best_mean_tour), sep = ", ")
    print("Best abs. tour so far at gen " + str(i) + " with " + str(best_pts) + " pts.")
    print(*([1] + best_tour), sep = ", ")
    print("\n ---------------------------------------- \n")
