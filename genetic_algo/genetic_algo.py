import dico as DO
import genetic_operators as GO

import numpy as np
import random

from env import Env


class GeneticAlgo:
  """
  Class that contains the functions required to operate the genetic algorithm loop.

  There is the initialisation step, then the evaluation step, the selection step, 
  the reproduction step and the mutation step. After the mutation, loop these steps from
  the evaluation step, on and on, until the stopping criterion is reached.

  So far, the stopping criterion is a number of loops.
  """

  def __init__(self, nodes_num, prob_instance, population_num=4992, parents_num=158, mutation_proba=0.05, dico_filename=None):
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
    self.individus = self.initialisation()


  def initialisation(self):
    """
    Initialise the population until there are self.P individus.
    Return a list of individu (which is a list comprising a permutation of the range(1, self.N+1)).
    """
    # TODO : Initialise the population with something fancier than random individus.
    individus = []
    while len(individus) < self.P:
      ind = list(range(1, self.N+1))
      random.shuffle(ind)
      individus.append(ind)
    return individus


  def evaluation(self):
    """
    Evaluate the population individus and add their pointage to the dico.
    Store the candidates to the selection step (with the tours, pointages and fitness levels)
    in self.individus.
    """
    candidates = []
    base = [1]
    for individu in self.individus:
      tour = base + individu
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
      a = np.random.randint(0, self.K)
      b = np.random.randint(0, self.K)
      while a == b:
        b = np.random.randint(0, self.K)
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
  nodes_num = 5
  seed = 12345
  generation_num = 30

  env = Env(nodes_num, seed=seed)

  darwin = GeneticAlgo(nodes_num, env)
  darwin.evaluation()

  for i in range(0, generation_num):
    darwin.selection()
    darwin.reproduction()
    darwin.mutation()
    darwin.evaluation()
  darwin.save_progress()
