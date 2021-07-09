import numpy as np


class GeneticOperators:
  """
  Class that implements various selection, crossing and mutation operators
  for genetic algorithms.
  """

  def __init__(self, nodes_num):
    """
    Constructor that needs the number of nodes in the problem instance.
    """
    self.N = nodes_num

  def chooseAB(self, amin, bmax):
    """
    Function that returns a and b tq amin <= a <= b < bmax.
    """
    a = np.random.randint(amin, bmax)
    b = np.random.randint(amin, bmax)
    if a <= b:
      return a, b
    else:
      return b, a


  def NWOX(self, parent1, parent2):
    """
    Non-Wrapping Ordered Crossover
    Crossing operator that creates 2 children from parent1 and parent2.
    """
    # TODO : Maybe control the span of the reserved section?
    a, b = self.chooseAB(0, len(parent1))
    child1 = []
    child2 = []
    for i in range(0, len(parent1)):
      # Copy the parent into the child if the gene is not in the reserved
      # section [a:b] of the other parent.
      if not parent1[i] in parent2[a:b]:
        child1.append(parent1[i])
      if not parent2[i] in parent1[a:b]:
        child2.append(parent2[i])
    # Insert the remaining of the reserved section at index a
    child1[a:a] = parent2[a:b]
    child2[a:a] = parent1[a:b]
    return child1, child2


  def Permutation(self, parent):
    """
    Permutation
    Mutation operator that creates a child from a parent.
    """
    # TODO Ensure permutation are effective (before return to base)
    a, b = self.chooseAB(0, len(parent))
    child = parent[:]
    child[a] = parent[b]
    child[b] = parent[a]
    return child


  def Permutation1(self, parent):
    """
    Permutation1
    Mutation operator that makes sure 1 is not at first position in individu (to call if it is the case).
    """
    b = np.random.randint(1, len(parent))
    child = parent[:]
    child[0] = parent[b]
    child[b] = parent[0]
    return child


  def BTS(self, candidates, target_k, ordering=True):
    """
    Batch Tournament Selection
    Selection operator that divides a poll of parent candidates for the next
    reproduction process.

    Candidates have a tour (list of nodes), a pointage (of the last evaluation, float)
    and a fitness (mean of all pointages obtained by that tour that have been
    saved in the dico, float).

    Two versions of BTS : the one with ordering supposedly gives a chance to
    individuals that perform especially good on one run, despite being less good on average.
    It helps with the variety of the population. The other is simply N/k batches of tournaments.
    """
    def fitnessOrdering(e):
      return e[2]
    if ordering:
      candidates.sort(key=fitnessOrdering)

    def tournament(advers):
      if len(advers) <= target_k:
        return
      else:
        # Pointage matches against two adjacent adversaries
        for i in range(1, len(advers)//2 + 1, 1):
          if advers[i-1][1] < advers[i][1]:
            advers.pop(i-1)
          else:
            advers.pop(i)
        tournament(advers)
      return
    tournament(candidates)

    # Clear the list from unecessary pts and fitness info
    selected_individus = []
    for candidate in candidates:
      selected_individus.append(candidate[0])
    return selected_individus


  def GPM(self, candidates, target_k, godparents, ordering=True):
    """
    God Parents Mixture
    Selection operator that use a tournament to find a fraction of the population, and
    use the best existing entries for another fraction (given that they exist).

    Candidates have a tour (list of nodes), a pointage (of the last evaluation, float)
    and a fitness (mean of all pointages obtained by that tour that have been
    saved in the dico, float).
    """
    def tournament(advers):
      if len(advers) <= target_k:
        return
      else:
        # Pointage matches against two adjacent adversaries
        for i in range(1, len(advers)//2 + 1, 1):
          if advers[i-1][1] < advers[i][1]:
            advers.pop(i-1)
          else:
            advers.pop(i)
        tournament(advers)
      return
    # Select the «best» candidates with tournament
    tournament(candidates)

    def fitnessOrdering(e):
      return e[2]
    if ordering:
      candidates.sort(key=fitnessOrdering, reverse=True)
    
    n_new_parents = int(target_k * 0.8)
    n_godparents = target_k - n_new_parents
    if len(godparents) >= n_godparents:
      godparents.sort(key=fitnessOrdering, reverse=True)
      godparents = godparents[:n_godparents]
      candidates = candidates[:n_new_parents] + godparents
    else:
      candidates = candidates[: len(candidates) - len(godparents)] + godparents
    assert len(candidates) == target_k, "The number of candidates does not match"

    # Clear the list from unecessary pts and fitness info
    selected_individus = []
    for candidate in candidates:
      selected_individus.append(candidate[0])
    return selected_individus
