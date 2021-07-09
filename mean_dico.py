import json
import os
import random
import time

from baseline_surrogate.demo_surrogate import check_surrogate_solution

class MeanDico:
  """
  Class to stock & retrieve previously tried path scores.
  Follows this philosophy : it stores the mean score from all
  evaluations done so far + number of evaluations performed for every key.
  It means it has structure key -> [mean_score, number_of_eval].
  """


  def __init__(self, nodes_num, dico_filename=None):
    """
    Constructor that creates empty dico if dico_filename is None,
    or retrieves a dico from the dico_filename otherwise.
    Needs the number of nodes in the problem instance.
    """
    if dico_filename is None:
      # Create new empty Dico
      self.dico = {
        "1&":[0, 0]
      }
    else:
      # Retrieve Dico from filename
      a_file = open(dico_filename, "r")
      self.dico = json.load(a_file)
      a_file.close()
    
    self.N = nodes_num


  def dump(self, dico_filename=None):
    """
    Function to dump the current dico into a file.
    Will generate a datetime filename if dico_filename is left empty.
    """
    if dico_filename is None:
      # Use datetime to generate a filename
      timestr = time.strftime("%Y%m%d-%H%M%S")
      a_file = open(timestr, "w")
    else:
      # Use given filename
      a_file = open(dico_filename, "w")

    json.dump(self.dico, a_file)
    a_file.close()


  def writeEntry(self, individu, mean_pts, n_eval):
    """
    Function that will write the pts of the individu aside its entry if it exists,
    and create a new entry for this individu if it does not.
    Return the mean pts performance of that individu.
    """
    # Get the dico key associated with that individu
    keystr = self.TabToKey(individu)

    if keystr in self.dico:
      # Append the pts to the right key
      m, n = self.dico[keystr]
      weighted_score = float(m * n + mean_pts * n_eval) / (n + n_eval)
      self.dico[keystr] = [weighted_score, n + n_eval]
      return weighted_score
    else:
      # Create a new entry
      self.dico[keystr] = [mean_pts, n_eval]
      return mean_pts


  def TabToKey(self, individu):
    """
    Function that will transform an individu table into a string that includes
    significant information for a tour (càd without the stuff coming after the
    return to the base).
    """
    keystr = ""
    for g in individu:
      # For all g in the tour, append to keystr with separator character &
      keystr += str(g) + "&"
      if g == 1:
        # Return to base has happened
        break
    return keystr


  def readEntry(self, individu):
    """
    Function that will retrieve the collected pts associated with a given tour so far.
    Return a list of [mean_score, number_of_eval] if it exists, and None otherwise.
    """
    # Get the dico key associated with that individu
    keystr = self.TabToKey(individu)

    if keystr in self.dico:
      # Return the list of pts
      return self.dico[keystr]
    else:
      # Return None
      return None


  def KeyToTab(self, keystr):
    """
    Function that will transform a dico key into an individu table that includes
    all the nodes from the instance problem, but what comes after the return to
    the base has been thrown there in no particular order.
    """
    table = keystr.split("&")[:-1]
    individu = []
    flag_matrix = [0] * (self.N + 1)

    for g in table:
      # Add first the nodes that are part of the tour
      individu.append(int(g))
      flag_matrix[int(g)] = 1
    for i in range(1, len(flag_matrix)):
      # Then, complete with the nodes that will not be visited
      if flag_matrix[i] == 0:
        individu.append(i)
    return individu


  def selectEntries(self, lower_bound, tabToIndFunc=None):
    """
    Function that returns the individu-styled entries whose average pts
    is bigger than the lower_bound. TabToIndFunc is a function passed to change the
    tabs (tour without base departure) into individus.
    """
    # The baseline in this case is to not move
    selection = []
    if tabToIndFunc is None:
      return selection
    for keystr, pts_vec in self.dico.items():
      if (pts_vec[0] > lower_bound):
        selection.append(tabToIndFunc(self.KeyToTab(keystr)))
    return selection


  def selectCandidates(self, lower_bound, tabToIndFunc=None):
    """
    Function that returns the candidate-styled entries whose average pts
    is bigger than the lower_bound. TabToIndFunc is a function passed to change the
    tabs (tour without base departure) into candidates.
    """
    # The baseline in this case is to not move
    selection = []
    if tabToIndFunc is None:
      return selection
    for keystr, pts_vec in self.dico.items():
      if (pts_vec[0] > lower_bound):
        selection.append([tabToIndFunc(self.KeyToTab(keystr)), pts_vec[0], pts_vec[0]])
    return selection


  def getBestEntry(self):
    """
    Function that returns the best entry so far, in the key, value fashion.
    The value is the average value obtained by a given tour.
    """
    # The baseline in this case is to not move
    best_mean_key = "1&"
    best_mean_pts = 0
    best_mean_n = 0
    for keystr, pts_vec in self.dico.items():
      if pts_vec[0] > best_mean_pts:
        best_mean_pts = pts_vec[0]
        best_mean_n = pts_vec[1]
        best_mean_key = keystr
    return self.KeyToTab(best_mean_key), best_mean_pts, best_mean_n


def writeGoodSolToFile(list, filename) :
  """
  Write a list to a file (named filename) in the .out format
  """
  a_file = open(filename, "w")
  for node in list:
    a_file.write(str(node) + "\n")
  a_file.close()


def addBase(list) :
  """
  Add the missing base node to create the tour
  """
  return [1] + list


if __name__ == '__main__':
  """
  Little helper main function to print the best entry of a mean_dico.
  nodes_num : number of nodes in the problem instance stored in the mean_dico.
  dico_filename : what mean_dico to look at.
  """
  # PARAMETERS
  dico_filename = os.path.join(os.getcwd(), "20210707-125553-env-65-6537855_pop-25600-200_gen-4.json")
  nodes_num = 65

  dico = MeanDico(nodes_num, dico_filename)

  # This block of code returns only one good solution
  # mean_t, mean_pts, mean_n = dico.getBestEntry()
  # print("Best mean tour with " + str(mean_pts) + " pts, tested " + str(mean_n) + " times.")
  #obj = check_surrogate_solution(([1] + mean_t))
  # print("\n ---------------------------------------- \n")

  # We want to find all solutions above a certain lower bound
  mydir = "justi"
  lb = 11.3
  directory = os.path.join(os.getcwd(), mydir)
  timestr = os.path.join(os.getcwd(), mydir, time.strftime("%Y%m%d-%H%M%S") + ".txt")
  o_file = open(timestr, "w")

  selection = dico.selectEntries(lb, addBase)
  for good_sol in selection:
    # Save the solution in a file, to be able to retrieve it easily
    filestr = time.strftime("%Y%m%d-%H%M%S") + str(random.randint(100, 999)) + ".out"
    writeGoodSolToFile(good_sol, os.path.join(os.getcwd(), mydir, filestr))

    # Also check the tour for 10k times to get a good idea of actual score
    obj = check_surrogate_solution(good_sol)
    o_file.write(filestr + ", " + str(obj) + "\n")
  o_file.close()
