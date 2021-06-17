import json
import statistics
import time


class Dico:
  """
  Class to stock & retrieve previously tried path scores.
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
        "1&":[0]
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


  def writeEntry(self, individu, pts):
    """
    Function that will write the pts of the individu aside its entry if it exists,
    and create a new entry for this individu if it does not.
    Return the mean pts performance of that individu.
    """
    # Get the dico key associated with that individu
    keystr = self.TabToKey(individu)

    if keystr in self.dico:
      # Append the pts to the right key
      self.dico[keystr].append(pts)
      return statistics.mean(self.dico[keystr])
    else:
      # Create a new entry
      self.dico[keystr] = [pts]
      return pts

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
  Return a list of pts if it exists, and None otherwise.
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
  table = keystr.split("&")
  individu = []
  flag_matrix = [0] * (self.N + 1)

  for g in table:
    # Add first the nodes that are part of the tour
    individu.append(int(g))
    flag_matrix[int(g)] = 1
  for i in range(1, len(flag_matrix) + 1):
    # Then, complete with the nodes that will not be visited
    if flag_matrix[i] == 0:
      individu.append(i)
  return individu


def getBestEntry(self):
  """
  Function that returns the best entry so far, in the key, value fashion.
  The value is the average value obtained by a given tour.
  """
  # The baseline in this case is to not move
  best_key = "1&"
  best_pts = 0
  for keystr, pts_vec in self.dico.items():
    # Compute the average for every dico entry and keep the best
    mean_pts = statistics.mean(pts_vec)
    if  mean_pts < best_pts:
      best_pts = mean_pts
      best_key = keystr

  return self.KeyToTab(best_key), best_pts
