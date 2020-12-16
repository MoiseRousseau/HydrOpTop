#TODO

class Heavyside_Density_Filter:
  """
  Filter the density paramater using a three field method according to
  https://link.springer.com/article/10.1007/s00158-009-0452-7
  """
  def __init__(self, base_density_filter, cutoff=0.5, steepness = 5):
    self.base_density_filter = base_density_filter
    self.n_eta = cutoff
    self.beta = steepness
    return
  
  def get_filtered_density(self, p, p_bar = None):
    p_tild = self.base_density_filter.get_filtered_density
    return
  
  def get_filter_derivative(self, p, out=None):
    return
  

