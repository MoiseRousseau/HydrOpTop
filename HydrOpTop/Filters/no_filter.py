import numpy as np

class No_Filter:
  """
  A dummy filter that do nothing but show the basic method for a filter
  """
  skip_test = True
  
  def __init__(self):
    return
  
  def get_filtered_density(self, p):
    return p.copy()
  
  def get_filter_derivative(self, p, out=None):
    return 1.
