import numpy as np

class Identity:
  def __init__(self, cell_ids_to_parametrize, property_name):
    if isinstance(cell_ids_to_parametrize, str) and \
             cell_ids_to_parametrize.lower() == "all":
      self.cell_ids = None
    else:
      self.cell_ids = np.array(cell_ids_to_parametrize)
    self.name= property_name
    return
  
  
  def get_cell_ids_to_parametrize(self):
    return self.cell_ids
  
  
  def convert_p_to_mat_properties(self, p, out=None):
    if out is None: out = np.zeros(len(p),dtype='f8')
    out[:] = p
    return out
  
  
  def d_mat_properties(self, p, out=None):
    """
    Return the derivative of the material properties according to 
    material parameter p.
    """
    out = np.ones(len(p),dtype='f8')
    return out
  
  def get_name(self):
    return self.name

