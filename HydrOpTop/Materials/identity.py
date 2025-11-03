import numpy as np
from .Base_Material_class import Base_Material

class Identity(Base_Material):
  r"""
  Description:
    Create a 1 for 1 correspondance between the density parameter and the material property.
    Designed for debugging purpose.
    
  """
  def __init__(self, cell_ids_to_parametrize, property_name, bounds=None):
    if isinstance(cell_ids_to_parametrize, str) and \
             cell_ids_to_parametrize.lower() == "all":
      self.cell_ids = None
    else:
      self.cell_ids = np.array(cell_ids_to_parametrize)
    self.name= property_name
    return


  def convert_p_to_mat_properties(self, p, out=None):
    if out is None: out = np.zeros(len(p),dtype='f8')
    out[:] = p
    return out


  def d_mat_properties(self, p, out=None):
    """
    Return the derivative of the material properties according to 
    material parameter p.
    """
    if out is None: out = np.zeros(len(p),dtype='f8')
    out[:] = np.ones(len(p),dtype='f8')
    return out
