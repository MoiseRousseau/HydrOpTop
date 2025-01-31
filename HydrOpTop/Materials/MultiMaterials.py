import numpy as np

class MultiMaterials:
  r"""
  Joint multiple material parametrization.
  
  :param :  
  """
  def __init__(self, parametrizations):
    property_name = [x.get_name() for x in parametrizations]
    if property_name.count(property_name[0]) != len(property_name):
      raise ValueError("Parametrized properties must be of the same kind")
    
    for x in parametrizations:
      for y in parametrizations:
        if x == y: continue
        cond = np.isin(
          x.get_cell_ids_to_parametrize(),
          y.get_cell_ids_to_parametrize()
        )
        if np.any(cond):
          raise ValueError(f"Cannot parametrize same cells with multiple parametrization: {x.get_cell_ids_to_parametrize()[cond]}")

    self.name = property_name[0]
    self.parametrizations = parametrizations
    #check if same cell parametrized two times
    return
  
  def get_cell_ids_to_parametrize(self):
    ret = []
    for x in self.parametrizations:
      ret += list(x.get_cell_ids_to_parametrize())
    return np.array(ret)
  
  def convert_p_to_mat_properties(self, p, out=None):
    if out is None: out = np.zeros(len(p),dtype='f8')
    start = 0
    for x in self.parametrizations:
      x.convert_p_to_mat_properties(
        p[start:start+len(x.get_cell_ids_to_parametrize())],
        out[start:start+len(x.get_cell_ids_to_parametrize())]
      )
      start += len(x.get_cell_ids_to_parametrize())
    return out
  
  def d_mat_properties(self, p, out=None):
    """
    Return the derivative of the material properties according to 
    material parameter p.
    """
    if out is None: out = np.zeros(len(p),dtype='f8')
    start = 0
    for x in self.parametrizations:
      x.d_mat_properties(
        p[start:start+len(x.get_cell_ids_to_parametrize())],
        out[start:start+len(x.get_cell_ids_to_parametrize())]
      )
      start += len(x.get_cell_ids_to_parametrize())
    return out
  
  def get_name(self):
    return self.name
