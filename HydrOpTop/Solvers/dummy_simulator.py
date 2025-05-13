import numpy as np
from scipy.sparse import coo_matrix
import meshio


class Dummy_Simulator:
  """
  A very simple simulator for testing.
  Take two input variables for each cell/point, and return their ratio.
  In matrix form, can be write as:
  A x = b, A=diag(input[0]), b=input[1], so x = input[0]/input[1]
  
  Argument:
  - meshfile: a path to a meshio readable mesh
  - var_loc: the variable location
  """
  def __init__(self, meshfile=None, var_loc="cell", problem_size=None):
    self.meshfile = meshfile
    self.input_variables_needed = ['a','b']
    self.solved_variables_needed = ['x']
    self.var_loc = var_loc
    self.no_run = False
    
    self.input_variables_value = {}
    self.problem_size = problem_size
    return
    
  def get_grid_size(self):
    return self.problem_size
  
  def disable_run(self):
    """
    Disable running simulation (for debug purpose)
    """
    self.no_run = True
    return
  
  def get_var_location(self):
    return self.var_loc
  
  
  def create_cell_indexed_dataset(self, X_dataset, name="", outfile="",
                                        X_ids=None, resize_to=True):
    self.input_variables_value[name] = X_dataset
    if self.problem_size is None:
      self.problem_size = len(X_dataset)
    return
   
    
  def get_mesh(self):
    """
    Return the mesh in meshio format
    """
    if self.meshfile is None: 
      return [], [], []
    import meshio
    mesh = meshio.read(self.meshfile)
    if self.var_loc == "point":
      indexes = np.arange(len(mesh.points))
    else:
      indexes = np.arange(len(mesh.cells))
    return mesh.points, mesh.cells, indexes

  
  # running
  def run(self):
    """
    Compute the sum of input value
    """
    self.value = self.input_variables_value["b"]/self.input_variables_value["a"]
    return 0
  
  def get_output_variable(self, var, out=None, i_timestep=-1):
    if var == "x":
      temp = self.value
    if var in self.input_variables_value.keys():
      temp = self.input_variables_value[var]
    if out is None:
      return temp
    else:
      out[:] = temp
    return
  
  def get_sensitivity(self, var, timestep=None, coo_mat=None):
    #R = Ax-b
    if var == 'x':
      data = self.input_variables_value['a']
    elif var == "a":
      data = self.value
    elif var == "b":
      data = -np.ones(len(self.input_variables_value['b']), dtype='f8')
    
    n = np.arange(len(data))
    if coo_mat is None:
      new_mat = coo_matrix( (data,(n,n)), dtype='f8')
      return new_mat
    else:
      coo_mat.data[:] = data
    return
  
  def analytical_deriv_dy_dx(self, var):
    A = self.input_variables_value['a']
    b = self.input_variables_value['b']
    #x=b/A
    if var == "a":
      return -b/(A*A)
    elif var == 'b':
      return 1/A
    
    
