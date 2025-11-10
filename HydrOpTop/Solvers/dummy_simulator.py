import numpy as np
import scipy.sparse as sp


class Dummy_Simulator:
  """
  A very simple simulator for testing.
  Take two input variables for each cell/point, and return their ratio.
  In matrix form, can be write as:
  A * diag(a) * x = r * diag(b)
  With a and b two array, so x = r*b/A*a
  
  Argument:
  - meshfile: a path to a meshio readable mesh
  - var_loc: the variable location
  """
  def __init__(self, problem_size=None, var_loc="cell", seed=-1):
    #super(Dummy_Simulator)
    self.problem_size = 10 if problem_size is None else problem_size
    if seed >= 0: np.random.seed(seed)
    self.A_diag = np.random.normal(size=self.problem_size)
    self.r = np.random.normal(size=self.problem_size)
    self.var_loc = var_loc
    self.xyz = np.random.random((self.problem_size,3))
    
    self.input_variables_value = {
      "a": np.ones(self.problem_size),
      "b": np.ones(self.problem_size),
    }
    self.solved_variables = ['x']
    self.x = 1.
    return

  def get_region_ids(self, name):
    return np.arange(self.problem_size)

  def get_grid_size(self):
    return self.problem_size
  
  def get_system_size(self):
    return self.problem_size
  
  def get_var_location(self, var):
    return self.var_loc
  
  def get_mesh(self):
    vert = np.random.rand(self.problem_size,2)
    cells = []
    indexes = []
    return vert, cells, indexes
  
  def create_cell_indexed_dataset(self, X_dataset, name="", outfile="",
                                        X_ids=None, resize_to=True):
    self.input_variables_value[name] = X_dataset
    return

  # running
  def run(self):
    """
    Compute the sum of input value
    """
    self.x = self.r * self.input_variables_value["b"] / self.input_variables_value["a"] / self.A_diag
    return 0
  
  def get_output_variables(self, vars_out, i_timestep=-1):
    for var in vars_out.keys():
      if var == "x":
        vars_out[var] = self.x
      elif var == "ELEMENT_CENTER_X":
        vars_out[var] = self.xyz[:,0]
      elif var == "ELEMENT_CENTER_Y":
        vars_out[var] = self.xyz[:,1]
      elif var == "ELEMENT_CENTER_Z":
        vars_out[var] = self.xyz[:,2]
      elif var == "VOLUME":
        vars_out[var] = self.A_diag
      elif var in self.input_variables_value.keys():
        vars_out[var] = self.input_variables_value[var]
      else:
        raise ValueError(f"{var} not an output of the Dummy Simulator")
    return vars_out
  
  def get_sensitivity(self, var, timestep=None, coo_mat=None):
    #R = A a x - r b
    if var == 'x':
      data = self.A_diag * self.input_variables_value['a']
    elif var == "a":
      data = self.A_diag * self.x
    elif var == "b":
      data = - self.input_variables_value['b']
    else:
      raise ValueError(f"{var} does not exists in Dummy Simulator")
    
    n = np.arange(len(data))
    if coo_mat is None:
      new_mat = sp.coo_matrix( (data,(n,n)), dtype='f8')
      return new_mat
    else:
      coo_mat.data[:] = data
    return
  
  def analytical_deriv_dy_dx(self, var):
    a = self.input_variables_value['a']
    b = self.input_variables_value['b']
    #x=r b / A a
    if var == "a":
      return - b * self.r / self.A_diag / (a*a)
    elif var == 'b':
      return self.r / self.A_diag / a
    
    
