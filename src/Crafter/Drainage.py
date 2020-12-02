import h5py
import numpy as np
import nlopt

class Steady_State_Optimization:
  """
  Solve a hydrogeological topological optimization problem using the 
  SIMP method with PFLOTRAN
  Argument:
  - mat_props: a list of material properties that vary with the density
               parameter p (Material classes instances)
  - solver: object that manage the PDE solver (PFLOTRAN class instance)
  - objectif: the objective function (Objectif class instance)
  - constrains: a list of constrains (Constrain class instances
  """
  def __init__(self, mat_props, solver, objectif, constrains):
    self.mat_props = mat_props
    self.solver = solver
    self.obj = objectif
    self.constrains = constrains
    
    self.first = True
    self.ids_to_optimize = None #all by default
    self.Xi = None #store material properties
    self.Yi = None #store PFLOTRAN output
    return
  
  def set_cell_ids_to_optimize(self, X_ids):
    """
    Set the ids to optimize in PFLOTRAN simulation. 
    Argument:
    - X_ids: a numpy array containing the ids. Must have the same length
             than the material density parameter p
    """
    self.ids_to_optimize = X_ids
    return
    
  
  def make_one_iteration(self, p, grad):
    ### INITIALIZE DATA STRUCTURE ###
    if self.first:
      self.first = False
      self.__initialize_IO_array__()
      self.object
      
    ### UPDATE MAT PROPERTIES AND RUN PFLOTRAN ###
    #Given p, update material properties
    for i,X in enumerate(self.Xi):
      self.mat_props[i].convert_p_to_mat_properties(p,X)
      self.solver.create_cell_indexed_dataset(X, X.get_name(),
                        X_ids=self.ids_to_optimize, resize_to=True)
    #run PFLOTRAN
    self.solver.run_PFLOTRAN()
    
    ### GET PFLOTRAN OUTPUT ###
    for i,var in enumerate(self.objective.__get_PFLOTRAN_output_variable_needed__()):
      self.solver.get_output_variable(var, self.Yi[i], -1) #last timestep
    
    ### EVALUATE COST FUNCTION AND ITS DERIVATIVE ###
    # note that we have in place assignement, so we don't have to
    # update the Yi in the objective
    cf = self.objectif.evaluate()
    if grad.size > 0:
      
    return cf
    
  
  def __initialize_IO_array__(self):
    if self.ids_to_optimize is None:
      n_cells_to_opt = self.solver.n_cells
    else:
      n_cells_to_opt = len(self.ids_to_optimize)
    #initialize input for pflotran
    self.Xi = [np.zeros(n_cells_to_opt, dtype='f8') for x in self.mat_props]
    #initialize output
    n_outputs = len(self.obj.__get_PFLOTRAN_output_variable_needed__())
    self.Yi = [np.zeros(self.solver.n_cells, dtype='f8') for x in range(n_outputs)]
    return
    

