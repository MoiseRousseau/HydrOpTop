# This file is a part of the topology optimization program by Moise Rousseau

import numpy as np
import h5py


class Base_Function:
  """
  Base function class which implement
  """
  def __init__(self):
    self.name = "Base Function"
    self.solved_variable_needed = [] #a lilst of variables solved by the solver needed to 
                                     #calculate the function
    self.input_variable_needed = [] #a lilst of variables NOT solved by the solver needed to 
                                     #calculate the function (e.g. material properties)
    self.initialized = False
    self.adjoint = None #a variable to stored the adjoint when passed by the crafter
    return
  
  def set_adjoint_problem(self,x):
    self.adjoint = x
    return
  
  def set_inputs(self, inputs):
    """
    Method required by the problem crafter to pass the solver output
    variables to the objective
    Inputs argument have the same size and in the same order given in
    "self.output_variable_needed".
    Note that the inputs will be passed one time only (during the
    initialization), and will after be changed in-place, so that function
    will never be called again...
    """
    return
  
  def get_inputs(self):
    """
    Return the current input values
    """
    return []
  
  def set_p_to_cell_ids(self,p_ids):
    """
    Method that pass the correspondance between the index in p and the
    cell ids in the simulation.
    p_ids[X] = Y means the Xth cell parametrized corresponds to the cell index
    Y in the solver
    Also derive ids_p which is the reverse
    """
    self.p_ids = p_ids
    self.ids_p = -np.ones(np.max(p_ids),dtype='i8') #-1 mean not optimized
    self.ids_p[self.p_ids-1] = np.arange(len(self.p_ids))
    return 
    
  ### COST FUNCTION ###
  def evaluate(self, p):
    """
    Evaluate the cost function
    Return a scalar
    """
    return 0.
  
  
  ### PARTIAL DERIVATIVES ###
  def d_objective_dY(self,p): 
    """
    Evaluate the derivative of the function according to the solved output variable.
    (e.g. those in self.solved_variable_needed)
    """
    return 0.
  
  
  def d_objective_dX(self,p):
    """
    Evaluate the derivative of the function according to the non-solved variable.
    (e.g. those in self.input_variable_needed)
    """
    return [0.]
  
  
  def d_objective_dp_partial(self,p): 
    """
    Evaluate the PARTIAL derivative of the function according to the density
    parameter p.
    """
    return 0.
    
  def __initialize__(self):
    self.initialized = True
    return 
  
  
  
  
  ### REQUIRED FOR CRAFTING ###
  def __require_adjoint__(self): return "RICHARDS"
  def __get_name__(self): return self.name
                      
