# This file is a part of the topology optimization program by Moise Rousseau

import numpy as np
import h5py


class My_Function:
  """
  Description
  """
  def __init__(self, ids_to_consider=None):
                     
    #objective argument
    self.ids_to_consider = ids_to_consider - 1 #be carefull here
                            #the user input the ids in PFLOTRAN indexing,
                            #(ie 1-based), while python is 0 based
    
    #required attribute
    self.p_ids = None #correspondance between index in p and PFLOTRAN cell ids
    self.dobj_dP = None #derivative of the obejctive wrt pressure
    self.dobj_dmat_prop = None #derivative objective wrt mat properties
    self.adjoint = None #attribute storing adjoint
    self.filter = None #store the filter object
    
    #required for problem crafting
    self.mat_props_dependance = [] #a list of the needed pflotran output variable
    return
    
  def set_ids_to_consider(self, x):
    self.ids_to_consider = x-1
    return
    
  def set_inputs(self, inputs):
    """
    Method required by the problem crafter to pass the pflotran output
    variables to the objective (the Yi)
    """
    self.X0 = inputs[0]
    self.X1 = inputs[1]
    return
  
  def set_p_cell_ids(self,p_ids):
    """
    Method that pass the correspondance between the index in p and the
    PFLOTRAN cell ids
    """
    self.p_ids = p_ids
    return 
    
  def set_filter(self, filter):
    self.filter = filter
    return
  
  def set_adjoint_problem(self, x):
    self.adjoint = x
    return

  
  ### COST FUNCTION ###
  def evaluate(self, p):
    """
    Evaluate the cost function
    """
    cf = np.copy(p)
    if self.ids_to_consider is None: 
      return np.sum(p)
    else: 
      return np.sum(cf[self.ids_to_consider])
  
  
  ### PARTIAL DERIVATIVES ###
  def d_objective_dP(self,p): 
    """
    Evaluate the derivative of the cost function according to the pressure.
    """
    #TODO
    return
  
  
  def d_objective_d_inputs(self,p):
    """
    Derivative of the objective function according to other input variable
    Argument:
    - p : the material parameter
    Return:
    - A list of the derivative in the same order they are listed in 
      self.mat_props_dependance
    Note, could return a dummy value if the objective input does not 
    depend on p explicitely or implicitely
    """
    # TODO
    return None
  
  
  
  ### TOTAL DERIVATIVE ###
  def d_objective_dp(self, p, out=None): 
    """
    Evaluate the derivative of the cost function according to the density
    parameter p. If a numpy array is provided, derivative will be copied 
    in this array, else create a new numpy array.
    """
    #this method could be used as is
    if out is None:
      out = np.zeros(len(self.p), dtype='f8')
    self.d_objective_dP(p) #update objective derivative wrt pressure
    self.d_objective_d_inputs(p) #update objective derivative wrt mat prop
    out[:] = self.adjoint.compute_sensitivity(p, self.dobj_dP, self.dobj_dmat_prop)
    if self.filter:
      out[:] = self.filter.get_filter_derivative(p).transpose().dot(out)
    return out
  
  
  ### WRAPPER FOR NLOPT ###
  def nlopt_optimize(self,p,grad):
    """
    Wrapper to evaluate and compute the derivative of the cost function
    for calling in nlopt
    """
    #could be used as is
    cf = self.evaluate(p)
    if grad.size > 0:
      self.d_objective_dp(p,grad)
    print(f"Current head sum: {cf}")
    return cf
  
  
  ### REQUIRED FOR CRAFTING ###
  def __require_adjoint__(self): return None #"RICHARDS" or "TRANSPORT"
  def __get_PFLOTRAN_output_variable_needed__(self):
    return ["LIQUID_PRESSURE", "Z_COORDINATE"]
  def __depend_of_mat_props__(self, var=None):
    if var is None: return self.mat_props_dependance
    if var in self.mat_props_dependance: return True
    else: return False

                      
