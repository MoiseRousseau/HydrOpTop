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
                            
    #inputs for function evaluation
    self.X0 = None
    self.X1 = None
    
    #quantities derived from the input calculated one time
    self.initialized = False #a flag indicating calculated (True) or not (False)
    self.constant = 0
    self.d_dummy_variable = None
    
    #required attribute
    self.p_ids = None #correspondance between index in p and PFLOTRAN cell ids
                      #set by the crafter, user don't need this
    self.dobj_dP = 0. #derivative of the function wrt pressure
                      #to be passed to Adjoint class
    self.dobj_dmat_props = [0.,0.] #derivative of the function wrt mat properties
                                   #to be passed to Adjoint class
                                   #same size than self.output_variable_needed (see below)
    self.dobj_dp_partial = 0. #derivative of the function wrt material parameter
    self.adjoint = None #attribute storing adjoint
    
    #required for problem crafting
    self.output_variable_needed = ["LIQUID_PRESSURE", "DUMMY_VARIABLE"] #a list of the needed pflotran output variable
    self.name = "Template" #function name for output
    return
    
  def set_ids_to_consider(self, x):
    self.ids_to_consider = x-1
    return
    
  def set_inputs(self, inputs):
    """
    Method required by the problem crafter to pass the pflotran output
    variables to the objective
    Inputs argument have the same size and in the same order given in
    "self.output_variable_needed".
    Note that the inputs will be passed one time only, and will after be changed
    in-place, so that function will never be called again...
    """
    self.X0 = inputs[0]
    self.X1 = inputs[1]
    return
  
  def get_inputs(self):
    """
    Method that return the inputs in the same order than in "self.output_variable_needed".
    Required to pass the verification test
    """
    #note X0 was linked to inputs[0], and X1 to inputs[1]
    return [self.X0, self.X1]
  
  def set_p_to_cell_ids(self,p_ids):
    """
    Method that pass the correspondance between the index in p and the
    PFLOTRAN cell ids.
    Required by the Crafter
    """
    self.p_ids = p_ids
    #self.ids_p = -np.ones(np.max(p_ids),dtype='i8') #-1 mean not optimized
    #self.ids_p[self.p_ids-1] = np.arange(len(self.p_ids))
    #with the above, we can assess the index in p of optimized PFLOTRAN cell with 
    # p_index = self.ids_p[PFLOTRAN_index-1]
    return 
  
  def set_adjoint_problem(self, x):
    """
    Method to pass the adjoint given by the crafter or by the user
    May be deleted if not needed
    """
    self.adjoint = x
    return

  
  ### COST FUNCTION ###
  def evaluate(self, p):
    """
    Evaluate the cost function
    """
    if not self.initialized: self.__initialize__()
    cf = np.copy(p)
    if self.ids_to_consider is None: 
      return np.sum(p)
    else: 
      return np.sum(cf[self.ids_to_consider])
  
  
  ### PARTIAL DERIVATIVES ###
  def d_objective_dP(self,p): 
    """
    Evaluate the derivative of the function according to the pressure.
    Must have the size of the input problem
    """
    if not self.initialized: self.__initialize__()
    self.dobj_dP = None #to complete
    return None
  
  
  def d_objective_d_mat_props(self, p):
    """
    Derivative of the function according to input variable
    Must have the size of the input problem and ordered as in PFLOTRAN output, i.e.
    dc/dXi[0] => cell_id = 1
    dc/dXi[1] => cell_id = 2
    ...
    Argument:
    - p : the material parameter
    Return:
    - A list of the derivative in the same order they are listed in 
      self.mat_props_dependance
    No need to output the derivative according to pressure (it's d_objective_dP task)
    Note: if the variable is not parametrized in the Materials module, the derivative is 
    not considered and therefore could be zero
    """
    if not self.initialized: self.__initialize__()
    #the input variables are ["LIQUID_PRESSURE", "DUMMY_VARIABLE"]
    #so this function must return the derivative of the pressure at 0, and
    #of dummy_variable at 1.
    #however, derivative of the pressure is already set in d_objective_dP, so
    #we can omit it and just replace by None, therefore:
    self.d_dummy_variable = self.d_objective_d_dummy() #to define and complete
    self.dobj_dmat_props = [None, self.d_dummy_variable]
    return None
  
  def d_objective_dp_partial(self, p): 
    """
    PARTIAL Derivative of the function wrt the material parameter p (in input)
    """
    if not self.initialized: self.__initialize__()
    self.dobj_dp_partial = 0. #to complete
    return None
  
  
  ### TOTAL DERIVATIVE ###
  def d_objective_dp_total(self, p, out=None): 
    """
    Evaluate the TOTAL derivative of the function according to the density
    parameter p. If a numpy array is provided, derivative will be copied 
    in this array, else create a new numpy array.
    """
    if not self.initialized: self.__initialize__()
    #this method could be used as is
    if out is None:
      out = np.zeros(len(self.p), dtype='f8')
    self.d_objective_dP(p) #update function derivative wrt pressure
    self.d_objective_d_mat_props(p) #update function derivative wrt mat prop
    self.d_objective_dp_partial(p) #update function derivative wrt mat parameter p
    out[:] = self.adjoint.compute_sensitivity(p, self.dobj_dP, 
                 self.dobj_dmat_props, self.output_variable_needed) + self.dobj_dp_partial
    return out
  
  
  
  ### INITIALIZER FUNCTION ###
  def __initialize__(self):
    """
    Initialize the derived quantities from the inputs that must be compute one 
    time only.
    """
    self.initialized = True
    self.constant = np.sum(X0) #for example
    return
  
  ### REQUIRED FOR CRAFTING ###
  def __require_adjoint__(self): return False #"RICHARDS" or "TRANSPORT" or False if not required
  def __get_PFLOTRAN_output_variable_needed__(self):
    return self.output_variable_needed 
  def __get_name__(self): return self.name
                       
