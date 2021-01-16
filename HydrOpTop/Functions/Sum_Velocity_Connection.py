# This file is a part of the topology optimization program by Moise Rousseau

import numpy as np
import h5py


class Sum_Velocity_Connection:
  """
  Description
  """
  def __init__(self, ids_to_consider=None):
                     
    #objective argument
    self.ids_to_consider = ids_to_consider - 1 #be carefull here
                            #the user input the ids in PFLOTRAN indexing,
                            #(ie 1-based), while python is 0 based
    
    self.permeability = None
    self.permeability_face = None
    
    #required attribute
    self.p_ids = None #correspondance between index in p and PFLOTRAN cell ids
    self.dobj_dP = None #derivative of the obejctive wrt pressure
    self.dobj_dmat_prop = None #derivative objective wrt mat properties
    self.adjoint = None #attribute storing adjoint
    self.filter = None #store the filter object
    
    #required for problem crafting
    self.mat_props_dependance = ["PERMEABILITY"] #a list of the needed pflotran output variable
    return
    
  def set_ids_to_consider(self, x):
    self.ids_to_consider = x-1
    return
    
  def set_inputs(self, inputs):
    """
    Method required by the problem crafter to pass the pflotran output
    variables to the objective (the Yi)
    """
    self.permeability = inputs[0]
    self.velocity_at_face = inputs[1]
    self.connection_ids = inputs[3]
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
    Compute the face velocity sum in each cell ids to consider
    """
    vel_sum = 0
    for i,ids enumerate(self.connection_ids):
      id1,id2 = ids
      if id1 in self.ids_to_consider or
         id2 in self.ids_to_consider: 
        vel_sum += self.face_velocity[i]
    return vel_sum
  
  
  ### PARTIAL DERIVATIVES ###
  def d_objective_dP(self,p): 
    """
    Evaluate the derivative of the cost function according to the pressure.
    """
    #TODO
    return
  
  def d_face_permeability_dK(self):
    for id1,id2 in self.connection_ids:
      pass
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
    return ["LIQUID_PRESSURE", "PERMEABILITY", "VELOCITY_AT_FACE",
            "CONNECTION_IDS", "CONNECTION_WEIGHT"]
  def __depend_of_mat_props__(self, var=None):
    if var is None: return self.mat_props_dependance
    if var in self.mat_props_dependance: return True
    else: return False

                      
