# This file is a part of the topology optimization program by Moise Rousseau

import numpy as np
import h5py
from .common import __cumsum_from_connection_to_array__


class p_Difference_Connection:
  """
  Description
  """
  def __init__(self, direction="Z", tolerance=0., power=1):
    #inputs for function evaluation
    if direction.upper() not in ["X","Y","Z"]:
      print("p_Gradient direction argument not recognized")
      print("Direction should be either X, Y or Z")
      exit(1)
    if direction == "X": self.direction = 0
    elif direction == "Y": self.direction = 1
    else: self.direction = 2
    self.power = power
    self.tol = tolerance
    
    #quantities derived from the input calculated one time
    self.initialized = False #a flag indicating calculated (True) or not (False)
    self.area_proj = None
    self.area_tot = None
    self.ids_j, self.ids_i = None, None
    self.k_smooth = 100000
    
    #required attribute
    self.p_ids = None 
    self.ids_p = None
    self.dobj_dP = 0. 
    self.dobj_dmat_props = [0.]
    self.dobj_dp_partial = None
    self.adjoint = None #attribute storing adjoint
    
    #required for problem crafting
    self.output_variable_needed = ["FACE_AREA", "CONNECTION_IDS",
                                   f"FACE_NORMAL_{direction}"]
    self.name = "p_Difference_Connection"
    return
    
  def set_ids_to_consider(self, x):
    self.ids_to_consider = x-1
    return
    
  def set_inputs(self, inputs):
    self.areas = inputs[0]
    self.connection_ids = inputs[1]
    self.face_normal_component = inputs[2]
    return
  
  def get_inputs(self):
    """
    Method that return the inputs in the same order than in "self.output_variable_needed".
    Required to pass the verification test
    """
    return [self.areas, self.distance, self.connection_ids, 
            self.face_normal_component]
  
  def set_p_to_cell_ids(self,p_ids):
    """
    Method that pass the correspondance between the index in p and the
    PFLOTRAN cell ids.
    Required by the Crafter
    """
    self.p_ids = p_ids
    return 
    
  
  def smooth_max_function(self, x, k=10000):
    cutoff = 100 / k
    res = x.copy()
    res[x < cutoff] = 1/k*np.log(1+np.exp(k*x[x < cutoff]))
    return res
  
  def d_smooth_max_function(self, x, k=10000):
    cutoff = 100 / k
    res = np.ones(len(x), dtype='f8')
    res[x < cutoff] = 0.
    res[abs(x) < cutoff] = 1/(1+np.exp(-k*x[abs(x) < cutoff]))
    return res
  
  
  ### COST FUNCTION ###
  def evaluate(self, p):
    """
    Evaluate the function
    """
    if not self.initialized: self.__initialize__()
    d_p_con = p[self.ids_j] - p[self.ids_i]
    d_p_con = self.smooth_max_function(self.area_proj * d_p_con, self.k_smooth)
    #objective value
    cf = np.sum(d_p_con**self.power) / self.area_tot - self.tol
    return cf
  
  
  ### PARTIAL DERIVATIVES ###
  def d_objective_dP(self,p): 
    return None
  
  def d_objective_dp_partial(self, p):
    if not self.initialized: self.__initialize__()
    if self.dobj_dp_partial is None:
      self.dobj_dp_partial = np.zeros(len(p), dtype='f8')
    else:
      self.dobj_dp_partial[:] = 0.
      
    d_p_con = p[self.ids_j] - p[self.ids_i]
    d_p_con_smooth = self.smooth_max_function(self.area_proj * d_p_con, self.k_smooth)
    
    n = self.power-1
    if n > 0:
      temp = d_p_con_smooth**n * \
                            self.d_smooth_max_function(d_p_con,self.k_smooth)
    elif n == 0:
      temp = self.d_smooth_max_function(d_p_con,self.k_smooth)
    elif n < 0:
      np.seterr(invalid='ignore', divide='ignore')
      temp = np.where(d_p_con_smooth == 0., 
                                  0.,
                                  d_p_con_smooth**n * \
                                  self.d_smooth_max_function(d_p_con,self.k_smooth))
      np.seterr(invalid='warn', divide='warn')
    
    for i in range(len(temp)): 
      if abs(self.area_proj[i]) < 0: print(self.area_proj[i], smooth_n_dsmooth[i], self.p_ids[self.ids_i[i]], self.p_ids[self.ids_j[i]], p[self.ids_i[i]])
    __cumsum_from_connection_to_array__(self.dobj_dp_partial, 
                                        self.ids_i,
                                        -temp, self.sorted_connections1_to_p)
    __cumsum_from_connection_to_array__(self.dobj_dp_partial, 
                                        self.ids_j,
                                        temp, self.sorted_connections2_to_p)
    self.dobj_dp_partial *= self.power / self.area_tot
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
      out = np.zeros(len(p), dtype='f8')
    self.d_objective_dp_partial(p) #update function derivative wrt mat parameter p
    out[:] = self.dobj_dp_partial
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
      self.d_objective_dp_total(p,grad)
    print(f"Current {self.name}: {cf+self.tol:.6e}")
    return cf
  
  
  ### INITIALIZER FUNCTION ###
  def __initialize__(self):
    """
    Initialize the derived quantities from the inputs that must be compute one 
    time only.
    """
    self.initialized = True
    #correspondance between cell ids and p
    self.ids_p = -np.ones(np.max(self.connection_ids) - 
                      np.min(self.connection_ids)+1, dtype='i8') #-1 mean not optimized
    self.ids_p[self.p_ids-1] = np.arange(len(self.p_ids))
    #parametrized
    #mask on connection to know which to sum
    self.mask = np.isin(self.connection_ids,self.p_ids)
    self.mask = self.mask[:,0]*self.mask[:,1]
    #the area proj vector (masked)
    self.area_proj = (-self.face_normal_component*self.areas)[self.mask]
    self.area_tot = np.sum(np.abs(self.area_proj))
    #index of sorted connections
    self.ids_i = self.ids_p[self.connection_ids[:,0][self.mask]-1]
    self.ids_j = self.ids_p[self.connection_ids[:,1][self.mask]-1]
    self.sorted_connections1_to_p = np.argsort(self.ids_i)
    self.sorted_connections2_to_p = np.argsort(self.ids_j)
    
  
  ### REQUIRED FOR CRAFTING ###
  def __require_adjoint__(self): return False 
  def __get_PFLOTRAN_output_variable_needed__(self):
    return self.output_variable_needed 
  def __get_name__(self): return self.name
                       
