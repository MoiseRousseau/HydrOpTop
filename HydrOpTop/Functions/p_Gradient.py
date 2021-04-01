# This file is a part of the topology optimization program by Moise Rousseau

import numpy as np
import h5py
from .common import __cumsum_from_connection_to_array__


class p_Gradient:
  """
  Description
  """
  def __init__(self, direction="Z", tolerance=0., power=1, correction=False):
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
    self.correct_error = correction
    
    #quantities derived from the input calculated one time
    self.initialized = False #a flag indicating calculated (True) or not (False)
    self.vec_con = None
    self.k_smooth = 100000
    
    #required attribute
    self.p_ids = None 
    self.ids_p = None
    self.dobj_dP = 0. 
    self.dobj_dmat_props = [0.]*8
    self.dobj_dp_partial = None
    self.adjoint = None #attribute storing adjoint
    
    #required for problem crafting
    self.output_variable_needed = ["FACE_AREA", "CONNECTION_IDS", "VOLUME",
                                   f"FACE_NORMAL_{direction}"] 
    self.name = "p_Gradient"
    return
    
  def set_ids_to_consider(self, x):
    self.ids_to_consider = x-1
    return
    
  def set_inputs(self, inputs):
    self.areas = inputs[0]
    self.connection_ids = inputs[1]
    self.volume = inputs[2]
    self.face_normal = inputs[3]
    return
  
  def get_inputs(self):
    """
    Method that return the inputs in the same order than in "self.output_variable_needed".
    Required to pass the verification test
    """
    return [self.areas, self.connection_ids, self.volume, self.face_normal]
  
  def set_p_to_cell_ids(self,p_ids):
    """
    Method that pass the correspondance between the index in p and the
    PFLOTRAN cell ids.
    Required by the Crafter
    """
    self.p_ids = p_ids
    return 
    
  def output(self, p, f_out="p_Gradient.h5"):
    import h5py
    out = h5py.File(f_out,'w')
    grad = self.compute_p_gradient(p)
    if self.correct_error: grad -= self.error*p
    if self.direction == 0: var_name = "X"
    elif self.direction == 1: var_name = "Y"
    else: var_name = "Z"
    n_boundary = np.min(self.connection_ids)
    out.create_dataset("Cell Ids", data=np.arange(np.max(self.connection_ids)),
                       dtype='i8')
    out.create_dataset("p", data=np.where(self.ids_p >= 0, 
                                          p[self.ids_p], np.nan)[0:n_boundary-1],
                       dtype='f8')
    out.create_dataset(f"p gradient {var_name}", 
                       data=np.where(self.ids_p >= 0, 
                                     grad[self.ids_p], np.nan)[0:n_boundary-1],
                       dtype='f8')
    if self.correct_error:
      out.create_dataset(f"p gradient {var_name} Error",  
                       data=np.where(self.ids_p >= 0, 
                                     self.error[self.ids_p], np.nan)[0:n_boundary-1],
                       dtype='f8')
    obj = self.volume[self.ids_p[0:n_boundary-1] >= 0] * \
          self.smooth_max_function(grad, self.k_smooth)**self.power / self.V_total
    out.create_dataset("obj", data=np.where(self.ids_p >= 0, 
                                            obj[self.ids_p], np.nan)[0:n_boundary-1],
                       dtype='f8')
    self.d_objective_dp_partial(p)
    out.create_dataset("d_obj",  
                       data=np.where(self.ids_p >= 0, 
                             self.dobj_dp_partial[self.ids_p], np.nan)[0:n_boundary-1],
                       dtype='f8')
    out.close()
    return
  
  
  ### COST FUNCTION ###
  def compute_p_gradient(self, p):
    """
    Compute the gradient of p and return a n*3 array
    """
    if not self.initialized: self.__initialize__()
    #prepare gradient at connection for the sum
    p_i, p_j = p[self.ids_i], p[self.ids_j]
    p_con = np.where((self.vec_con < 0.) * (self.ids_j != -1), 
                     p_j, p_i)
    grad_con = self.vec_con * p_con
    #sum
    grad = np.zeros(len(p), dtype='f8') #gradient at each parametrized cell
    __cumsum_from_connection_to_array__(grad, self.ids_i,
                                        grad_con, self.sorted_connections1_to_p)
    __cumsum_from_connection_to_array__(grad, 
                    self.ids_j[self.ids_j != -1],
                    -grad_con[self.ids_j != -1], self.sorted_connections2_to_p)
    grad /= self.volume[self.p_ids-1]
    return grad
  
  
  def evaluate(self, p):
    """
    Evaluate the function
    """
    if not self.initialized: self.__initialize__()
    grad = self.compute_p_gradient(p)
    if self.correct_error: grad -= self.error*p
    grad = self.smooth_max_function(grad, self.k_smooth)
    #objective value
    cf = np.sum(self.volume[self.p_ids-1] * grad**self.power) / self.V_total - self.tol
    return cf
  
  
  ### PARTIAL DERIVATIVES ###
  def d_objective_dP(self,p): 
    return None
  
  def d_objective_d_mat_props(self, p):
    return None
  
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
    
  def d_objective_dp_partial_FD(self,p):
    if self.dobj_dp_partial is None:
      self.dobj_dp_partial = np.zeros(len(p), dtype='f8')
    else:
      self.dobj_dp_partial[:] = 0.
    obj = self.evaluate(p)
    eps = 1e-6
    for i in range(len(p)):
      p[i] += eps
      pertub_p = self.evaluate(p)
      p[i] -= 2*eps
      pertub_m = self.evaluate(p)
      p[i] += eps
      self.dobj_dp_partial[i] = 0.5*(pertub_p-pertub_m)/eps
    return
  
  def d_objective_dp_partial(self, p):
    if self.dobj_dp_partial is None:
      self.dobj_dp_partial = np.zeros(len(p), dtype='f8')
    else:
      self.dobj_dp_partial[:] = 0.
      
    #gradient value
    grad = self.compute_p_gradient(p)
    if self.correct_error: grad -= self.error*p
    smooth_grad = self.smooth_max_function(grad, self.k_smooth)
    
    Sij = self.vec_con
    n = self.power-1
    if n > 0:
      smooth_n_dsmooth = smooth_grad**n * \
                            self.d_smooth_max_function(grad,self.k_smooth)
    elif n == 0:
      smooth_n_dsmooth = self.d_smooth_max_function(grad,self.k_smooth)
    elif n < 0:
      np.seterr(invalid='ignore', divide='ignore')
      smooth_n_dsmooth = np.where(smooth_grad == 0., 
                                  0.,
                                  smooth_grad**n * \
                                  self.d_smooth_max_function(grad,self.k_smooth))
      np.seterr(invalid='warn', divide='warn')
    #i above j contribution
    temp = np.where((Sij < 0.) * (self.ids_j != -1), Sij *
      (smooth_n_dsmooth[self.ids_i] - smooth_n_dsmooth[self.ids_j]), 0.)
    __cumsum_from_connection_to_array__(self.dobj_dp_partial, 
                                        self.ids_j,
                                        temp, self.sorted_connections2_to_p)
    #j above i contribution
    temp = np.where((Sij > 0.) * (self.ids_j != -1), Sij *
      (smooth_n_dsmooth[self.ids_i] - smooth_n_dsmooth[self.ids_j]), 0.)
    __cumsum_from_connection_to_array__(self.dobj_dp_partial, 
                                        self.ids_i,
                                        temp, self.sorted_connections1_to_p)
    # bc connections
    temp = np.where((self.ids_j == -1), Sij * smooth_n_dsmooth[self.ids_i], 0.)
    __cumsum_from_connection_to_array__(self.dobj_dp_partial, 
                                        self.ids_i,
                                        temp, self.sorted_connections1_to_p)
    
    if self.correct_error: #validated
      self.dobj_dp_partial -= self.volume[self.p_ids-1] * smooth_n_dsmooth * self.error        
    self.dobj_dp_partial *= self.power / self.V_total
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
    self.V_total = np.sum(self.volume[self.p_ids-1])
    #correspondance between cell ids and p
    self.ids_p = -np.ones(np.max(self.connection_ids) - 
                      np.min(self.connection_ids)+1, dtype='i8') #-1 mean not optimized
    self.ids_p[self.p_ids-1] = np.arange(len(self.p_ids))
    #parametrized
    #mask on connection to know which to sum
    self.mask = np.isin(self.connection_ids,self.p_ids)
    self.mask = np.where(self.mask[:,0]+self.mask[:,1], True, False)
    #the cell center vector (masked)
    self.vec_con = (-self.face_normal*self.areas)[self.mask]
    #index of sorted connections
    self.ids_i = self.ids_p[self.connection_ids[:,0][self.mask]-1]
    self.ids_j = self.ids_p[self.connection_ids[:,1][self.mask]-1]
    self.sorted_connections1_to_p = np.argsort(self.ids_i[self.ids_i != -1])
    self.sorted_connections2_to_p = np.argsort(self.ids_j[self.ids_j != -1])
    
    #initialize original error
    if self.correct_error:
      self.error = self.compute_p_gradient(np.ones(len(self.p_ids),dtype='f8'))
    else:
      self.error = 0.
    return
  
  ### REQUIRED FOR CRAFTING ###
  def __require_adjoint__(self): return False 
  def __get_PFLOTRAN_output_variable_needed__(self):
    return self.output_variable_needed 
  def __get_name__(self): return self.name
  def __get_constrain_tol__(self): return self.tol
                       
