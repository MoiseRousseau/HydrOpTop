# This file is a part of the topology optimization program by Moise Rousseau

import numpy as np
import h5py
from .common import __cumsum_from_connection_to_array__


class p_Gradient:
  """
  Description
  """
  def __init__(self, direction="Z", tolerance=0.3, power=3):
    #inputs for function evaluation
    if direction.upper() not in ["X","Y","Z"]:
      print("p_Gradient direction argument not recognized")
      print("Direction should be either X, Y or Z")
      exit(1)
    self.direction = direction.upper()
    self.power = power
    self.tol = tolerance
    
    #quantities derived from the input calculated one time
    self.initialized = False #a flag indicating calculated (True) or not (False)
    self.constant = 0
    
    #required attribute
    self.p_ids = None 
    self.ids_p = None
    self.dobj_dP = 0. 
    self.dobj_dmat_props = [0.]*6 
    self.dobj_dp_partial = 0.
    self.adjoint = None #attribute storing adjoint
    
    #required for problem crafting
    self.output_variable_needed = ["FACE_AREA", "FACE_UPWIND_FRACTION", 
                                   "FACE_DISTANCE_BETWEEN_CENTER",
                                   "CONNECTION_IDS", "VOLUME",
                                   "X_COORDINATE", "Y_COORDINATE", "Z_COORDINATE"] 
    self.name = "p_Gradient"
    return
    
  def set_ids_to_consider(self, x):
    self.ids_to_consider = x-1
    return
    
  def set_inputs(self, inputs):
    self.areas = inputs[0]
    self.d_fraction = inputs[1]
    self.distance = inputs[2]
    self.connection_ids = inputs[3]
    self.volume = inputs[4]
    self.x, self.y, self.z = inputs[5:]
    return
  
  def get_inputs(self):
    """
    Method that return the inputs in the same order than in "self.output_variable_needed".
    Required to pass the verification test
    """
    return [self.areas, self.d_fraction, self.distance, self.connection_ids, 
            self.x, self.y, self.z]
  
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
    out.create_dataset("Cell Ids", data=self.p_ids, dtype='i8')
    out.create_dataset("p", data=p, dtype='f8')
    out.create_dataset("p gradient X", data=grad[:,0], dtype='f8')
    out.create_dataset("p gradient Y", data=grad[:,1], dtype='f8')
    out.create_dataset("p gradient Z", data=grad[:,2], dtype='f8')
    obj = grad[:,2] / np.sum(grad[:,2]**self.power)**(1/self.power)
    out.create_dataset("obj", data=obj, dtype='f8')
    out.close()
    return
  
  
  ### COST FUNCTION ###
  def compute_p_gradient(self, p):
    """
    Compute the gradient of p and return a n*3 array
    """
    if not self.initialized: self.__initialize__()
    #prepare gradient at connection for the sum
    p_i = p[self.ids_p[self.connection_ids[:,0][self.mask]-1]]
    #p_j = np.where(self.connection_ids[:,1][self.mask]>0, 
    #               p[self.ids_p[self.connection_ids[:,1][self.mask]-1]],
    #               p[self.ids_p[self.connection_ids[:,0][self.mask]-1]])
    p_j = p[self.ids_p[self.connection_ids[:,1][self.mask]-1]]
    p_con = p_i * self.d_fraction[self.mask] + (1-self.d_fraction[self.mask]) * p_j
    grad_con = np.zeros((np.sum(self.mask),3), dtype='f8')
    for i in range(3): grad_con[:,i] = self.vec_con[:,i] * p_con
    #sum
    grad = np.zeros((len(p),3), dtype='f8') #gradient at each parametrized cell
    __cumsum_from_connection_to_array__(grad, self.connection_ids[:,0][self.mask]-1,
                                        grad_con, self.sorted_connections1_to_p)
    __cumsum_from_connection_to_array__(grad, self.connection_ids[:,1][self.mask]-1,
                                        -grad_con, self.sorted_connections2_to_p)
    for i in range(3): grad[:,i] /= self.volume
    return grad
  
  
  def evaluate(self, p):
    """
    Evaluate the function
    """
    if not self.initialized: self.__initialize__()
    grad = self.compute_p_gradient(p)
    if self.direction == "X":
      grad_dir = grad[:,0] / np.sum(grad[:,0]**self.power)**(1/self.power)
    elif self.direction == "Y":
      grad_dir = grad[:,1] / np.sum(grad[:,1]**self.power)**(1/self.power)
    else:
      grad_dir = grad[:,2] / np.sum(grad[:,2]**self.power)**(1/self.power)
    grad_dir[grad_dir < 0] = 0
    #objective value
    cf = np.sum(grad_dir)/len(p) - self.tol
    return cf
  
  
  ### PARTIAL DERIVATIVES ###
  def d_objective_dP(self,p): 
    return None
  
  def d_objective_d_mat_props(self, p):
    return None
  
  def d_objective_dp_partial(self, p):
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
    print(f"Current {self.name}: {cf:.6e}")
    return cf
  
  
  ### INITIALIZER FUNCTION ###
  def __initialize__(self):
    """
    Initialize the derived quantities from the inputs that must be compute one 
    time only.
    """
    self.initialized = True
    #correspondance between cell ids and p
    self.ids_p = -np.ones(np.max(self.connection_ids),dtype='i8') #-1 mean not optimized
    self.ids_p[self.p_ids-1] = np.arange(len(self.p_ids))
    #mask on connection to know which to sum
    self.mask = np.isin(self.connection_ids,self.p_ids)
    self.mask = np.where(self.mask[:,0]*self.mask[:,1], True, False)
    #the cell center vector (masked)
    self.vec_con = np.zeros((np.sum(self.mask),3), dtype='f8')
    self.vec_con[:,0] = (self.x[self.connection_ids[:,1]-1] - \
                               self.x[self.connection_ids[:,0]-1])[self.mask]
    self.vec_con[:,1] = (self.y[self.connection_ids[:,1]-1] - \
                               self.y[self.connection_ids[:,0]-1])[self.mask]
    self.vec_con[:,2] = (self.z[self.connection_ids[:,1]-1] - \
                               self.z[self.connection_ids[:,0]-1])[self.mask]
    norm = np.sqrt(np.sum(self.vec_con**2, axis=1))
    for i in range(3): self.vec_con[:,i] *= self.areas[self.mask]/norm
    #index of sorted connections
    self.sorted_connections1_to_p = \
              np.argsort(self.ids_p[self.connection_ids[:,0][self.mask]-1])
    self.sorted_connections2_to_p = \
              np.argsort(self.ids_p[self.connection_ids[:,1][self.mask]-1])
    return
  
  ### REQUIRED FOR CRAFTING ###
  def __require_adjoint__(self): return False 
  def __get_PFLOTRAN_output_variable_needed__(self):
    return self.output_variable_needed 
  def __get_name__(self): return self.name
                       
