# This file is a part of the topology optimization program by Moise Rousseau

import numpy as np
import h5py
from .common import __cumsum_from_connection_to_array__

class Head_Gradient:
  """
  Description
  """
  def __init__(self, ids_to_consider="everywhere", power=1., correction_iteration=2,
               gravity=9.8068, density=997.16, ref_pressure=101325, restrict_domain=False):
    #the correction could be more powerfull considering the iterative scheme 
    #decribed in moukalled 2016.
    #inputs for function evaluation
    if isinstance(ids_to_consider, str) and \
             ids_to_consider.lower() == "everywhere":
      self.ids_to_consider = None
    elif isinstance(ids_to_consider, np.ndarray):
      self.ids_to_consider = ids_to_consider-1
    else:
      try:
        self.ids_to_consider = np.array(ids_to_consider) -1
      except:
        print("ERROR: The argument 'ids_to_consider' must be a numpy array or a object that can be converted to")
        exit(1)
    self.power = power
    self.correction_it = correction_iteration
    self.density = density
    self.gravity = gravity
    self.ref_pressure = ref_pressure
    self.restrict_domain = restrict_domain
    
    #quantities derived from the input calculated one time
    self.initialized = False #a flag indicating calculated (True) or not (False)
    self.vec_con = None
    self.k_smooth = 100000
    
    #required attribute
    self.p_ids = None 
    self.ids_p = None
    self.dobj_dP = None
    self.dobj_dmat_props = [0.]*4
    self.dobj_dp_partial = 0.
    self.adjoint = None #attribute storing adjoint
    
    #required for problem crafting
    self.output_variable_needed = ["LIQUID_PRESSURE", "CONNECTION_IDS", 
                                   "FACE_AREA", "FACE_UPWIND_FRACTION", 
                                   "VOLUME", "Z_COORDINATE", 
                                   "FACE_NORMAL_X", "FACE_NORMAL_Y", "FACE_NORMAL_Z"] 
    self.name = "Head Gradient"
    return
    
  def set_ids_to_consider(self, x):
    self.ids_to_consider = x-1
    return
    
  def set_inputs(self, inputs):
    no_bc_connections = (inputs[1][:,0] > 0) * (inputs[1][:,1] > 0)
    self.pressure = inputs[0]
    self.connection_ids = inputs[1][no_bc_connections]-1
    self.areas = inputs[2][no_bc_connections]
    self.fraction = inputs[3][no_bc_connections]
    self.volume = inputs[4]
    self.z = inputs[5]
    self.normal = [x[no_bc_connections] for x in inputs[6:]]
    return
  
  def set_adjoint_problem(self, x):
    self.adjoint = x
    return
  
  def set_p_to_cell_ids(self,p_ids):
    """
    Method that pass the correspondance between the index in p and the
    PFLOTRAN cell ids.
    Required by the Crafter
    """
    #self.p_ids = p_ids #p to PFLOTRAN index
    if self.ids_to_consider is None: #sum on all parametrized cell
      self.ids_to_consider = np.array(p_ids)-1
    return 
  
  ### COST FUNCTION ###
  def compute_head_gradient(self, head):
    """
    Compute the gradient of the head and return a n*3 array
    """
    if not self.initialized: self.__initialize__()
    #prepare gradient at connection for the sum
    head_i, head_j = head[self.connection_ids[:,0]], head[self.connection_ids[:,1]]
    head_con = head_i * self.fraction + (1-self.fraction) * head_j
    grad_con = self.vec_con * head_con[:,np.newaxis]
    #sum
    grad = np.zeros((len(head),3), dtype='f8') #gradient at each considered cells
    if self.restrict_domain: 
      for i in range(3):
        __cumsum_from_connection_to_array__(grad[:,i], self.connection_ids[:,0][self.mask_restricted],
                                            grad_con[:,i][self.mask_restricted])
        __cumsum_from_connection_to_array__(grad[:,i], self.connection_ids[:,1][self.mask_restricted],
                                            -grad_con[:,i][self.mask_restricted])
    else: 
      for i in range(3):
        __cumsum_from_connection_to_array__(grad[:,i], self.connection_ids[:,0],
                                            grad_con[:,i])
        __cumsum_from_connection_to_array__(grad[:,i], self.connection_ids[:,1],
                                            -grad_con[:,i])
    
    grad /= self.volume[:,np.newaxis]
    return grad
  
  
  def get_head_gradient(self):
    """
    Return the XYZ head gradient at every cell considered
    """
    if not self.initialized: self.__initialize__()
    head = (self.pressure-self.ref_pressure) / (self.density * self.gravity) + self.z
    gradXYZ = self.compute_head_gradient(head) - self.grad_correction*head[:,np.newaxis]
    return gradXYZ[self.ids_to_consider]
  
  def write_head_gradient_magnitude(self, f_out="grad.h5"):
    """
    Write head gradient calculated at every cell of the (restricted) domain
    """
    if not self.initialized: self.__initialize__()
    head = (self.pressure-self.ref_pressure) / (self.density * self.gravity) + self.z
    gradXYZ = self.compute_head_gradient(head) - self.grad_correction*head[:,np.newaxis]
    #write it
    import h5py
    out = h5py.File(f_out, "w")
    out.create_dataset("Grad X", data=gradXYZ[:,0])
    out.create_dataset("Grad Y", data=gradXYZ[:,1])
    out.create_dataset("Grad Z", data=gradXYZ[:,2])
    out.create_dataset("Grad magnitude", data=np.sqrt(gradXYZ[:,0]**2+gradXYZ[:,1]**2+gradXYZ[:,2]**2))
    out.close()
    return
    
  
  def evaluate(self, p):
    """
    Evaluate the function
    """
    if not self.initialized: self.__initialize__()
    head = (self.pressure-self.ref_pressure) / (self.density * self.gravity) + self.z
    gradXYZ = self.compute_head_gradient(head) - self.grad_correction*head[:,np.newaxis]
    #for it in self.correction_it:
      #make correction
    #objective value
    grad_mag = np.sqrt(gradXYZ[:,0]**2+gradXYZ[:,1]**2+gradXYZ[:,2]**2)
    cf = np.sum(self.volume[self.ids_to_consider] * grad_mag[self.ids_to_consider]**self.power) / self.V_tot
    return cf
  
  
  ### PARTIAL DERIVATIVES ###
  def d_objective_dP(self,p):
    #TODO: see what happen when grad_mag is null
    if self.dobj_dP is None:
      self.dobj_dP = np.zeros(len(self.pressure), dtype='f8')
    else:
      self.dobj_dP[:] = 0.
      
    #gradient value
    n = self.power
    head = (self.pressure-self.ref_pressure) / (self.density * self.gravity) + self.z
    gradXYZ = self.compute_head_gradient(head) - self.grad_correction*head[:,np.newaxis]
    grad_mag = np.sqrt(gradXYZ[:,0]**2+gradXYZ[:,1]**2+gradXYZ[:,2]**2)
    
    #d_con_x_y for cell x when head at cell y is increased
    #increase of head at i on grad at i
    d_grad = self.fraction[:,np.newaxis] * self.vec_con# + \
                  #(self.volume[:,np.newaxis]*self.grad_correction)[self.connection_ids[:,0]-1]
    d_norm = np.sum(d_grad * gradXYZ[self.connection_ids[:,0]], axis=1)
    d_con = d_norm * grad_mag[self.connection_ids[:,0]]**(n-2)
    d_con[~self.mask_i] = 0.
    if self.restrict_domain: d_con[~self.mask_restricted] = 0.
    __cumsum_from_connection_to_array__(self.dobj_dP, self.ids_i,
                                        d_con[self.mask_ij])
    #increase of head at j on grad at i
    d_grad = self.vec_con * (1-self.fraction[:,np.newaxis])
    d_norm = np.sum(d_grad * gradXYZ[self.connection_ids[:,0],:], axis=1)
    d_con = d_norm * grad_mag[self.connection_ids[:,0]]**(n-2)
    d_con[~self.mask_i] = 0.
    if self.restrict_domain: d_con[~self.mask_restricted] = 0.
    __cumsum_from_connection_to_array__(self.dobj_dP, self.ids_j,
                                        d_con[self.mask_ij])
                                        
    #increase of head at j on grad at j
    d_grad = -self.vec_con * (1-self.fraction[:,np.newaxis])
    d_norm = np.sum(d_grad * gradXYZ[self.connection_ids[:,1],:], axis=1)
    d_con = d_norm * grad_mag[self.connection_ids[:,1]]**(n-2)
    d_con[~self.mask_j] = 0.
    if self.restrict_domain: d_con[~self.mask_restricted] = 0.
    __cumsum_from_connection_to_array__(self.dobj_dP, self.ids_j,
                                        d_con[self.mask_ij])
    #increase of head at i on grad at j
    d_grad = -self.vec_con * self.fraction[:,np.newaxis]
    d_norm = np.sum(d_grad * gradXYZ[self.connection_ids[:,1],:], axis=1)
    d_con = d_norm * grad_mag[self.connection_ids[:,1]]**(n-2)
    d_con[~self.mask_j] = 0.
    if self.restrict_domain: d_con[~self.mask_restricted] = 0.
    __cumsum_from_connection_to_array__(self.dobj_dP, self.ids_i,
                                        d_con[self.mask_ij])
    
    #correction
    d_grad = - self.volume[:,np.newaxis] * self.grad_correction
    d_norm = np.sum(d_grad * gradXYZ, axis=1)
    d_con = d_norm * grad_mag**(n-2)
    self.dobj_dP[self.ids_to_consider] += d_con[self.ids_to_consider] 
    
    self.dobj_dP *= n / self.V_tot / (self.density * self.gravity) 
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
      out = np.zeros(len(p), dtype='f8')
    self.d_objective_dP(p) #update function derivative wrt mat parameter p
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
    if self.ids_to_consider is None: #by default, sum everywhere
      self.ids_to_consider = np.arange(len(self.volume))
    self.V_tot = np.sum(self.volume[self.ids_to_consider])
    #correspondance between cell ids and p: not needed
    #parametrized
    #mask on connection to know which to sum
    self.mask_i = np.isin(self.connection_ids[:,0],self.ids_to_consider)
    self.mask_j = np.isin(self.connection_ids[:,1],self.ids_to_consider)
    self.mask_ij = self.mask_i + self.mask_j
    self.mask_restricted = self.mask_i * self.mask_j
    #the cell center vector
    self.face_normal = np.array(self.normal).transpose()
    self.vec_con = (-self.face_normal*self.areas[:,np.newaxis])
    #index of sorted connections
    self.ids_i = self.connection_ids[:,0][self.mask_ij]
    self.ids_j = self.connection_ids[:,1][self.mask_ij]
    #bc correction
    head = np.ones(len(self.volume),dtype='f8')
    self.grad_correction = self.compute_head_gradient(head)
    return
  
  ### REQUIRED FOR CRAFTING ###
  def __require_adjoint__(self): return "RICHARDS" 
  def __get_PFLOTRAN_output_variable_needed__(self):
    return self.output_variable_needed 
  def __get_name__(self): return self.name
  def __get_constrain_tol__(self): return self.tol
                       
