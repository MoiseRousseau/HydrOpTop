# This file is a part of the topology optimization program by Moise Rousseau

import numpy as np
import h5py
from .common import __cumsum_from_connection_to_array__
from .Base_Function_class import Base_Function

class p_Weighted_Head_Gradient(Base_Function):
  r"""
    Calculate the mean head gradient in the prescribed domain weighted by the density
    parameter `p`. In practice, can be used to just consider the mean head gradient in
    the material defined by ``p=1``:

    .. math::
       
       f = \frac{1}{V_D} \sum_{i \in D} p_i V_i ||\nabla {h_i} ||^n

    For more detail, see description of the objective function ``Head_Gradient``.
  
  :param ids_to_consider: the cell ids on which to compute the mean gradient
  :type ids_to_consider: iterable
  :param power: the penalizing power `n` above
  :type power: float
  :param restrict_domain: An option to calculate the gradient considering 
    only the considered cells instead considering the whole simulation. Might
    change the gradient calculated at the boundary of the optimization domain.
  :type restrict_domain: bool
  :param invert_weighting: Can be set to ``True`` to rather consider the mean head
    gradient in the material designed by ``p=0``.
  :type invert_weighting: bool
  
  Required PFLOTRAN outputs:
    ``LIQUID_PRESSURE``, ``CONNECTION_IDS``, 
    ``FACE_AREA``, ``FACE_UPWIND_FRACTION``, ``VOLUME``,
    ``FACE_NORMAL_X``, ``FACE_NORMAL_Y`` and ``FACE_NORMAL_Z``

  """
  def __init__(
    self, ids_to_consider="everywhere", power=1.,
    invert_weighting=False, restrict_domain=False
  ):
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
        raise RuntimeError("The argument 'ids_to_consider' must be a numpy array or a object that can be converted to")

    super(p_Weighted_Head_Gradient, self).__init__()
    self.power = power
    self.invert_weighting = invert_weighting
    self.restrict_domain = restrict_domain
    
    #quantities derived from the input calculated one time
    self.initialized = False #a flag indicating calculated (True) or not (False)
    self.vec_con = None
    
    #required for problem crafting
    self.variables_needed = [
      "LIQUID_HEAD",
      "CONNECTION_IDS",
      "FACE_AREA", "FACE_UPWIND_FRACTION",
      "VOLUME",
      "FACE_NORMAL_X", "FACE_NORMAL_Y", "FACE_NORMAL_Z"
    ]
    self.name = "p-Weighted Head Gradient"
    return

    
  def set_inputs(self, inputs):
    no_bc_connections = (inputs["CONNECTION_IDS"][:,0] > 0) * (inputs["CONNECTION_IDS"][:,1] > 0)
    self.head = inputs["LIQUID_HEAD"]
    self.connection_ids = inputs["CONNECTION_IDS"][no_bc_connections]-1
    self.fraction = inputs["FACE_UPWIND_FRACTION"][no_bc_connections]
    self.areas = inputs["FACE_AREA"][no_bc_connections]
    self.volume = inputs["VOLUME"]
    self.normal = [inputs["FACE_NORMAL_"+c][no_bc_connections] for c in "XYZ"]
    return
  
  def set_p_to_cell_ids(self,p_ids):
    """
    Method that pass the correspondance between the index in p and the
    PFLOTRAN cell ids.
    Required by the Crafter
    """
    self.p_ids = p_ids #p to PFLOTRAN index
    if self.ids_to_consider is None: #sum on all parametrized cell
      self.ids_to_consider = p_ids-1
    else: #check if all cell are parametrized
      mask = ~np.isin(self.ids_to_consider, self.p_ids-1)
      if np.sum(mask) > 0:
        print("Error! Some cell ids to consider are not parametrized (i.e. p is not defined at these cells):")
        print(self.ids_to_consider[mask]+1)
        exit(1)
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
  
  
  def evaluate(self, p):
    """
    Evaluate the function
    """
    if not self.initialized: self.__initialize__()
    if self.invert_weighting: p_ = 1-p[self.ids_to_consider_p]
    else: p_ = p[self.ids_to_consider_p]
    gradXYZ = self.compute_head_gradient(self.head) - self.grad_correction*self.head[:,np.newaxis]
    #for it in self.correction_it:
      #make correction
    #objective value
    grad_mag = np.sqrt(gradXYZ[:,0]**2+gradXYZ[:,1]**2+gradXYZ[:,2]**2)
    V = np.sum(p_*self.volume[self.ids_to_consider])
    cf = np.sum(p_*self.volume[self.ids_to_consider] * grad_mag[self.ids_to_consider]**self.power) / V
    return cf
  
  
  ### PARTIAL DERIVATIVES ###
  def d_objective(self, var, p):
    if var == "LIQUID_HEAD":
      dobj = np.zeros(len(self.head), dtype='f8')
      
      #gradient value
      gradXYZ = self.compute_head_gradient(self.head) - self.grad_correction*self.head[:,np.newaxis]
      grad_mag = np.sqrt(gradXYZ[:,0]**2+gradXYZ[:,1]**2+gradXYZ[:,2]**2)

      if self.invert_weighting: p_ = 1-p
      else: p_ = p

      #increase of head at i on grad at i
      d_grad = self.fraction[:,np.newaxis] * self.vec_con
      d_norm = np.sum(d_grad * gradXYZ[self.connection_ids[:,0]], axis=1)
      d_con = d_norm * grad_mag[self.connection_ids[:,0]]**(self.power-2)
      d_con[self.mask_i] *= p_[self.ids_p[self.connection_ids[self.mask_i,0]]]
      d_con[~self.mask_i] = 0.
      if self.restrict_domain: d_con[~self.mask_restricted] = 0.
      __cumsum_from_connection_to_array__(dobj, self.ids_i,
                                          d_con[self.mask_ij])
      #increase of head at j on grad at i
      d_grad = self.vec_con * (1-self.fraction[:,np.newaxis])
      d_norm = np.sum(d_grad * gradXYZ[self.connection_ids[:,0],:], axis=1)
      d_con = d_norm * grad_mag[self.connection_ids[:,0]]**(self.power-2)
      d_con[self.mask_i] *= p_[self.ids_p[self.connection_ids[self.mask_i,0]]]
      d_con[~self.mask_i] = 0.
      if self.restrict_domain: d_con[~self.mask_restricted] = 0.
      __cumsum_from_connection_to_array__(dobj, self.ids_j,
                                          d_con[self.mask_ij])

      #increase of head at j on grad at j
      d_grad = -self.vec_con * (1-self.fraction[:,np.newaxis])
      d_norm = np.sum(d_grad * gradXYZ[self.connection_ids[:,1],:], axis=1)
      d_con = d_norm * grad_mag[self.connection_ids[:,1]]**(self.power-2)
      d_con[self.mask_j] *= p_[self.ids_p[self.connection_ids[self.mask_j,1]]]
      d_con[~self.mask_j] = 0.
      if self.restrict_domain: d_con[~self.mask_restricted] = 0.
      __cumsum_from_connection_to_array__(dobj, self.ids_j,
                                          d_con[self.mask_ij])
      #increase of head at i on grad at j
      d_grad = -self.vec_con * self.fraction[:,np.newaxis]
      d_norm = np.sum(d_grad * gradXYZ[self.connection_ids[:,1],:], axis=1)
      d_con = d_norm * grad_mag[self.connection_ids[:,1]]**(self.power-2)
      d_con[self.mask_j] *= p_[self.ids_p[self.connection_ids[self.mask_j,1]]]
      d_con[~self.mask_j] = 0.
      if self.restrict_domain: d_con[~self.mask_restricted] = 0.
      __cumsum_from_connection_to_array__(dobj, self.ids_i,
                                          d_con[self.mask_ij])

      #correction
      d_grad = - self.volume[:,np.newaxis] * self.grad_correction
      d_norm = np.sum(d_grad * gradXYZ, axis=1)
      d_con = d_norm * grad_mag**(self.power-2)
      dobj[self.ids_to_consider] += p_[self.ids_to_consider_p] * d_con[self.ids_to_consider]

      V = np.sum(p_[self.ids_to_consider_p]*self.volume[self.ids_to_consider])
      dobj *= self.power / V
    else:
      raise NotImplementedError()
    return dobj

  
  def d_objective_dp_partial(self, p):
    dobj_dp_partial = np.zeros(len(p), dtype='f8')
    if self.invert_weighting: 
      factor = -1
      p_ = 1-p
    else: 
      factor = 1
      p_ = p
    gradXYZ = self.compute_head_gradient(self.head) - self.grad_correction*self.head[:,np.newaxis]
    grad_mag = np.sqrt(gradXYZ[:,0]**2+gradXYZ[:,1]**2+gradXYZ[:,2]**2)
    num = np.sum(p_[self.ids_to_consider_p]*self.volume[self.ids_to_consider] * grad_mag[self.ids_to_consider]**self.power)
    d_num = factor * self.volume[self.ids_to_consider] * grad_mag[self.ids_to_consider]**self.power
    den = np.sum(p_[self.ids_to_consider_p]*self.volume[self.ids_to_consider])
    d_den = factor * self.volume[self.ids_to_consider]
    dobj_dp_partial[self.ids_to_consider_p] = (d_num * den - d_den*num) / den**2
    return dobj_dp_partial
  
  
  ### INITIALIZER FUNCTION ###
  def __initialize__(self):
    """
    Initialize the derived quantities from the inputs that must be compute one 
    time only.
    """
    self.initialized = True
    self.V_tot = np.sum(self.volume[self.ids_to_consider])
    #correspondance between cell ids and p
    self.ids_p = -np.ones(np.max(self.connection_ids)+1,dtype='i8')
    self.ids_p[self.p_ids-1] = np.arange(len(self.p_ids))
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
    #index in p of ids to consider
    self.ids_to_consider_p = self.ids_p[self.ids_to_consider]
    #bc correction
    head = np.ones(len(self.volume),dtype='f8')
    self.grad_correction = self.compute_head_gradient(head)
    return
                       
