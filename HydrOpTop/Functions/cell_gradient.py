import numpy as np
from .Base_Function_class import Base_Function
from .common import __cumsum_from_connection_to_array__


class Cell_Gradient(Base_Function):
  r"""
  Calculate the mean (cell-computed) gradient of the given variable in the prescribed domain:

  .. math::

      f = \frac{1}{V_D} \sum_{i \in D} V_i ||\nabla {X_i} ||^n

  and gradient is estimated using the Green-Gauss cell-centered scheme:

  .. math::

      \nabla X_i = \frac{1}{V_i}
           \sum_{j \in \partial i} A_{ij} \boldsymbol{n_{ij}} \left[d_i X_i + (1-d_i) X_j \right]

  Required output, ``CONNECTION_IDS``,
  ``FACE_AREA``, ``FACE_UPWIND_FRACTION``, ``VOLUME``, ``Z_COORDINATE``,
  ``FACE_NORMAL_X``, ``FACE_NORMAL_Y`` and ``FACE_NORMAL_Z``.


  :param ids_to_consider: the cell ids on which to compute the mean gradient
  :type ids_to_consider: iterable
  :param power: the penalizing power ``n`` above
  :type power: float
  :param gravity: norm of the gravity vector ``g``
  :type gravity: float
  :param density: fluid density :math:`\rho`.
  :type density: float
  :param ref_pressure: reference pressure in PFLOTRAN simulation
  :type ref_pressure: float
  :param restrict_domain: an option to calculate the gradient considering only the considered cells instead considering the whole simulation. Might change the gradient calculated at the boundary of the considered cells.
  :type restrict_domain: bool
  """

  deriv_var_to_skip = [
    "CONNECTION_IDS",
    "FACE_AREA", "FACE_UPWIND_FRACTION",
    "VOLUME",
    "FACE_NORMAL_X", "FACE_NORMAL_Y", "FACE_NORMAL_Z"
  ]

  def __init__(self, cell_ids, variable, power=1., restrict_domain=False):

    super(Cell_Gradient, self).__init__()
    self.ids_to_consider = cell_ids
    self.power = power
    self.variable = variable.upper()
    self.restrict_domain = restrict_domain

    #quantities derived from the input calculated one time
    self.initialized = False #a flag indicating calculated (True) or not (False)
    self.vec_con = None

    #required for problem crafting
    self.variables_needed = [
      self.variable,
      "CONNECTION_IDS",
      "FACE_AREA", "FACE_UPWIND_FRACTION",
      "VOLUME",
      "FACE_NORMAL_X", "FACE_NORMAL_Y", "FACE_NORMAL_Z"
    ]
    self.name = f"{self.variable} Cell Gradient"
    self.cell_id_start_at = 0 #updated in place by the crafter
    return


  def set_inputs(self, inputs):
    self.inputs = inputs
    self.inputs[self.variable] = self.inputs[self.variable][self.indexes]
    self.inputs["VOLUME"] = self.inputs["VOLUME"][self.indexes]
    no_bc_connections = (self.inputs["CONNECTION_IDS"][:,0] > 0) * (self.inputs["CONNECTION_IDS"][:,1] > 0)
    self.var_data = self.inputs[self.variable]
    self.connection_ids = self.inputs["CONNECTION_IDS"][no_bc_connections]-1
    self.areas = self.inputs["FACE_AREA"][no_bc_connections]
    self.fraction = self.inputs["FACE_UPWIND_FRACTION"][no_bc_connections]
    self.volume = self.inputs["VOLUME"]
    self.normal = [self.inputs["FACE_NORMAL_"+c][no_bc_connections] for c in "XYZ"]
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
    head = self.var_data
    gradXYZ = self.compute_head_gradient(head) - self.grad_correction*head[:,np.newaxis]
    return gradXYZ

  def write_head_gradient_magnitude(self, f_out="grad.h5"):
    """
    Write head gradient calculated at every cell of the (restricted) domain
    """
    gradXYZ = self.get_head_gradient()
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
    gradXYZ = self.get_head_gradient()
    #objective value
    grad_mag = np.sqrt(gradXYZ[:,0]**2+gradXYZ[:,1]**2+gradXYZ[:,2]**2)
    cf = np.sum(self.volume[self.ids_to_consider] * grad_mag[self.ids_to_consider]**self.power) / self.V_tot
    return cf


  ### PARTIAL DERIVATIVES ###
  def d_objective(self,var, p):
    """
    Derivative according to pressure
    """
    #TODO: see what happen when grad_mag is null
    dobj = np.zeros_like(self.var_data)

    if var == self.variable:
      #gradient value
      n = self.power
      head = self.var_data
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
      __cumsum_from_connection_to_array__(dobj, self.ids_i,
                                          d_con[self.mask_ij])
      #increase of head at j on grad at i
      d_grad = self.vec_con * (1-self.fraction[:,np.newaxis])
      d_norm = np.sum(d_grad * gradXYZ[self.connection_ids[:,0],:], axis=1)
      d_con = d_norm * grad_mag[self.connection_ids[:,0]]**(n-2)
      d_con[~self.mask_i] = 0.
      if self.restrict_domain: d_con[~self.mask_restricted] = 0.
      __cumsum_from_connection_to_array__(dobj, self.ids_j,
                                          d_con[self.mask_ij])

      #increase of head at j on grad at j
      d_grad = -self.vec_con * (1-self.fraction[:,np.newaxis])
      d_norm = np.sum(d_grad * gradXYZ[self.connection_ids[:,1],:], axis=1)
      d_con = d_norm * grad_mag[self.connection_ids[:,1]]**(n-2)
      d_con[~self.mask_j] = 0.
      if self.restrict_domain: d_con[~self.mask_restricted] = 0.
      __cumsum_from_connection_to_array__(dobj, self.ids_j,
                                          d_con[self.mask_ij])
      #increase of head at i on grad at j
      d_grad = -self.vec_con * self.fraction[:,np.newaxis]
      d_norm = np.sum(d_grad * gradXYZ[self.connection_ids[:,1],:], axis=1)
      d_con = d_norm * grad_mag[self.connection_ids[:,1]]**(n-2)
      d_con[~self.mask_j] = 0.
      if self.restrict_domain: d_con[~self.mask_restricted] = 0.
      __cumsum_from_connection_to_array__(dobj, self.ids_i,
                                          d_con[self.mask_ij])

      #correction
      d_grad = - self.volume[:,np.newaxis] * self.grad_correction
      d_norm = np.sum(d_grad * gradXYZ, axis=1)
      d_con = d_norm * grad_mag**(n-2)
      dobj[self.ids_to_consider] += d_con[self.ids_to_consider]

      dobj *= n / self.V_tot
    else:
      dobj = super(Cell_Gradient, self).d_objective(var,p)
    return dobj

  def output_to_user(self):
    grad = np.linalg.norm(self.get_head_gradient(), axis=1)
    out = {
      f"{self.variable} Gradient corrected":("cell",np.arange(len(self.volume))+1, grad),
      f"Gradient correction":("cell",np.arange(len(self.volume))+1, np.linalg.norm(self.grad_correction,axis=1)),
    }
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
    else: #correct for the solver id start_at
      self.ids_to_consider = self.ids_to_consider - self.cell_id_start_at

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

  @classmethod
  def sample_instance(cls):
    # Suppose this grid
    # 1 - 2 - 3
    # |   |   |
    # 4 - 5 - 6
    # |   |   |
    # 7 - 8 - 9
    connections = np.array([
      (1, 2), (1, 4),
      (2, 3), (2, 5),
      (3, 6),
      (4, 5), (4, 7),
      (5, 6), (5, 8),
      (6, 9),
      (7, 8),
      (8, 9),
    ], dtype='i4')
    face_normal = np.array([
      [1,0,0], [0,-1,0],
      [1,0,0], [0,-1,0],
      [0,-1,0],
      [1,0,0], [0,-1,0],
      [1,0,0], [0,-1,0],
      [0,-1,0],
      [1,0,0], [1,0,0]
    ], dtype='f8')
    N = 9
    M = len(connections)
    face_normal[:,2] = np.random.random(M)
    face_normal[:,:2] *= np.random.random((M,2))
    face_normal /= np.linalg.norm(face_normal, axis=1)[:,None]
    inputs = {
      "TEST_VAR":np.random.random(N),
      "CONNECTION_IDS":connections,
      "FACE_AREA":np.random.random(M),
      "FACE_UPWIND_FRACTION":np.random.random(M),
      "VOLUME":np.random.random(N),
      "FACE_NORMAL_X":face_normal[:,0],
      "FACE_NORMAL_Y":face_normal[:,1],
      "FACE_NORMAL_Z":face_normal[:,2],
    }

    N = 9
    cell_ids = np.arange(N)
    inst1 = cls(cell_ids, "TEST_VAR", power=1.)
    inst1.set_inputs(inputs)
    inst1.indexes = np.arange(0,N)
    inst2 = cls(cell_ids, "TEST_VAR", power=2.)
    inst2.set_inputs(inputs)
    inst2.indexes = np.arange(0,N)
    return [inst1,inst2]