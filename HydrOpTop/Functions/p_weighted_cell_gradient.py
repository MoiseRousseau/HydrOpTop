# This file is a part of the topology optimization program by Moise Rousseau

import numpy as np
from .common import __cumsum_from_connection_to_array__
from .cell_gradient import Cell_Gradient

class p_Weighted_Cell_Gradient(Cell_Gradient):
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

  Required outputs:
    ``CONNECTION_IDS``,
    ``FACE_AREA``, ``FACE_UPWIND_FRACTION``, ``VOLUME``,
    ``FACE_NORMAL_X``, ``FACE_NORMAL_Y`` and ``FACE_NORMAL_Z``

  """
  def __init__(self, cell_ids, variable, power=1., invert_weighting=False, restrict_domain=False):
    #inputs for function evaluation

    super(p_Weighted_Cell_Gradient, self).__init__(
      cell_ids, variable, power=power, restrict_domain=restrict_domain
    )
    self.invert_weighting = invert_weighting
    self.name = f"p Weighted {self.variable} Cell Gradient"
    return


  def evaluate(self, p):
    """
    Evaluate the function
    """
    """ if self.invert_weighting: p_ = 1-p[self.ids_to_consider]
    else: p_ = p[self.ids_to_consider]
    self.volume *= p_
    self.V_tot = np.sum(self.volume[self.ids_to_consider])
    cf = super(p_Weighted_Cell_Gradient, self).evaluate(p)
    self.volume /= p_
    self.V_tot = np.sum(self.volume[self.ids_to_consider])
    return cf """

    if not self.initialized: self.__initialize__()
    if self.invert_weighting: p_ = 1-p[self.ids_to_consider]
    else: p_ = p[self.ids_to_consider]
    self.head = self.var_data
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
    if var == self.variable:
      dobj = np.zeros(len(self.head), dtype='f8')
      self.head = self.var_data

      #gradient value
      gradXYZ = self.compute_head_gradient(self.head) - self.grad_correction*self.head[:,np.newaxis]
      grad_mag = np.sqrt(gradXYZ[:,0]**2+gradXYZ[:,1]**2+gradXYZ[:,2]**2)

      if self.invert_weighting: p_ = 1-p
      else: p_ = p

      #increase of head at i on grad at i
      d_grad = self.fraction[:,np.newaxis] * self.vec_con
      d_norm = np.sum(d_grad * gradXYZ[self.connection_ids[:,0]], axis=1)
      d_con = d_norm * grad_mag[self.connection_ids[:,0]]**(self.power-2)
      d_con[self.mask_i] *= p_[self.connection_ids[self.mask_i,0]]
      d_con[~self.mask_i] = 0.
      if self.restrict_domain: d_con[~self.mask_restricted] = 0.
      __cumsum_from_connection_to_array__(dobj, self.ids_i,
                                          d_con[self.mask_ij])
      #increase of head at j on grad at i
      d_grad = self.vec_con * (1-self.fraction[:,np.newaxis])
      d_norm = np.sum(d_grad * gradXYZ[self.connection_ids[:,0],:], axis=1)
      d_con = d_norm * grad_mag[self.connection_ids[:,0]]**(self.power-2)
      d_con[self.mask_i] *= p_[self.connection_ids[self.mask_i,0]]
      d_con[~self.mask_i] = 0.
      if self.restrict_domain: d_con[~self.mask_restricted] = 0.
      __cumsum_from_connection_to_array__(dobj, self.ids_j,
                                          d_con[self.mask_ij])

      #increase of head at j on grad at j
      d_grad = -self.vec_con * (1-self.fraction[:,np.newaxis])
      d_norm = np.sum(d_grad * gradXYZ[self.connection_ids[:,1],:], axis=1)
      d_con = d_norm * grad_mag[self.connection_ids[:,1]]**(self.power-2)
      d_con[self.mask_j] *= p_[self.connection_ids[self.mask_j,1]]
      d_con[~self.mask_j] = 0.
      if self.restrict_domain: d_con[~self.mask_restricted] = 0.
      __cumsum_from_connection_to_array__(dobj, self.ids_j,
                                          d_con[self.mask_ij])
      #increase of head at i on grad at j
      d_grad = -self.vec_con * self.fraction[:,np.newaxis]
      d_norm = np.sum(d_grad * gradXYZ[self.connection_ids[:,1],:], axis=1)
      d_con = d_norm * grad_mag[self.connection_ids[:,1]]**(self.power-2)
      d_con[self.mask_j] *= p_[self.connection_ids[self.mask_j,1]]
      d_con[~self.mask_j] = 0.
      if self.restrict_domain: d_con[~self.mask_restricted] = 0.
      __cumsum_from_connection_to_array__(dobj, self.ids_i,
                                          d_con[self.mask_ij])

      #correction
      d_grad = - self.volume[:,np.newaxis] * self.grad_correction
      d_norm = np.sum(d_grad * gradXYZ, axis=1)
      d_con = d_norm * grad_mag**(self.power-2)
      dobj[self.ids_to_consider] += p_[self.ids_to_consider] * d_con[self.ids_to_consider]

      V = np.sum(p_[self.ids_to_consider]*self.volume[self.ids_to_consider])
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
    self.head = self.var_data
    gradXYZ = self.compute_head_gradient(self.head) - self.grad_correction*self.head[:,np.newaxis]
    grad_mag = np.sqrt(gradXYZ[:,0]**2+gradXYZ[:,1]**2+gradXYZ[:,2]**2)
    num = np.sum(p_[self.ids_to_consider]*self.volume[self.ids_to_consider] * grad_mag[self.ids_to_consider]**self.power)
    d_num = factor * self.volume[self.ids_to_consider] * grad_mag[self.ids_to_consider]**self.power
    den = np.sum(p_[self.ids_to_consider]*self.volume[self.ids_to_consider])
    d_den = factor * self.volume[self.ids_to_consider]
    dobj_dp_partial[self.ids_to_consider] = (d_num * den - d_den*num) / den**2
    return dobj_dp_partial


  @classmethod
  def sample_instance(cls):
    # Get parameter from the super class
    insts_super = Cell_Gradient.sample_instance()
    inputs = insts_super[0].inputs
    # create tests
    N = 9
    cell_ids = np.arange(N)
    inst1 = cls(cell_ids, "TEST_VAR", power=1.)
    inst1.set_inputs(inputs)
    inst2 = cls(cell_ids, "TEST_VAR", power=2.)
    inst2.set_inputs(inputs)
    inst3 = cls(cell_ids, "TEST_VAR", power=1., invert_weighting=True)
    inst3.set_inputs(inputs)
    return [inst1, inst2, inst3]
